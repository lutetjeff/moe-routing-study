#!/usr/bin/env bash
# End-to-end run: boot vLLM + routing proxy, run pier on the curated
# subset with a wall-clock watchdog, then assemble a tarball.
#
# Usage: run.sh <N> [JOB_NAME]    where N in {1, 10, 20}
#
# Budgets (overridable via env JOB_BUDGET_SEC / TASK_BUDGET_SEC):
#   N=1   -> JOB_BUDGET_SEC=900    TASK_BUDGET_SEC=540
#   N=10  -> JOB_BUDGET_SEC=5400   TASK_BUDGET_SEC=540
#   N=20  -> JOB_BUDGET_SEC=10800  TASK_BUDGET_SEC=540
set -euo pipefail
: "${PLAYBOOK_DIR:?}"; : "${WORK_DIR:?}"; : "${PY:?}"; : "${MODEL_ID:?}"

N="${1:-1}"
JOB_NAME="${2:-$(date +%Y%m%d-%H%M%S)-n${N}}"
case "$N" in
  1)  JOB_BUDGET_SEC_DEF=900   ;;
  10) JOB_BUDGET_SEC_DEF=5400  ;;
  20) JOB_BUDGET_SEC_DEF=10800 ;;
  *)  JOB_BUDGET_SEC_DEF=$(( 540 * N + 300 )) ;;
esac
JOB_BUDGET_SEC="${JOB_BUDGET_SEC:-$JOB_BUDGET_SEC_DEF}"
TASK_BUDGET_SEC="${TASK_BUDGET_SEC:-540}"
CONCURRENCY="${CONCURRENCY:-1}"   # serial by default; tunable knob

VLLM_PORT="${VLLM_PORT:-8000}"
PROXY_PORT="${PROXY_PORT:-8001}"
JOB_DIR="$WORK_DIR/runs/$JOB_NAME"
mkdir -p "$JOB_DIR/logs"

log() { printf '[run] %s\n' "$*" >&2; }
fail() { printf 'RUN: FAIL %s\n' "$*"; exit 1; }
banner() { printf '\n===== %s =====\n' "$*" >&2; }

# Auto-derive per-model defaults so this script works across vLLM recipes
# (Qwen, Mixtral, DeepSeek, MiniMax, etc.) without per-model edits.
# All four can be overridden via env vars set by the caller.
DEFAULTS_JSON="$("$PY" "$PLAYBOOK_DIR/derive_model_defaults.py" "$MODEL_ID" 2>/dev/null || echo '{}')"
get_default() {
  printf '%s' "$DEFAULTS_JSON" | "$PY" -c "import json,sys; d=json.loads(sys.stdin.read() or '{}'); print(d.get('$1') or '')"
}
MODEL_ALIAS="${MODEL_ALIAS:-$(get_default alias)}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-$(get_default tool_call_parser)}"
MODEL_CLASS="${MODEL_CLASS:-litellm}"
PROXY_BASE_URL_FROM_CONTAINER="${PROXY_BASE_URL_FROM_CONTAINER:-http://172.17.0.1:${PROXY_PORT}/v1}"
[ -n "$MODEL_ALIAS" ] || MODEL_ALIAS="$(printf '%s' "$MODEL_ID" | awk -F/ '{print tolower($NF)}')"
if [ -z "$TOOL_CALL_PARSER" ] || [ "$TOOL_CALL_PARSER" = "unknown" ]; then
  fail "could not auto-detect tool-call parser for $MODEL_ID. Set TOOL_CALL_PARSER=... explicitly. (raw probe: $DEFAULTS_JSON)"
fi

# Context length. The probe surfaces ``max_position_embeddings`` from the
# HF config (e.g. 262144 for Qwen3.5-MoE), which is the model's upper
# bound. We DO NOT auto-jump to that — long contexts cost KV-cache
# memory and a poorly chosen max-len will OOM mid-run. The user provides
# the right value for their GPU; we fall back to 16384 if they don't.
PROBE_MAX_POS="$(get_default max_position_embeddings)"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
log "model_id=$MODEL_ID"
log "model_alias=$MODEL_ALIAS"
log "tool_call_parser=$TOOL_CALL_PARSER"
log "model_class=$MODEL_CLASS"
log "container_proxy_url=$PROXY_BASE_URL_FROM_CONTAINER"
log "max_model_len=$MAX_MODEL_LEN  (model max: ${PROBE_MAX_POS:-unknown}; override with MAX_MODEL_LEN=...)"

# 0. cleanup hooks ---------------------------------------------------------
VLLM_PID=""
PROXY_PID=""
cleanup() {
  log "tearing down..."
  for pid in "$PROXY_PID" "$VLLM_PID"; do
    [ -z "$pid" ] && continue
    kill -0 "$pid" 2>/dev/null || continue
    kill -TERM "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 30); do
    still=0
    for pid in "$PROXY_PID" "$VLLM_PID"; do
      [ -z "$pid" ] && continue
      kill -0 "$pid" 2>/dev/null && still=1
    done
    [ "$still" -eq 0 ] && break
    sleep 1
  done
  for pid in "$PROXY_PID" "$VLLM_PID"; do
    [ -z "$pid" ] && continue
    kill -KILL "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

# 1. prepare curated dataset -----------------------------------------------
case "$N" in
  1|10|20) curated_list="$PLAYBOOK_DIR/config/curated-${N}.txt" ;;
  *) curated_list="$PLAYBOOK_DIR/config/curated-20.txt" ;;
esac
[ -f "$curated_list" ] || fail "curated list missing: $curated_list"

DATASET_DIR="$JOB_DIR/dataset"
"$PY" "$PLAYBOOK_DIR/prepare_curated.py" \
  --src "$WORK_DIR/deep-swe/tasks" \
  --list "$curated_list" \
  --out "$DATASET_DIR" \
  --task-timeout-sec "$TASK_BUDGET_SEC" \
  >"$JOB_DIR/logs/prepare.log" 2>&1 || fail "prepare_curated failed (see $JOB_DIR/logs/prepare.log)"
log "curated dataset: $DATASET_DIR ($(ls "$DATASET_DIR" | wc -l) entries)"

# 2. boot vLLM (with capture) ----------------------------------------------
banner "starting vLLM"
# --served-model-name aliases the loaded model so the OpenAI client can
# address it by a short name (the agent YAML uses MODEL_ALIAS).
#
# mini-swe-agent sends `tool_choice: auto` on every turn; vLLM rejects
# that unless a parser is configured. TOOL_CALL_PARSER is auto-detected
# by family in derive_model_defaults.py and overridable from the env.
"$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$MODEL_ID" "$MODEL_ALIAS" \
  --port "$VLLM_PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
  --enable-return-routed-experts \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  ${EXTRA_VLLM_ARGS:-} \
  >"$JOB_DIR/logs/vllm.log" 2>&1 &
VLLM_PID=$!
log "vllm pid=$VLLM_PID port=$VLLM_PORT log=$JOB_DIR/logs/vllm.log"

# 3. boot proxy ------------------------------------------------------------
banner "starting routing proxy"
PROXY_VENV="$WORK_DIR/proxy-venv"
[ -x "$PROXY_VENV/bin/python" ] || fail "proxy venv missing — run setup.sh first"
ROUTING_OUTPUT_DIR="$JOB_DIR" \
  ROUTING_JOB_ID="$JOB_NAME" \
  VLLM_BASE="http://127.0.0.1:$VLLM_PORT" \
  PROXY_PORT="$PROXY_PORT" \
  "$PROXY_VENV/bin/python" "$PLAYBOOK_DIR/routing_proxy.py" \
  >"$JOB_DIR/logs/proxy.log" 2>&1 &
PROXY_PID=$!
log "proxy pid=$PROXY_PID port=$PROXY_PORT log=$JOB_DIR/logs/proxy.log"

# 4. wait for both healthy --------------------------------------------------
banner "waiting for services (max 5 min)"
deadline=$(( $(date +%s) + 300 ))
v_ok=0; p_ok=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  if [ "$v_ok" -eq 0 ] && curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
    v_ok=1; log "vllm healthy"
  fi
  if [ "$v_ok" -eq 1 ] && curl -fsS "http://127.0.0.1:$PROXY_PORT/health" >/dev/null 2>&1; then
    p_ok=1; log "proxy healthy"; break
  fi
  sleep 3
done
[ "$v_ok" -eq 1 ] || fail "vllm never became healthy (see logs/vllm.log)"
[ "$p_ok" -eq 1 ] || fail "proxy never became healthy (see logs/proxy.log)"

# 5. run pier under a hard wall-clock watchdog -----------------------------
banner "running pier (job budget ${JOB_BUDGET_SEC}s, task budget ${TASK_BUDGET_SEC}s)"
PIER="$(command -v pier || echo "$HOME/.local/bin/pier")"
[ -x "$PIER" ] || fail "pier missing"

PIER_LOG="$JOB_DIR/logs/pier.log"

# Materialize the pier JobConfig from the template with this run's values.
# Pier does NOT env-interpolate YAML, so the substitution has to happen
# before invocation. Per-job copy goes under the job dir so the
# trial-replay tooling can find the exact config that was used.
PIER_CFG="$JOB_DIR/pier-mini-vllm.yaml"
sed \
  -e "s|__MODEL_ALIAS__|${MODEL_ALIAS}|g" \
  -e "s|__API_BASE__|${PROXY_BASE_URL_FROM_CONTAINER}|g" \
  -e "s|__MODEL_CLASS__|${MODEL_CLASS}|g" \
  "$PLAYBOOK_DIR/config/pier-mini-vllm.yaml.template" >"$PIER_CFG"
log "materialized pier config: $PIER_CFG"

# Note: pier replaces config.agents when --agent is passed on the CLI, so we
# rely entirely on the JobConfig YAML for agent setup. Just override the
# dataset path (-p) and the few output / concurrency knobs from CLI.
set +e
timeout --signal=TERM --kill-after=60 "${JOB_BUDGET_SEC}s" \
  "$PIER" run \
    -p "$DATASET_DIR" \
    --env docker \
    -c "$PIER_CFG" \
    --job-name "$JOB_NAME" \
    --jobs-dir "$JOB_DIR/pier-jobs" \
    --n-concurrent "$CONCURRENCY" \
    >"$PIER_LOG" 2>&1
pier_rc=$?
set -e
log "pier exit=$pier_rc"

# 6. analysis ---------------------------------------------------------------
banner "building analysis report"
"$PROXY_VENV/bin/python" "$PLAYBOOK_DIR/analysis/build_report.py" \
  --job-dir "$JOB_DIR" \
  --out "$JOB_DIR/report" \
  >"$JOB_DIR/logs/analysis.log" 2>&1 || log "analysis script returned non-zero (see logs/analysis.log)"

# 7. tarball ----------------------------------------------------------------
banner "packing artifacts"
tar -C "$WORK_DIR/runs" -czf "$WORK_DIR/runs/$JOB_NAME.tar.gz" "$JOB_NAME" 2>/dev/null \
  && log "wrote $WORK_DIR/runs/$JOB_NAME.tar.gz" \
  || log "tarball failed (ignored)"

if [ "$pier_rc" -eq 124 ] || [ "$pier_rc" -eq 137 ]; then
  echo "RUN: PARTIAL job=$JOB_NAME reason=job_budget_hit"
elif [ "$pier_rc" -ne 0 ]; then
  echo "RUN: PARTIAL job=$JOB_NAME reason=pier_exit_$pier_rc"
else
  echo "RUN: OK job=$JOB_NAME"
fi
