#!/usr/bin/env bash
# Boot vLLM with capture, send one tiny prompt, validate the routed_experts
# blob comes back with the expected shape. Tears everything down before
# exiting. Designed to take < 5 minutes including weight load; if it takes
# longer that's the operator agent's signal to ABORT.
#
# Inputs (env):
#   PLAYBOOK_DIR, WORK_DIR, PY, MODEL_ID
# Outputs:
#   prints `SMOKE: OK shape=(...)` or `SMOKE: FAIL reason=...` to stdout.

set -euo pipefail
: "${PLAYBOOK_DIR:?}"; : "${WORK_DIR:?}"; : "${PY:?}"; : "${MODEL_ID:?}"

VLLM_PORT="${VLLM_PORT:-8000}"
LOG="$WORK_DIR/logs/smoke-vllm.log"
PIDFILE="$WORK_DIR/.smoke-vllm.pid"

log() { printf '[smoke] %s\n' "$*" >&2; }

cleanup() {
  if [ -f "$PIDFILE" ]; then
    pid="$(cat "$PIDFILE" || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      log "killing vllm pid=$pid"
      kill -TERM "$pid" 2>/dev/null || true
      for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
  fi
}
trap cleanup EXIT INT TERM

# Boot vLLM with capture flag. Conservative memory so this works inside a
# container without surprising us; full run.sh uses better defaults.
log "starting vllm serve $MODEL_ID on :$VLLM_PORT (logs: $LOG)"
# Derive per-model defaults so this smoke works with any recipe, not
# just Qwen. Caller can override via MODEL_ALIAS / TOOL_CALL_PARSER.
DEFAULTS_JSON="$("$PY" "$PLAYBOOK_DIR/derive_model_defaults.py" "$MODEL_ID" 2>/dev/null || echo '{}')"
MODEL_ALIAS="${MODEL_ALIAS:-$(printf '%s' "$DEFAULTS_JSON" | "$PY" -c "import json,sys;print(json.loads(sys.stdin.read() or '{}').get('alias') or '')")}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-$(printf '%s' "$DEFAULTS_JSON" | "$PY" -c "import json,sys;print(json.loads(sys.stdin.read() or '{}').get('tool_call_parser') or '')")}"
[ -n "$MODEL_ALIAS" ] || MODEL_ALIAS="$(printf '%s' "$MODEL_ID" | awk -F/ '{print tolower($NF)}')"
log "smoke model_alias=$MODEL_ALIAS tool_call_parser=$TOOL_CALL_PARSER"

# Smoke does not exercise tool-calling; we skip --enable-auto-tool-choice
# so we still smoke-test models whose tool parser isn't known yet. Capture
# is independent of the tool path.
"$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$MODEL_ID" "$MODEL_ALIAS" \
  --port "$VLLM_PORT" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enable-return-routed-experts \
  >"$LOG" 2>&1 &
echo $! > "$PIDFILE"

# Wait for /health (up to 5 min — model load on first run can be slow).
deadline=$(( $(date +%s) + 300 ))
ready=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
    ready=1; break
  fi
  sleep 3
done
if [ "$ready" -ne 1 ]; then
  echo "SMOKE: FAIL reason=vllm_did_not_become_ready_in_300s log=$LOG"
  exit 1
fi
log "vllm is up; sending capture probe"

# One tiny completion. We use /v1/completions (simpler than chat) and
# ask for a few tokens. The response is expected to carry a base64
# numpy blob in `routed_experts`.
resp="$(curl -fsS -X POST "http://127.0.0.1:$VLLM_PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"def fib(n):\",\"max_tokens\":12,\"temperature\":0.0}" \
  )" || { echo "SMOKE: FAIL reason=completion_request_failed"; exit 1; }

"$PY" - "$resp" <<'PY' || { echo "SMOKE: FAIL reason=routed_experts_decode_failed"; exit 1; }
import base64, io, json, sys
import numpy as np
resp = json.loads(sys.argv[1])
# The field is per-choice (see vllm/entrypoints/openai/.../protocol.py),
# never at top level — checking both is harmless.
choices = resp.get("choices") or []
b64 = (choices[0].get("routed_experts") if choices else None) or resp.get("routed_experts")
if not b64:
    print("SMOKE: FAIL reason=routed_experts_field_missing")
    sys.exit(1)
arr = np.load(io.BytesIO(base64.b64decode(b64)), allow_pickle=False)
if arr.ndim != 3:
    print(f"SMOKE: FAIL reason=unexpected_rank shape={arr.shape}")
    sys.exit(1)
if arr.size == 0:
    print(f"SMOKE: FAIL reason=empty_array shape={arr.shape}")
    sys.exit(1)
n_tokens, n_layers, top_k = arr.shape
# Hybrid MoE/linear-attn models (Qwen3.5-MoE) still route every layer
# through experts even when attention is linear — at least one layer must
# have non-zero IDs to confirm the capturer is wired.
nonzero_layers = int((arr.sum(axis=(0, 2)) > 0).sum())
if nonzero_layers == 0:
    print(f"SMOKE: FAIL reason=all_layers_zero shape={arr.shape}")
    sys.exit(1)
print(f"SMOKE: OK shape={arr.shape} dtype={arr.dtype} nonzero_layers={nonzero_layers}")
PY
