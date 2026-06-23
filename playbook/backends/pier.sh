#!/usr/bin/env bash
# Backend: pier+docker.
#
# Materializes the pier JobConfig from the template, then invokes
# pier with --env docker under a hard wall-clock watchdog. Pier spawns
# one harbor sandbox + one squid egress proxy per trial.
#
# Inputs (env, set by run.sh):
#   PLAYBOOK_DIR, WORK_DIR, PY, MODEL_ID, MODEL_ALIAS, MODEL_CLASS,
#   PROXY_BASE_URL_FROM_CONTAINER, JOB_NAME, JOB_DIR, DATASET_DIR,
#   JOB_BUDGET_SEC, CONCURRENCY
#
# Stdout: last line ``BACKEND: OK|PARTIAL <reason>``. Exit code mirrors
# pier's so run.sh can distinguish budget-hit (124/137) from other failures.
set -euo pipefail
: "${PLAYBOOK_DIR:?}"; : "${WORK_DIR:?}"; : "${JOB_DIR:?}"; : "${JOB_NAME:?}"
: "${DATASET_DIR:?}"; : "${MODEL_ALIAS:?}"; : "${PROXY_BASE_URL_FROM_CONTAINER:?}"
: "${JOB_BUDGET_SEC:?}"
CONCURRENCY="${CONCURRENCY:-1}"
MODEL_CLASS="${MODEL_CLASS:-litellm}"

log() { printf '[pier] %s\n' "$*" >&2; }

PIER="$(command -v pier || echo "$HOME/.local/bin/pier")"
[ -x "$PIER" ] || { echo "BACKEND: FAIL pier missing"; exit 2; }

# Pier does NOT env-interpolate YAML, so substitute upfront. Per-job
# copy lives under the job dir so trial-replay tooling can find the
# exact config that was used.
PIER_CFG="$JOB_DIR/pier-mini-vllm.yaml"
sed \
  -e "s|__MODEL_ALIAS__|${MODEL_ALIAS}|g" \
  -e "s|__API_BASE__|${PROXY_BASE_URL_FROM_CONTAINER}|g" \
  -e "s|__MODEL_CLASS__|${MODEL_CLASS}|g" \
  "$PLAYBOOK_DIR/config/pier-mini-vllm.yaml.template" >"$PIER_CFG"
log "materialized pier config: $PIER_CFG"

PIER_LOG="$JOB_DIR/logs/pier.log"

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
rc=$?
set -e
log "pier exit=$rc"

if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "BACKEND: PARTIAL job_budget_hit"
elif [ "$rc" -ne 0 ]; then
  echo "BACKEND: PARTIAL pier_exit_$rc"
else
  echo "BACKEND: OK"
fi
exit "$rc"
