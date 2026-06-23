#!/usr/bin/env bash
# Backend: docker-free local runner.
#
# Drives local_runner.py, which clones each task's repo, runs
# mini-swe-agent as the unprivileged ``mse-runner`` user against a
# fresh work copy, captures (trajectory, model.patch) per trial in a
# pier-compatible directory layout, and leaves a verifier/.pending
# sentinel for the post-hoc grader.
#
# Inputs (env, set by run.sh):
#   PLAYBOOK_DIR, WORK_DIR, PY, MODEL_ID, MODEL_ALIAS, MODEL_CLASS,
#   JOB_NAME, JOB_DIR, DATASET_DIR, JOB_BUDGET_SEC, TASK_BUDGET_SEC,
#   PROXY_PORT
#
# Stdout: last line ``BACKEND: OK|PARTIAL <reason>``.
set -euo pipefail
: "${PLAYBOOK_DIR:?}"; : "${WORK_DIR:?}"; : "${PY:?}"; : "${MODEL_ID:?}"
: "${JOB_DIR:?}"; : "${JOB_NAME:?}"; : "${DATASET_DIR:?}"
: "${JOB_BUDGET_SEC:?}"; : "${TASK_BUDGET_SEC:?}"
PROXY_PORT="${PROXY_PORT:-8001}"
MSE_RUNNER_USER="${MSE_RUNNER_USER:-mse-runner}"

log() { printf '[local] %s\n' "$*" >&2; }

command -v git >/dev/null || { echo "BACKEND: FAIL git missing"; exit 2; }
command -v mini-swe-agent >/dev/null || {
  echo "BACKEND: FAIL mini-swe-agent missing on PATH"; exit 2;
}

# Verify the unprivileged user exists. setup.sh creates them; we just
# refuse to run as root-on-host if they're missing because then
# everything the agent does would run as root.
if ! id "$MSE_RUNNER_USER" >/dev/null 2>&1; then
  echo "BACKEND: FAIL user '$MSE_RUNNER_USER' missing (run setup.sh)"
  exit 2
fi

# Pick a privilege drop tool. Prefer sudo (sudoers entry is added by
# setup.sh); fall back to runuser if sudo isn't usable.
DROP_TOOL=""
if command -v sudo >/dev/null && sudo -n -u "$MSE_RUNNER_USER" true 2>/dev/null; then
  DROP_TOOL="sudo"
elif command -v runuser >/dev/null && [ "$(id -u)" = "0" ]; then
  DROP_TOOL="runuser"
else
  echo "BACKEND: FAIL no privilege-drop tool (need passwordless sudo to $MSE_RUNNER_USER or runuser as root)"
  exit 2
fi
log "privilege drop tool: $DROP_TOOL"

LOCAL_LOG="$JOB_DIR/logs/local.log"

set +e
timeout --signal=TERM --kill-after=60 "${JOB_BUDGET_SEC}s" \
  "$PY" "$PLAYBOOK_DIR/local_runner.py" \
    --dataset "$DATASET_DIR" \
    --jobs-dir "$JOB_DIR/pier-jobs" \
    --job-name "$JOB_NAME" \
    --proxy-base-url "http://127.0.0.1:${PROXY_PORT}/v1" \
    --model-id "$MODEL_ID" \
    --model-alias "${MODEL_ALIAS:-}" \
    --model-class "${MODEL_CLASS:-litellm}" \
    --task-timeout-sec "$TASK_BUDGET_SEC" \
    --clone-cache "$WORK_DIR/clones" \
    --user "$MSE_RUNNER_USER" \
    --drop-tool "$DROP_TOOL" \
    >"$LOCAL_LOG" 2>&1
rc=$?
set -e
log "local_runner exit=$rc"

if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "BACKEND: PARTIAL job_budget_hit"
elif [ "$rc" -ne 0 ]; then
  echo "BACKEND: PARTIAL local_runner_exit_$rc"
else
  echo "BACKEND: OK"
fi
exit "$rc"
