#!/usr/bin/env bash
# bootstrap.sh — single drop-in entry point for the expert-routing benchmark.
#
# Usage:
#   ./bootstrap.sh run [N]          # full pipeline (default N=1)
#   ./bootstrap.sh setup            # install deps only, no run
#   ./bootstrap.sh smoke            # capture smoke test only
#   ./bootstrap.sh agent [N]        # run via opencode operator agent
#   ./bootstrap.sh teardown         # kill servers, drop containers
#   ./bootstrap.sh report <job>     # rebuild analysis for an existing run
#
# What it does (`run` mode):
#   1. Resolves PY/MODEL/WORK from env or sensible defaults.
#   2. Calls playbook/setup.sh (installs pier, mini-swe-agent, proxy venv,
#      clones deep-swe, wires MiniMax credentials).
#   3. Calls playbook/smoke.sh (boots vllm with --enable-return-routed-experts,
#      verifies one completion returns a real routing tensor). If this
#      fails, the script aborts immediately so you can stop the GPU.
#   4. Calls playbook/run.sh N (boots vllm+proxy, runs pier on the curated
#      subset with the right wall-clock watchdog, builds the report).
#
# Watchdog budgets:
#   N=1  -> 15 min total
#   N=10 -> 90 min total, kill any single task at 9 min
#   N=20 -> 180 min total, kill any single task at 9 min
#
# Env knobs (all optional; defaults aimed at the local A100 dev box):
#   PY               python with vllm installed
#   MODEL_ID         HF id, e.g. cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit (the
#                    one knob users normally set when switching recipes)
#   WORK_DIR         where to store runs, logs, deep-swe checkout, proxy venv
#   HF_HOME          override HuggingFace cache
#   MINIMAX_KEY_FILE path to a file holding the MiniMax API key
#   VLLM_PORT        default 8000
#   PROXY_PORT       default 8001
#   GPU_MEM_UTIL     default 0.85
#   MAX_MODEL_LEN    default 16384
#   CONCURRENCY      default 1 (serial; safest on a single GPU)
#
# Per-model overrides (usually NOT needed — derive_model_defaults.py
# infers them from MODEL_ID + the local HF cache):
#   MODEL_ALIAS      short served-model alias (e.g. qwen3.6-35b-a3b-awq)
#   TOOL_CALL_PARSER vLLM --tool-call-parser to use (qwen3_xml, hermes,
#                    deepseek_v3, minimax_m2, llama3_json, ...)
#   MODEL_CLASS      mini-swe-agent adapter (litellm | litellm_response).
#                    Default litellm hits /v1/chat/completions which is
#                    where the routed_experts field lives.
#   PROXY_BASE_URL_FROM_CONTAINER  URL inside the sandbox that points to
#                    our proxy on the host. Default http://172.17.0.1:$PROXY_PORT/v1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAYBOOK_DIR="${PLAYBOOK_DIR:-$SCRIPT_DIR/playbook}"
WORK_DIR="${WORK_DIR:-$SCRIPT_DIR/work}"
MINIMAX_KEY_FILE="${MINIMAX_KEY_FILE:-$SCRIPT_DIR/secrets/minimax_api_key.txt}"
MODEL_ID="${MODEL_ID:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}"

# Resolve a python interpreter that can `import vllm`.
#
# Preference order:
#   1. $PY env override
#   2. Anything matching a small allowlist of reasonable directories where
#      vllm is commonly installed in production (project venvs, conda
#      envs, uv-managed pythons, /opt, /workspace, system python3).
#   3. We refuse to walk the entire filesystem.
find_python_with_vllm() {
  local seen=""
  local cands=()
  # User overrides come first.
  [ -n "${PY:-}" ] && cands+=("$PY")
  # Common project venvs.
  for d in \
      "$HOME/finetune/vllm-0.21-venv" \
      "$HOME/finetune/vllm-venv" \
      "$HOME/finetune/.venv" \
      "$HOME/.venv" \
      "$HOME/.venvs/vllm" \
      "$HOME/venv-vllm" \
      "$HOME/envs/vllm" \
      "/opt/vllm" \
      "/opt/vllm-venv" \
      "/opt/venv" \
      "/workspace/.venv" \
      "/workspace/vllm-venv"; do
    [ -x "$d/bin/python" ] && cands+=("$d/bin/python")
  done
  # Glob a few directories that often contain many venvs.
  for parent in \
      "$HOME/finetune" \
      "$HOME/.venvs" \
      "$HOME/envs" \
      "$HOME/.cache/uv/python" \
      "$HOME/.local/share/uv/python" \
      "/root/.cache/uv/python" \
      "/root/.local/share/uv/python" \
      "/opt"; do
    [ -d "$parent" ] || continue
    while IFS= read -r p; do
      [ -x "$p" ] && cands+=("$p")
    done < <(find "$parent" -maxdepth 4 -type f -name python -path '*/bin/python' 2>/dev/null | head -50)
  done
  # Conda / micromamba envs.
  if command -v conda >/dev/null 2>&1; then
    while IFS= read -r p; do
      [ -x "$p" ] && cands+=("$p")
    done < <(conda env list 2>/dev/null | awk 'NR>2 && $NF != "" {print $NF"/bin/python"}' | head -10)
  fi
  # pyenv versions.
  if command -v pyenv >/dev/null 2>&1; then
    while IFS= read -r v; do
      p="$(pyenv root 2>/dev/null)/versions/$v/bin/python"
      [ -x "$p" ] && cands+=("$p")
    done < <(pyenv versions --bare 2>/dev/null | head -10)
  fi
  # uv tool / pipx envs (long shot but cheap).
  for p in \
      "$HOME/.local/share/pipx/venvs/vllm/bin/python" \
      "$HOME/.local/share/uv/tools/vllm/bin/python"; do
    [ -x "$p" ] && cands+=("$p")
  done
  # Anything named pythonX on PATH that we haven't already considered.
  for n in python3 python python3.10 python3.11 python3.12; do
    p="$(command -v "$n" 2>/dev/null || true)"
    [ -n "$p" ] && [ -x "$p" ] && cands+=("$p")
  done
  # Resolve symlinks and de-dupe before probing — `import vllm` is
  # expensive enough that 10s of redundant attempts matter.
  for c in "${cands[@]}"; do
    rp="$(readlink -f "$c" 2>/dev/null || echo "$c")"
    case "$seen" in *"|$rp|"*) continue ;; esac
    seen="$seen|$rp|"
    if "$rp" -c "import vllm; import sys; sys.exit(0)" >/dev/null 2>&1; then
      printf '%s\n' "$rp"
      return 0
    fi
  done
  return 1
}

if [ -z "${PY:-}" ] || ! "$PY" -c "import vllm" >/dev/null 2>&1; then
  echo "[bootstrap] searching for a python with vllm..." >&2
  PY="$(find_python_with_vllm || true)"
fi
if [ -z "${PY:-}" ] || ! "$PY" -c "import vllm" >/dev/null 2>&1; then
  echo "BOOTSTRAP: FAIL no python found with vllm importable" >&2
  echo "Set PY=/path/to/python (must have vllm>=0.21 with routed-experts support)" >&2
  exit 1
fi
echo "[bootstrap] using PY=$PY" >&2

export PLAYBOOK_DIR WORK_DIR PY MODEL_ID MINIMAX_KEY_FILE
mkdir -p "$WORK_DIR"

cmd="${1:-run}"
shift || true

case "$cmd" in
  setup)
    bash "$PLAYBOOK_DIR/setup.sh"
    ;;
  smoke)
    bash "$PLAYBOOK_DIR/setup.sh"
    bash "$PLAYBOOK_DIR/smoke.sh"
    ;;
  run)
    N="${1:-1}"
    bash "$PLAYBOOK_DIR/setup.sh"
    if ! bash "$PLAYBOOK_DIR/smoke.sh"; then
      echo "BOOTSTRAP: ABORT smoke test failed — stop your GPU instance to save spend." >&2
      exit 2
    fi
    bash "$PLAYBOOK_DIR/run.sh" "$N"
    ;;
  agent)
    N="${1:-1}"
    bash "$PLAYBOOK_DIR/setup.sh"
    if ! command -v opencode >/dev/null 2>&1; then
      echo "BOOTSTRAP: FAIL opencode not installed; run setup or 'curl https://opencode.ai/install | sh'" >&2
      exit 1
    fi
    # Make the agent discoverable: copy our agent dir into the project's
    # local .opencode/ if not already linked.
    proj_agent="$SCRIPT_DIR/.opencode/agent"
    mkdir -p "$proj_agent"
    cp -f "$PLAYBOOK_DIR/opencode/agent/router-bench.md" "$proj_agent/router-bench.md"
    export N
    # Stream the agent's messages live to the user and capture a copy to
    # disk for postmortem. `opencode run` writes formatted messages to
    # stdout — we `tee` it. `--print-logs` would also surface opencode's
    # own internal logs, which is too noisy for normal use; flip on by
    # setting BOOTSTRAP_VERBOSE=1.
    transcript="$WORK_DIR/logs/opencode-$(date +%Y%m%d-%H%M%S).log"
    mkdir -p "$WORK_DIR/logs"
    echo "[bootstrap] opencode transcript: $transcript" >&2
    extra_flags=()
    [ "${BOOTSTRAP_VERBOSE:-0}" = "1" ] && extra_flags+=(--print-logs --log-level INFO)
    opencode run \
      --agent router-bench \
      --dir "$SCRIPT_DIR" \
      "${extra_flags[@]}" \
      "Run the expert-routing benchmark for N=$N. Stick to the loop in the agent system prompt." \
      2>&1 | tee "$transcript"
    ;;
  teardown)
    bash "$PLAYBOOK_DIR/teardown.sh"
    ;;
  report)
    job="${1:?usage: bootstrap.sh report <job-name>}"
    JOB_DIR="$WORK_DIR/runs/$job"
    [ -d "$JOB_DIR" ] || { echo "no such job: $JOB_DIR" >&2; exit 1; }
    "$WORK_DIR/proxy-venv/bin/python" "$PLAYBOOK_DIR/analysis/build_report.py" \
      --job-dir "$JOB_DIR" --out "$JOB_DIR/report"
    bash "$PLAYBOOK_DIR/analysis/describe_with_mmx.sh" "$JOB_DIR/report" || true
    ls -la "$JOB_DIR/report"
    ;;
  *)
    echo "usage: $0 {setup|smoke|run [N]|agent [N]|teardown|report <job>}" >&2
    exit 2
    ;;
esac
