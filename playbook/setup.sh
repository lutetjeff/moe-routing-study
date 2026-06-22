#!/usr/bin/env bash
# Idempotent installer. Safe to re-run.
#
# Expects env (set by bootstrap.sh):
#   PLAYBOOK_DIR     this dir
#   WORK_DIR         where to put deep-swe, runs, venvs, logs
#   PY               python interpreter that already has vllm
#   MODEL_ID         HF id we'll serve
#   MINIMAX_KEY_FILE optional path to a file with the MiniMax API key
#
# What this does:
#   1. Verify GPU + vllm + capture flag (calls probe_vllm.py).
#   2. Install pier + mini-swe-agent (via uv tool).
#   3. Install proxy deps (fastapi, uvicorn, httpx, numpy) into a side venv
#      so we don't pollute the vllm venv.
#   4. Clone deep-swe at a known commit.
#   5. Verify model files are present (DOES NOT download large weights
#      silently; warns the user instead — bootstrap can opt-in to dl).
#   6. Authenticate opencode to MiniMax-M3 if the key file exists.
#
# Exits non-zero with a clear `SETUP: FAIL ...` line for the operator agent.
set -euo pipefail

# shellcheck disable=SC2153
: "${PLAYBOOK_DIR:?}"
: "${WORK_DIR:?}"
: "${PY:?}"
: "${MODEL_ID:?}"

log() { printf '[setup] %s\n' "$*" >&2; }
fail() { printf 'SETUP: FAIL %s\n' "$*"; exit 1; }
warn() { printf 'SETUP: WARN %s\n' "$*"; }

mkdir -p "$WORK_DIR" "$WORK_DIR/runs" "$WORK_DIR/logs"

# 1. GPU + vLLM + capture probe ---------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "no nvidia-smi on PATH"
fi
nvidia-smi -L >&2 || fail "nvidia-smi failed"

probe_json="$("$PY" "$PLAYBOOK_DIR/probe_vllm.py")" || fail "vllm probe failed: $probe_json"
tier="$(printf '%s' "$probe_json" | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["tier"])')"
log "vllm probe: $probe_json"
case "$tier" in
  A) log "tier A — native capture flag available";;
  B) warn "tier B — capturer present but flag missing; manual init required";;
  *) fail "vllm capture unavailable (tier=$tier). Pin vllm>=0.21 with routed-experts support; tier C (forward-hook fallback) is forbidden.";;
esac
echo "$tier" >"$WORK_DIR/.tier"

# 2. pier + mini-swe-agent via uv ------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Both are no-op if already installed.
uv tool install datacurve-pier 2>&1 | tail -3 >&2 || warn "pier install reported issues"
uv tool install mini-swe-agent 2>&1 | tail -3 >&2 || warn "mini-swe install reported issues"

PIER="$(command -v pier || true)"
[ -z "$PIER" ] && PIER="$HOME/.local/bin/pier"
[ -x "$PIER" ] || fail "pier not on PATH after install"
log "pier: $PIER ($("$PIER" --version 2>/dev/null || echo unknown))"

# 3. proxy venv -------------------------------------------------------------
PROXY_VENV="$WORK_DIR/proxy-venv"
if [ ! -d "$PROXY_VENV" ]; then
  log "creating proxy venv at $PROXY_VENV"
  "$PY" -m venv "$PROXY_VENV" 2>/dev/null || python3 -m venv "$PROXY_VENV"
fi
"$PROXY_VENV/bin/pip" install --quiet --upgrade pip
"$PROXY_VENV/bin/pip" install --quiet \
  'fastapi>=0.110' 'uvicorn[standard]>=0.27' 'httpx>=0.27' 'numpy>=1.26' \
  'matplotlib>=3.7' 'pandas>=2.0' 2>&1 | tail -3 >&2 || fail "proxy deps install failed"
log "proxy venv ready"

# 4. deep-swe --------------------------------------------------------------
DEEPSWE_DIR="$WORK_DIR/deep-swe"
if [ ! -d "$DEEPSWE_DIR/.git" ]; then
  log "cloning deep-swe..."
  git clone --depth 1 https://github.com/datacurve-ai/deep-swe.git "$DEEPSWE_DIR" >&2 \
    || fail "deep-swe clone failed"
fi
[ -d "$DEEPSWE_DIR/tasks" ] || fail "deep-swe/tasks missing after clone"
log "deep-swe ready: $(ls "$DEEPSWE_DIR/tasks" | wc -l) entries"

# 5. model presence check --------------------------------------------------
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
hf_dir_name="models--$(echo "$MODEL_ID" | tr '/' '-' | sed 's/^/&/')"
# HF cache uses models--<org>--<name>
hf_dir_name="models--$(echo "$MODEL_ID" | tr '/' '-')"
if find "$HF_HOME/hub" -maxdepth 1 -type d -name "models--*$(echo "$MODEL_ID" | tr '/' '-')*" 2>/dev/null | grep -q .; then
  log "model files present under $HF_HOME"
else
  warn "model $MODEL_ID not found under $HF_HOME — vllm serve will download (~tens of GB)"
fi

# 6. opencode + MiniMax auth ------------------------------------------------
if command -v opencode >/dev/null 2>&1; then
  log "opencode: $(opencode --version 2>&1 | tr -d '\n')"
  if [ -n "${MINIMAX_KEY_FILE:-}" ] && [ -f "$MINIMAX_KEY_FILE" ]; then
    key="$(tr -d ' \t\r\n' <"$MINIMAX_KEY_FILE")"
    if [ -n "$key" ]; then
      # Non-interactive auth: opencode auth login uses a TTY prompt by
      # default, so we write straight to its credentials file. The schema
      # is documented at opencode.ai/docs/auth.
      auth_file="$HOME/.local/share/opencode/auth.json"
      mkdir -p "$(dirname "$auth_file")"
      tmp="$(mktemp)"
      "$PY" - "$auth_file" "$key" "$tmp" <<'PY'
import json, os, sys
path, key, tmp = sys.argv[1], sys.argv[2], sys.argv[3]
data = {}
if os.path.exists(path):
    try:
        data = json.load(open(path))
    except Exception:
        data = {}
data["minimax"] = {"type": "api", "key": key}
json.dump(data, open(tmp, "w"))
PY
      mv "$tmp" "$auth_file"
      chmod 600 "$auth_file"
      log "wrote MiniMax credential to $auth_file"
    fi
  else
    warn "no MINIMAX_KEY_FILE — operator agent will need manual auth"
  fi
else
  warn "opencode CLI not on PATH — install via npm or curl https://opencode.ai/install"
fi

# 7. final sanity-touch ----------------------------------------------------
echo "$tier" > "$WORK_DIR/.tier"
echo "ok" > "$WORK_DIR/.setup_ok"
log "setup complete"
echo "SETUP: OK tier=$tier"
