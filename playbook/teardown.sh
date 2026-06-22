#!/usr/bin/env bash
# Best-effort cleanup of vllm + proxy + any lingering pier sandboxes.
set -u
log() { printf '[teardown] %s\n' "$*" >&2; }

# Kill anything bound to our ports.
for port in "${VLLM_PORT:-8000}" "${PROXY_PORT:-8001}"; do
  pid="$(ss -ltnp 2>/dev/null | awk -v p=":$port" '$4 ~ p {print $7}' | sed -nE 's/.*pid=([0-9]+).*/\1/p' | head -1)"
  if [ -n "${pid:-}" ]; then
    log "killing pid=$pid on port=$port"
    kill -TERM "$pid" 2>/dev/null || true
  fi
done

# Pier docker containers (best-effort).
if command -v docker >/dev/null 2>&1; then
  ids="$(docker ps --filter 'label=harbor.trial' -q 2>/dev/null || true)"
  if [ -n "${ids:-}" ]; then
    log "stopping pier sandbox containers"
    docker stop $ids >/dev/null 2>&1 || true
  fi
fi
log "done"
