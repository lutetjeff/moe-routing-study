"""Server-side shim that ensures the OpenAI response carries `routed_experts`.

Only used for tier C (and as a fallback safety net) — for tier A we don't
need this at all, vLLM does it natively. Tier B is rare enough to handle
manually if/when we hit it. For now we provide a single export
``ensure_routed_experts_enabled(server_args: list[str]) -> list[str]``
that appends the flag if missing in tier A.

Tier-C hook (loaded into the vLLM worker process via VLLM_PLUGINS) lives
under ``forward_hook/`` — it's invoked only when this module is imported
inside the engine process. The proxy invokes it through an env var rather
than monkey-patching from outside.
"""
from __future__ import annotations
import os


def maybe_append_flag(args: list[str], tier: str) -> list[str]:
    """For tier A, ensure ``--enable-return-routed-experts`` is in args.

    No-op for B/C — those need engine-side work and won't be solved by a
    flag. The script falls back to printing a warning when running on B/C.
    """
    if tier != "A":
        return args
    if "--enable-return-routed-experts" in args:
        return args
    return args + ["--enable-return-routed-experts"]


def tier_from_env(default: str = "A") -> str:
    return os.environ.get("ROUTING_TIER", default).strip().upper() or default
