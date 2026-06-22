#!/usr/bin/env python3
"""Capability probe for vLLM's routed-experts capture path.

Tiers:
  A — `ModelConfig.enable_return_routed_experts` flag exists (sgl-fork / our
      custom build / any future upstream merge). Pass `--enable-return-routed-experts`
      to `vllm serve` and the OpenAI `routed_experts` field is populated for free.
  B — `RoutedExpertsCapturer` class is importable but the flag isn't wired.
      Caller must (a) call `init_routed_experts_capturer()` on each worker via
      RPC after engine init, and (b) patch the serving layer to attach the
      base64-encoded buffer onto the response. This is the rarer case.
  C — Neither exists. We can fall back to a Python forward-hook over every
      `FusedMoE` module, but it adds per-token Python overhead in the hot
      path. Acceptable for offline analysis but slower.
  X — vLLM not importable or < 0.19.

Writes a single JSON object to stdout. Exit 0 unless vLLM is unusable.
"""
from __future__ import annotations
import json
import sys


def main() -> int:
    out: dict = {"tier": "X", "vllm_version": None, "notes": []}
    try:
        import vllm  # noqa: F401
    except Exception as e:
        out["notes"].append(f"vllm import failed: {e!r}")
        print(json.dumps(out))
        return 1

    out["vllm_version"] = getattr(vllm, "__version__", "unknown")

    # Minimum supported version: 0.21. Below that the RoutedExpertsCapturer
    # internals are too divergent to rely on, so we refuse rather than
    # silently fall back. We parse the major/minor prefix and ignore any
    # dev/rc suffix (so 0.21.1rc1.dev400 still passes).
    ver = out["vllm_version"] or ""
    head = ver.split("+", 1)[0].split("rc", 1)[0].split("dev", 1)[0].rstrip(".")
    try:
        major, minor, *_ = (int(x) for x in head.split("."))
        if (major, minor) < (0, 21):
            out["notes"].append(f"vllm {ver} below minimum 0.21")
            print(json.dumps(out))
            return 1
    except Exception:
        out["notes"].append(f"could not parse version {ver!r}; assuming OK")

    # Tier A: flag exists on ModelConfig
    try:
        from vllm.config import ModelConfig

        if hasattr(ModelConfig, "enable_return_routed_experts") or (
            "enable_return_routed_experts" in getattr(ModelConfig, "__dataclass_fields__", {})
        ):
            out["tier"] = "A"
            out["notes"].append("ModelConfig.enable_return_routed_experts present")
            print(json.dumps(out))
            return 0
    except Exception as e:
        out["notes"].append(f"ModelConfig inspect failed: {e!r}")

    # Tier B: capturer class importable
    try:
        import vllm.model_executor.layers.fused_moe.routed_experts_capturer  # noqa: F401

        out["tier"] = "B"
        out["notes"].append("RoutedExpertsCapturer class present but flag missing")
        print(json.dumps(out))
        return 0
    except Exception:
        pass

    # Tier C explicitly forbidden — we restrict to vllm builds that ship the
    # capturer. Falling back to forward-hooks at this version range is
    # fragile and adds enough per-token Python overhead to skew the run.
    out["notes"].append("tier C (forward-hook fallback) is forbidden on vllm>=0.21")
    print(json.dumps(out))
    return 1


if __name__ == "__main__":
    sys.exit(main())
