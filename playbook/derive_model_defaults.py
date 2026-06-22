#!/usr/bin/env python3
"""Suggest sensible defaults for serving a MoE model with capture enabled.

We inspect the HF model config to pick:
  * A short serving alias (so OpenAI clients can address it without the org/).
  * A vLLM ``--tool-call-parser`` that matches the chat template's tool-call
    format (matters for any agent that sends ``tool_choice: "auto"``,
    which mini-swe-agent does on every turn).

The output is one JSON line on stdout so shell scripts can read it with
``python -c json``. Exit 0 even when we couldn't read the config; we just
return ``"unknown"`` and let the caller decide. Exit non-zero only on
catastrophic failure (e.g. import error).

Usage:
    python derive_model_defaults.py <hf-model-id> [--hf-home <dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Map a family substring (found in architecture or model_type) to the
# vLLM tool-call parser shipped under vllm/tool_parsers/. Order matters —
# first hit wins. Add to this list as we support more recipes.
_FAMILY_TO_PARSER: list[tuple[str, str]] = [
    ("qwen3_5", "qwen3_xml"),
    ("qwen3moe", "qwen3_xml"),
    ("qwen3_moe", "qwen3_xml"),
    ("qwen3", "qwen3_xml"),
    ("qwen2", "hermes"),
    ("minimaxm2", "minimax_m2"),
    ("minimaxm1", "minimax"),
    ("minimax", "minimax"),
    ("deepseekv4", "deepseek_v4"),
    ("deepseekv32", "deepseek_v32"),
    ("deepseekv31", "deepseek_v31"),
    ("deepseekv3", "deepseek_v3"),
    ("deepseek", "deepseek_v3"),
    ("glm47", "glm47"),
    ("glm45", "glm45"),
    ("granite4", "granite4"),
    ("granite", "granite"),
    ("gemma4", "gemma4"),
    ("hunyuan_a13b", "hunyuan_a13b"),
    ("llama4", "llama4_json"),
    ("llama3", "llama3_json"),
    ("mistral", "mistral"),
    ("mimo", "mimo"),
    ("phi4_mini", "phi4_mini_json"),
    ("internlm", "internlm"),
    ("kimi_k2", "kimi_k2"),
    ("step3p5", "step3p5"),
    ("step3", "step3"),
]


def _hf_cache_config(model_id: str, hf_home: str | None) -> dict | None:
    """Walk the local HF cache for a config.json. None if not present."""
    bases: list[Path] = []
    for cand in (hf_home, os.environ.get("HF_HOME"), os.path.expanduser("~/.cache/huggingface")):
        if cand:
            bases.append(Path(cand))
    cache_name = "models--" + model_id.replace("/", "--")
    for base in bases:
        snap_root = base / "hub" / cache_name / "snapshots"
        if not snap_root.is_dir():
            continue
        # Take the lex-largest snapshot dir (usually the most recent).
        for snap in sorted(snap_root.iterdir(), reverse=True):
            cfg = snap / "config.json"
            if cfg.is_file():
                try:
                    return json.loads(cfg.read_text())
                except Exception:
                    continue
    return None


def _alias(model_id: str) -> str:
    """A short alias the agent can use to address the model.

    Strips the org prefix and quant suffix so e.g.
    ``cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`` → ``qwen3.6-35b-a3b-awq``.
    """
    tail = model_id.rsplit("/", 1)[-1].lower()
    for suffix in ("-4bit", "-8bit", "-int4", "-int8", "-awq-4bit", "-gptq-4bit"):
        if tail.endswith(suffix):
            tail = tail[: -len(suffix)]
            break
    return tail


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _parser_for(config: dict | None, model_id: str) -> str:
    """Pick a tool-call parser by family substring match."""
    candidates: list[str] = []
    if config:
        for arch in config.get("architectures", []) or []:
            candidates.append(arch)
        if mt := config.get("model_type"):
            candidates.append(mt)
        # qwen3.5-moe etc. live under text_config in newer multimodal exports
        text_cfg = config.get("text_config") or {}
        if mt := text_cfg.get("model_type"):
            candidates.append(mt)
    candidates.append(model_id)

    haystack = " ".join(_norm(c) for c in candidates)
    for needle, parser in _FAMILY_TO_PARSER:
        if needle in haystack:
            return parser
    return "unknown"


def _max_position_embeddings(config: dict | None) -> int | None:
    if not config:
        return None
    for k in ("max_position_embeddings", "max_seq_len", "max_seq_length"):
        if k in config:
            try:
                return int(config[k])
            except (TypeError, ValueError):
                pass
    text_cfg = config.get("text_config") or {}
    for k in ("max_position_embeddings", "max_seq_len"):
        if k in text_cfg:
            try:
                return int(text_cfg[k])
            except (TypeError, ValueError):
                pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--hf-home")
    args = ap.parse_args()

    config = _hf_cache_config(args.model_id, args.hf_home)
    out: dict[str, object] = {
        "model_id": args.model_id,
        "alias": _alias(args.model_id),
        "tool_call_parser": _parser_for(config, args.model_id),
        "have_local_config": config is not None,
        "max_position_embeddings": _max_position_embeddings(config),
    }
    if config:
        out["architectures"] = config.get("architectures")
        out["model_type"] = config.get("model_type")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
