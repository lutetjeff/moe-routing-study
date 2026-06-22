"""Transparent OpenAI proxy that peels off and persists vLLM's
``routed_experts`` field on every response.

Why a proxy: vLLM's `enable_return_routed_experts` returns a base64-encoded
numpy blob inline on the OpenAI response, but mini-swe-agent / LiteLLM
strip unknown fields before the calling code ever sees them. We sit
between the agent and vLLM, transparently forward the OpenAI request,
and on the way back: persist the routed_experts to disk under the
current pier trial dir, then strip the field so downstream consumers
get a clean OpenAI payload.

We also capture cheap sidecar telemetry (timings, token counts, sampling
params, prefix-cache hits scraped from /metrics) into a JSON file next
to each .npz, so the analysis stage doesn't have to re-correlate anything.

Run with:
    ROUTING_OUTPUT_DIR=/path/to/runs \\
    VLLM_BASE=http://127.0.0.1:8000 \\
    PROXY_PORT=8001 \\
    python routing_proxy.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse


VLLM_BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8000").rstrip("/")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8001"))
OUTPUT_DIR = Path(os.environ.get("ROUTING_OUTPUT_DIR", "/tmp/expert-routing-runs"))
JOB_ID = os.environ.get("ROUTING_JOB_ID", "default")
METRICS_URL = f"{VLLM_BASE}/metrics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("routing_proxy")

app = FastAPI()
client = httpx.AsyncClient(base_url=VLLM_BASE, timeout=httpx.Timeout(600.0))

# Tracks the latest /metrics scrape so request sidecars carry context
# (prefix-cache hits, queue depth) without a per-request scrape.
_metrics_cache: dict[str, Any] = {"ts": 0.0, "data": {}}
_metrics_lock = asyncio.Lock()


def _trial_dir(trial: str, task: str) -> Path:
    d = OUTPUT_DIR / JOB_ID / (task or "untagged") / (trial or "untagged") / "routing"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _decode_routed_experts(b64: str) -> np.ndarray | None:
    try:
        raw = base64.b64decode(b64)
        return np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as e:
        log.warning("routed_experts decode failed: %s", e)
        return None


async def _refresh_metrics() -> None:
    """Pull /metrics at most once per second; parse the few keys we care about."""
    now = time.time()
    if now - _metrics_cache["ts"] < 1.0:
        return
    async with _metrics_lock:
        if now - _metrics_cache["ts"] < 1.0:
            return
        try:
            r = await client.get("/metrics", timeout=5.0)
            data: dict[str, float] = {}
            for line in r.text.splitlines():
                if not line or line.startswith("#"):
                    continue
                name, _, rest = line.partition(" ")
                try:
                    val = float(rest.strip().split()[0])
                except (IndexError, ValueError):
                    continue
                # Keep only what we'll use; strip label sets to keep the dict small.
                if any(name.startswith(p) for p in (
                    "vllm:prefix_cache_queries",
                    "vllm:prefix_cache_hits",
                    "vllm:gpu_cache_usage_perc",
                    "vllm:num_requests_running",
                    "vllm:num_requests_waiting",
                )):
                    data[name] = val
            _metrics_cache["data"] = data
            _metrics_cache["ts"] = now
        except Exception as e:
            log.debug("metrics scrape failed: %s", e)


def _persist(
    trial: str,
    task: str,
    turn: str,
    record: dict[str, Any],
    routed_experts: np.ndarray | None,
) -> None:
    d = _trial_dir(trial, task)
    seq = record.get("request_id") or uuid.uuid4().hex[:12]
    stem = f"turn{turn or '0'}_req_{seq}"
    if routed_experts is not None:
        np.savez_compressed(
            d / f"{stem}.npz",
            routed_experts=routed_experts,
        )
        record["routed_experts_shape"] = list(routed_experts.shape)
        record["routed_experts_dtype"] = str(routed_experts.dtype)
    else:
        record["routed_experts_shape"] = None
    with open(d / f"{stem}.json", "w") as f:
        json.dump(record, f, indent=2, default=str)


def _extract_sampling(body: dict[str, Any]) -> dict[str, Any]:
    return {
        k: body.get(k)
        for k in ("temperature", "top_p", "top_k", "max_tokens", "stop")
        if k in body
    }


async def _proxy_json(
    path: str,
    body: dict[str, Any],
    trial: str,
    task: str,
    turn: str,
    upstream_headers: dict[str, str],
) -> Response:
    # vLLM always returns a response_id we can use for correlation. We force
    # non-streaming because that's the path that carries routed_experts on
    # the final JSON; streaming would require buffering everything anyway
    # to get the final frame, so non-streaming is simpler and equivalent
    # for our offline-analysis use case.
    streaming_requested = bool(body.get("stream"))
    body = dict(body)
    body["stream"] = False

    ts_start = time.time_ns()
    await _refresh_metrics()
    prefix_cache_before = dict(_metrics_cache["data"])

    try:
        r = await client.post(path, json=body, headers=upstream_headers)
    except httpx.HTTPError as e:
        log.error("upstream error: %s", e)
        return JSONResponse({"error": {"message": str(e)}}, status_code=502)

    ts_end = time.time_ns()
    await _refresh_metrics()
    prefix_cache_after = dict(_metrics_cache["data"])

    if r.status_code >= 400:
        return Response(content=r.content, status_code=r.status_code,
                        headers={"content-type": r.headers.get("content-type", "application/json")})

    try:
        payload = r.json()
    except Exception:
        return Response(content=r.content, status_code=r.status_code,
                        headers={"content-type": r.headers.get("content-type", "application/json")})

    # Peel routed_experts. vLLM places it on each choice (per the OpenAI
    # *ResponseChoice protocols in entrypoints/openai/*.py) — never at
    # top level. We concatenate across choices on the rare n>1 path.
    usage = payload.get("usage") or {}
    choices = payload.get("choices") or []
    routed_parts: list[np.ndarray] = []
    for ch in choices:
        b64 = ch.pop("routed_experts", None)
        if isinstance(b64, str):
            arr = _decode_routed_experts(b64)
            if arr is not None:
                routed_parts.append(arr)
    # Drop any stray top-level field too (forward compatibility).
    payload.pop("routed_experts", None)
    if routed_parts:
        routed_arr: np.ndarray | None = (
            routed_parts[0] if len(routed_parts) == 1
            else np.concatenate(routed_parts, axis=0)
        )
    else:
        routed_arr = None
    completion_text = ""
    if choices:
        msg = choices[0].get("message") or {}
        completion_text = msg.get("content") or choices[0].get("text") or ""

    record = {
        "request_id": payload.get("id"),
        "endpoint": path,
        "trial_id": trial,
        "task_id": task,
        "turn_idx": turn,
        "model": body.get("model"),
        "ts_start_ns": ts_start,
        "ts_end_ns": ts_end,
        "latency_ms": (ts_end - ts_start) / 1e6,
        "stream_requested": streaming_requested,
        "sampling": _extract_sampling(body),
        "usage": usage,
        "stop_reason": (choices[0].get("finish_reason") if choices else None),
        "completion_chars": len(completion_text),
        "prefix_cache_before": prefix_cache_before,
        "prefix_cache_after": prefix_cache_after,
    }
    # First and last few completion chars only; full text lives in pier's
    # trajectory file. Keeping sidecars small.
    record["completion_head"] = completion_text[:400]
    record["completion_tail"] = completion_text[-200:] if len(completion_text) > 600 else ""

    _persist(trial, task, turn, record, routed_arr)

    if routed_arr is None:
        log.warning("trial=%s task=%s turn=%s: routed_experts MISSING from response",
                    trial, task, turn)
    else:
        log.info("trial=%s task=%s turn=%s: captured routing %s",
                 trial, task, turn, routed_arr.shape)

    # Return cleaned payload back to the agent.
    return JSONResponse(payload)


def _forward_headers(req: Request) -> dict[str, str]:
    """Drop hop-by-hop and our internal trial/turn headers when forwarding."""
    skip = {"host", "content-length", "connection", "x-trial-id", "x-task-id", "x-turn-idx"}
    return {k: v for k, v in req.headers.items() if k.lower() not in skip}


def _trial_meta(
    request: Request,
    x_trial_id: str | None,
    x_task_id: str | None,
    x_turn_idx: str | None,
) -> tuple[str, str, str]:
    trial = x_trial_id or request.headers.get("x-trial-id") or os.environ.get("PIER_TRIAL_ID", "")
    task = x_task_id or request.headers.get("x-task-id") or os.environ.get("PIER_TASK_ID", "")
    turn = x_turn_idx or request.headers.get("x-turn-idx") or ""
    return trial, task, turn


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_trial_id: str | None = Header(default=None),
    x_task_id: str | None = Header(default=None),
    x_turn_idx: str | None = Header(default=None),
):
    body = await request.json()
    trial, task, turn = _trial_meta(request, x_trial_id, x_task_id, x_turn_idx)
    return await _proxy_json("/v1/chat/completions", body, trial, task, turn,
                             _forward_headers(request))


@app.post("/v1/completions")
async def completions(
    request: Request,
    x_trial_id: str | None = Header(default=None),
    x_task_id: str | None = Header(default=None),
    x_turn_idx: str | None = Header(default=None),
):
    body = await request.json()
    trial, task, turn = _trial_meta(request, x_trial_id, x_task_id, x_turn_idx)
    return await _proxy_json("/v1/completions", body, trial, task, turn,
                             _forward_headers(request))


@app.get("/health")
async def health():
    try:
        r = await client.get("/health", timeout=2.0)
        return Response(status_code=r.status_code, content=r.content)
    except Exception:
        return Response(status_code=503, content=b"upstream-unreachable")


@app.get("/v1/models")
async def models(request: Request):
    r = await client.get("/v1/models", headers=_forward_headers(request))
    return Response(content=r.content, status_code=r.status_code,
                    headers={"content-type": r.headers.get("content-type", "application/json")})


# Catch-all GET passthrough for anything else the agent might probe.
@app.api_route("/{path:path}", methods=["GET"])
async def passthrough(path: str, request: Request):
    r = await client.get("/" + path, headers=_forward_headers(request))
    return Response(content=r.content, status_code=r.status_code,
                    headers={"content-type": r.headers.get("content-type", "application/json")})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="info")
