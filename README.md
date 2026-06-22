# moe-routing-study

A drop-in framework that runs a real-world software-engineering workload
against a local vLLM instance serving any Mixture-of-Experts model, and
captures **which experts the router selected, per layer, per token**
across the whole agent trajectory. The captured tensors are assembled
into three canonical analysis charts — per-(layer, expert) heatmap,
coverage CDF across token-aggregation block sizes, and experts-required
bar chart — matching the reference `graph{1,2,3}.png` in this repo.

This is intentionally one script (`bootstrap.sh`) you drop on any GPU
instance that already has vLLM ≥ 0.21. It installs everything else
(pier, mini-swe-agent, deep-swe, proxy venv), boots vLLM with the
capture flag, runs a curated subset of the [deep-swe](https://github.com/datacurve-ai/deep-swe)
benchmark through the [pier](https://github.com/datacurve-ai/pier)
harness with `mini-swe-agent`, persists routing tensors as the agent
works, and produces a per-job report.

---

## What you get out

For each job, under `work/runs/<job>/`:

| Artifact | Shape / contents |
|---|---|
| `**/routing/turn*_req_*.npz` | one file per API request. `routed_experts: uint8/uint16 array (num_tokens, num_layers, top_k)` — top-k expert IDs for every forward-passed token (prompt + completion, minus the last sampled token). |
| `**/routing/turn*_req_*.json` | sidecar metadata: `request_id`, sampling params, usage, latency, `stop_reason`, prefix-cache hit counters scraped from `/metrics`, completion head/tail. |
| `pier-jobs/<trial>/trajectory.json` | mini-swe-agent ATIF v1.7 trajectory for the trial (pier output). |
| `pier-jobs/<trial>/verifier/reward.json` | binary reward + f2p / p2p pass fractions from the harbor verifier. |
| `report/graph1_heatmap.png` | log-scale per-(layer, expert) routed-count heatmap. |
| `report/graph2_cdf.png` | cumulative coverage-vs-#experts curves, fanned over block sizes `B ∈ {1,2,4,8,16,32,64,128}`. |
| `report/graph3_bars.png` | experts required to hit `{50, 80, 90, 95, 99}%` coverage, grouped by block size. |
| `report/summary.json` | aggregate counters: `n_requests`, `n_tokens`, `n_layers`, `top_k`, `num_experts_seen`, `active_moe_layers`, and the experts-required matrix. |
| `<job>.tar.gz` | a tarball of all of the above for shipping. |

Worked example from the local validation run (Qwen3.5-MoE-A3B AWQ, 1
deep-swe task, mini-swe-agent in a harbor docker sandbox): 11 chat
completions, **70 177 tokens with full routing**, 40 layers × top-k 8,
256 experts seen, all under a 15-minute wall-clock budget.

---

## Prerequisites

| Component | Why | Verified version |
|---|---|---|
| Linux + NVIDIA GPU | vLLM runs here | A100 40 GB, driver 570 / CUDA 12.8 |
| vLLM ≥ 0.21 with `RoutedExpertsCapturer` | the `--enable-return-routed-experts` flag is what makes everything work | `0.21.1rc1.dev400+g7e53283b1` (sgl-project fork) |
| A MoE model | nothing to capture on a dense model | Qwen3.5-MoE, DeepSeek-V3, MiniMax-M1/M2, Mixtral, etc. |
| Python 3.11+ in the vLLM environment | bootstrap searches for it | 3.11.15 |
| `docker`, with caller in the `docker` group | pier needs it to build the harbor sandbox + egress proxy containers | 27.5.1 |
| Internet during setup | pulls pier, mini-swe-agent, deep-swe tasks, harbor docker images | — |
| Optional: a [MiniMax](https://www.minimax.io/) key | only needed if you want the opencode operator agent to drive the run | — |

If vLLM is missing the capture flag, `setup.sh` exits with
`SETUP: FAIL vllm capture unavailable (tier=X). Pin vllm>=0.21 with
routed-experts support`. The probe checks two ways and forbids the
forward-hook fallback (too slow / skews data).

---

## Quick start

```bash
git clone git@github.com:lutetjeff/moe-routing-study.git
cd moe-routing-study

# 1. (optional) opencode operator key
echo 'sk-yourkey' > secrets/minimax_api_key.txt

# 2. tell the framework which HF id this vLLM recipe serves
export MODEL_ID=cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit

# 3. full pipeline: setup -> smoke test -> run -> analyze -> tarball
./bootstrap.sh run 10
```

`./bootstrap.sh run N` is the supported entry point. `N ∈ {1, 10, 20}`
maps to the curated task lists in `playbook/config/curated-N.txt` and to
the wall-clock budget below.

If you'd rather have an LLM operator drive (probe the model, run setup,
catch and retry transient failures, surface a one-line "stop your GPU"
abort if the smoke test breaks), use:

```bash
./bootstrap.sh agent 10
```

Its messages stream live; a transcript is also saved to
`work/logs/opencode-<timestamp>.log`. Set `BOOTSTRAP_VERBOSE=1` to also
surface opencode's internal logs.

---

## Bootstrap commands

`bootstrap.sh` is a thin dispatcher around the playbook:

| Command | What it does |
|---|---|
| `./bootstrap.sh setup` | install pier + mini-swe-agent, build the proxy venv (Python 3.11, fastapi/uvicorn/httpx/numpy/matplotlib), clone deep-swe, probe vLLM, write the MiniMax credential if present. Idempotent. |
| `./bootstrap.sh smoke` | runs setup, then boots vLLM, sends one tiny `/v1/completions`, decodes `choices[0].routed_experts`, asserts the array is `(T, L, K)` with non-zero layers. Tears everything down. Useful as a 5-minute "is this going to work?" probe before paying for a long run. |
| `./bootstrap.sh run [N]` | setup + smoke + the full N-task playbook. Aborts on smoke failure with a "stop your GPU instance" message. |
| `./bootstrap.sh agent [N]` | setup, then hands control to the opencode `router-bench` operator agent (default `minimax/MiniMax-M3`). Streams live. |
| `./bootstrap.sh teardown` | best-effort cleanup of dangling vLLM, proxy, pier sandbox containers, and egress-proxy containers. Safe to run anytime. |
| `./bootstrap.sh report <job>` | rebuild the analysis graphs + run mmx descriptions for an existing job dir (useful after pulling someone else's tarball). |

---

## Per-model configuration

`MODEL_ID` is the only knob you usually need. `playbook/derive_model_defaults.py`
inspects the HF config in your cache (or falls back to substring
inference against the id) and emits a JSON line:

```bash
$PY playbook/derive_model_defaults.py cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit
{"model_id":"...","alias":"qwen3.6-35b-a3b-awq",
 "tool_call_parser":"qwen3_xml","have_local_config":true,
 "max_position_embeddings":262144,
 "architectures":["Qwen3_5MoeForConditionalGeneration"],
 "model_type":"qwen3_5_moe"}
```

`run.sh` calls this once per job and uses it to:

1. set vLLM's `--served-model-name <MODEL_ID> <alias>` so OpenAI
   clients can address the model with the short name;
2. set vLLM's `--tool-call-parser <parser>` so `tool_choice: "auto"`
   works (mini-swe-agent sends it on every turn);
3. substitute `__MODEL_ALIAS__`, `__API_BASE__`, `__MODEL_CLASS__` into
   `playbook/config/pier-mini-vllm.yaml.template` and write the resolved
   YAML to `work/runs/<job>/pier-mini-vllm.yaml` for replay.

When auto-detection is wrong, override from the environment:

| Variable | When to set it | Example |
|---|---|---|
| `MODEL_ID` | always (or take the default) | `deepseek-ai/DeepSeek-V3` |
| `MODEL_ALIAS` | vLLM returns `404 The model "X" does not exist`. Use the exact name shown in the error. | `qwen3-coder` |
| `TOOL_CALL_PARSER` | probe returns `unknown`, or vLLM logs `unknown tool parser`, or LiteLLM raises `BadRequestError: "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`. | `deepseek_v3` |
| `MODEL_CLASS` | only for models that serve **only** the Responses API. Default `litellm` (hits `/v1/chat/completions`, which is where capture lives). Setting `litellm_response` will break capture — don't unless you're rewiring the proxy too. | `litellm` |
| `MAX_MODEL_LEN` | vLLM rejects the default 16384 because the model's `max_position_embeddings` is lower. Use the probe value. | `32768` |
| `GPU_MEM_UTIL` | the model + KV doesn't fit at 0.85 of GPU mem. | `0.92` |
| `PROXY_BASE_URL_FROM_CONTAINER` | Container can't reach `172.17.0.1:$PROXY_PORT`. Common on rootless docker. | `http://192.168.65.2:8001/v1` |
| `VLLM_PORT` / `PROXY_PORT` | something else owns 8000/8001 on the host. | `9000` / `9001` |
| `CONCURRENCY` | you have enough VRAM for parallel trials (rare on a single A100). | `2` |
| `EXTRA_VLLM_ARGS` | tunes you want appended to the vLLM serve invocation. | `--swap-space 16` |

`./bootstrap.sh setup` warns when `MODEL_ID` isn't already cached at
`$HF_HOME` and would trigger a multi-GB download.

### Recognized model families

`derive_model_defaults.py` ships a substring → parser table covering
Qwen3 / Qwen3-MoE / Qwen3-coder, Qwen2, DeepSeek-V3 / V3.1 / V3.2 / V4,
MiniMax-M1 / M2, Llama-3 / Llama-4, Mixtral / Mistral, Granite /
Granite-4, Gemma-4, GLM-4.5 / 4.7, Hermes-style generic, InternLM,
Kimi-K2, Step3 / Step3.5, Phi4-mini, Hunyuan-A13B. Unknown families
return `tool_call_parser: "unknown"` so `run.sh` fails clean.

Edit the `_FAMILY_TO_PARSER` list at the top of
`playbook/derive_model_defaults.py` to add more.

---

## What `run.sh` actually does

```
0. probe MODEL_ID  -> alias, parser, max_position_embeddings
1. prepare_curated.py
     - copy curated tasks into work/runs/<job>/dataset/
     - patch [agent].timeout_sec = 540s
     - patch [environment].allow_internet = true     (see Troubleshooting #2)
2. boot vLLM         (work/runs/<job>/logs/vllm.log)
     vllm serve $MODEL_ID
       --served-model-name $MODEL_ID $MODEL_ALIAS
       --enable-return-routed-experts
       --enable-auto-tool-choice
       --tool-call-parser $TOOL_CALL_PARSER
       --max-model-len ${MAX_MODEL_LEN:-16384}
       --gpu-memory-utilization ${GPU_MEM_UTIL:-0.85}
3. boot routing proxy (work/runs/<job>/logs/proxy.log)
     uvicorn routing_proxy:app on :8001
4. wait for both /health (max 5 min)
5. materialize pier-mini-vllm.yaml from template
6. run pier under timeout(1) wrapper with the job budget
     pier run -p <curated dataset> --env docker -c <materialized yaml>
              --jobs-dir work/runs/<job>/pier-jobs --n-concurrent 1
7. analysis/build_report.py over every captured .npz
8. tar the job dir to work/runs/<job>.tar.gz
9. tear down vLLM + proxy + dangling sandbox containers
10. print one of:
    RUN: OK job=<name>
    RUN: PARTIAL job=<name> reason=job_budget_hit | pier_exit_<rc>
```

### Watchdogs

| `N` | `JOB_BUDGET_SEC` (total) | `TASK_BUDGET_SEC` (per-task) |
|---|---|---|
| 1 | 900 | 540 |
| 10 | 5400 | 540 |
| 20 | 10800 | 540 |
| other | `540 N + 300` | 540 |

Both are env-overridable: `JOB_BUDGET_SEC=3600 ./bootstrap.sh run 10`.
The total budget is enforced by GNU `timeout`; the per-task budget is
written into the curated `task.toml`'s `[agent].timeout_sec` field so
harbor enforces it inside the sandbox.

---

## Routing proxy

`playbook/routing_proxy.py` is a ~300-line FastAPI app that forwards
`/v1/chat/completions` and `/v1/completions` (plus a passthrough for
`/v1/models` and `/health`) to vLLM, then on the response:

- pops `routed_experts` from each `choices[i]` (vLLM puts it
  **per-choice**, not at the top level);
- decodes the base64 `numpy.save` blob;
- persists to `${ROUTING_OUTPUT_DIR}/${ROUTING_JOB_ID}/<task>/<trial>/routing/turn<idx>_req_<id>.npz`
  along with a sidecar JSON;
- strips the field from the outgoing payload (so LiteLLM, which is
  strict about unknown fields, doesn't reject it);
- returns the cleaned payload to the agent.

It scrapes `/metrics` at most once per second for prefix-cache hit /
queue depth counters and embeds them in the sidecar so the analysis
stage can correlate routing decisions with serving state without joining
a Prometheus dump.

Streaming is intentionally collapsed to non-streaming on the wire to vLLM
because the routed-experts blob is only attached to the final response
JSON; the upstream stream would have to be buffered anyway to extract it.

Per-request sidecar shape:

```json
{
  "request_id": "chatcmpl-866d44...",
  "endpoint": "/v1/chat/completions",
  "trial_id": "", "task_id": "", "turn_idx": "",
  "model": "qwen3.6-35b-a3b-awq",
  "ts_start_ns": 1750000..., "ts_end_ns": 1750000..., "latency_ms": 1884.5,
  "sampling": {"temperature":0.2,"top_p":0.95,"max_tokens":4096},
  "usage": {"prompt_tokens":1532,"completion_tokens":217,"total_tokens":1749},
  "stop_reason": "tool_calls",
  "completion_head": "...", "completion_tail": "...",
  "prefix_cache_before": { "vllm:prefix_cache_queries_total{...}": 0 },
  "prefix_cache_after":  { "vllm:prefix_cache_queries_total{...}": 1 },
  "routed_experts_shape": [9575, 40, 8],
  "routed_experts_dtype": "uint8"
}
```

Correlation with a pier trial is by `request_id` joined against pier's
`pier-jobs/<trial>/trajectory.json`.

---

## Analysis

`playbook/analysis/build_report.py` globs every `*.npz` under the job
dir, stacks `(num_tokens_total, num_layers, top_k)`, infers active MoE
layers from "layers with any non-zero routings" (so hybrid linear/full
attention models don't have dead rows in the heatmap), and emits the
three reference graph styles plus a `summary.json`.

| Graph | What it answers |
|---|---|
| `graph1_heatmap.png` | "Is the load balanced across experts and layers?" Log-scale `(layer, expert)` histogram, plasma colormap, +1 offset so empty cells render dark instead of white. |
| `graph2_cdf.png` | "How concentrated is routing?" For each block size `B ∈ {1, 2, 4, 8, 16, 32, 64, 128}`, fraction of total routing mass covered by the top-N experts. B=1 is the most peaked; B=128 approaches uniform. Reference lines at 0.50 / 0.80 / 0.95. |
| `graph3_bars.png` | "How many experts must a system keep hot to cover X% of routing?" Grouped bar chart at coverage targets {50, 80, 90, 95, 99}%, fanned over the same block sizes. |

`playbook/analysis/describe_with_mmx.sh report/` calls
`mmx vision describe` on each generated PNG and writes a paragraph
summary alongside — useful for shipping a job report in a chat / email
without having to inline the images.

---

## Operator agent

`playbook/opencode/agent/router-bench.md` defines a primary opencode
agent (defaults to `minimax/MiniMax-M3`) that the `bootstrap.sh agent N`
entry point hands control to. The agent prompt enforces a fixed loop:

1. **Probe.** Run `derive_model_defaults.py` and save the result as ground truth.
2. **Setup.** Run `setup.sh`. Make one targeted fix attempt per known failure mode (missing `uv`, docker group, missing proxy deps) or print `OPERATOR: BLOCKED reason=<...> fix=<one-liner>` and stop.
3. **Smoke test.** If `SMOKE: FAIL`, print `OPERATOR: ABORT smoke_failed=<reason>` with a "stop this GPU instance" recommendation. Never proceed past a smoke failure — the routing data is the whole point.
4. **Main run.** Run `run.sh N`. Tail pier's log. Note transient failures (proxy 502, container died, OOM in prefill) for retry; abandon other failures.
5. **Retries.** At most one retry per task in the retry list, using `JOB_BUDGET_SEC=900 run.sh 1 <job>-retry-<task>`.
6. **Final summary.** Counts, tarball path, retried trials, any overrides used, report paths, warnings.

The "Model overrides" table from the previous section is repeated inside
the prompt so the agent can pick a parser without external lookup when
it sees `unknown tool parser` or `The model "X" does not exist`.

---

## Troubleshooting

Things we hit during validation. If you see one of these symptoms,
this is the fix:

| # | Symptom (in log) | Cause | Fix |
|---|---|---|---|
| 1 | `routed_experts: NoneType` at top level | vLLM puts the blob per-choice, not at the response root. | The proxy already reads `choices[i].routed_experts`. Don't put it back. |
| 2 | `ERR_ACCESS_DENIED` from squid in mini-swe-agent.txt | pier's egress proxy only allows Safe_ports (80, 443) — our proxy lives on 8001. | `prepare_curated.py` patches `[environment].allow_internet = true` so pier skips squid mediation for the agent container. The verifier section is left untouched at `false`. |
| 3 | `POST /v1/responses HTTP/1.1 405 Method Not Allowed` | mini-swe-agent's adapter auto-picks `litellm_response` for `openai/...` prefixes, which talks to `/v1/responses` — and vLLM doesn't carry `routed_experts` on that endpoint. | The pier YAML forces `model_class: litellm` (chat/completions adapter). |
| 4 | `NotFoundError: The model "X" does not exist` | served-model-name doesn't match what the client sent. | Set `MODEL_ALIAS` to the name vLLM reports in the 404. |
| 5 | `BadRequestError: "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser` | the model serves but vLLM refuses tool_choice without a parser. | `run.sh` already adds both flags; if you see this, the auto-picked `TOOL_CALL_PARSER` is wrong for this template. Override it (see table above). |
| 6 | `Smoke: FAIL reason=vllm_did_not_become_ready_in_300s` | torch.compile cold start can exceed 5 minutes on first boot. | Re-run; the second boot uses cached graphs. Or set `VLLM_USE_PRECOMPILED=1` if your build supports it. |
| 7 | proxy logs `routed_experts MISSING from response` | vLLM came up without the capture flag, or the model isn't actually MoE. | `setup.sh`'s probe should have failed first; re-run it. |
| 8 | `n_total_trials = 1, n_errored_trials = 1`, reward = 0 | mini-swe-agent ran but didn't produce a correct patch. This is normal — we don't care about reward, only about whether routing captured. | Confirm routing was captured: `find work/runs/<job> -name '*.npz' \| wc -l` should match the request count in `pier.log`. |

---

## Repo layout

```
bootstrap.sh                              single entry point (setup|smoke|run|agent|teardown|report)
README.md
graph1.png graph2.png graph3.png          reference analysis chart style
.gitignore                                excludes work/, secrets/*.txt (keeps .example)

playbook/
  setup.sh                                idempotent installer + vLLM capture probe
  smoke.sh                                ~5-min boot+probe+decode validation
  run.sh                                  end-to-end orchestrator with watchdog
  teardown.sh                             best-effort cleanup
  probe_vllm.py                           tier A/B detection, forbids tier C, requires >= 0.21
  derive_model_defaults.py                MODEL_ID -> {alias, tool_call_parser, max_seq_len, ...}
  vllm_compat.py                          flag shim used during probing
  routing_proxy.py                        FastAPI proxy: peels routed_experts, writes .npz + sidecar
  prepare_curated.py                      materializes curated subset, patches task.toml fields
  config/
    pier-mini-vllm.yaml.template          __MODEL_ALIAS__/__API_BASE__/__MODEL_CLASS__ placeholders
    curated-{1,10,20}.txt                 frozen task IDs (lang-balanced + 17 software domains)
  analysis/
    build_report.py                       3 graphs + summary.json
    describe_with_mmx.sh                  natural-language graph descriptions via MiniMax mmx
  opencode/
    agent/router-bench.md                 operator agent prompt

secrets/
  minimax_api_key.txt.example             copy to minimax_api_key.txt (gitignored)

work/                                     (gitignored) generated per-instance
  proxy-venv/                             python3.11 venv with fastapi/uvicorn/numpy/matplotlib
  deep-swe/                               cloned dataset
  runs/<job>/
    dataset/                              curated task copies with patched task.toml
    pier-mini-vllm.yaml                   materialized agent config for replay
    logs/{vllm,proxy,pier,prepare,analysis}.log
    <task>/<trial>/routing/turn*_req_*.{npz,json}
    pier-jobs/<job>/<trial>/              mini-swe trajectories, verifier outputs
    report/{graph1,2,3}_*.png, summary.json
  runs/<job>.tar.gz                       shippable bundle
  logs/opencode-<ts>.log                  agent-mode transcript
```

---

## Validation notes

The end-to-end pipeline was validated on a 40 GB A100 with vLLM
`0.21.1rc1.dev400+g7e53283b1` serving `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`,
running pier 0.3.0 + mini-swe-agent 2.4.2 against one curated deep-swe
task (`fd-deterministic-multi-key-sorting`). The harbor sandbox + egress
proxy containers came up, mini-swe-agent ran 11 turns through the
routing proxy, and 70 177 tokens of routing (`(T, 40, 8) uint8`) landed
on disk. mmx's auto-description of the resulting heatmap: *"generally
healthy MoE routing heatmap... no single expert dominates a row, and
no expert is universally ignored... mild per-layer load imbalance,
though nothing severe enough to suggest a routing collapse."*
