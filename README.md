# expert-routing-analysis

A drop-in framework that runs a real-world SWE workload against a local
vLLM instance and captures, per layer per token, which experts the MoE
router selected. The captured tensors get assembled into the three
canonical analysis charts (per-(layer, expert) heatmap, coverage CDF
across block sizes, experts-required bar chart) — same style as the
reference `graph{1,2,3}.png` in this directory.

## Quick start (one command)

```bash
# 1. drop your MiniMax key into secrets/ (only needed if you want the
#    opencode operator agent to drive the run end-to-end)
echo 'sk-yourkey' > secrets/minimax_api_key.txt

# 2. point MODEL_ID at whatever HF id this recipe serves
export MODEL_ID=cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit

# 3. full pipeline: setup -> smoke test -> run -> analyze
./bootstrap.sh run 10        # or 1, or 20

# alternative: have the opencode operator agent drive it
./bootstrap.sh agent 10
```

The pipeline boots vLLM with `--enable-return-routed-experts`, starts a
transparent proxy on :8001 that peels off the `routed_experts` field of
each response, runs `pier` with `mini-swe-agent` against a curated subset
of deep-swe tasks, and produces a report under
`work/runs/<job>/report/`.

## Switching models / recipes

The only knob you usually need is `MODEL_ID`. Everything else is
auto-detected by `playbook/derive_model_defaults.py`, which reads the HF
config and picks:

  * a short served-model alias (e.g. `qwen3.6-35b-a3b-awq`), so the OpenAI
    client doesn't have to use the long org/name id
  * a `--tool-call-parser` matching the chat template's format (qwen3_xml,
    deepseek_v3, minimax_m2, llama3_json, hermes, ...) — mini-swe-agent
    sends `tool_choice: "auto"` on every turn and vLLM rejects that without
    a parser

You can probe what defaults the framework would pick for a recipe before
running anything:

```bash
$PY playbook/derive_model_defaults.py deepseek-ai/DeepSeek-V3
{"model_id": "deepseek-ai/DeepSeek-V3", "alias": "deepseek-v3",
 "tool_call_parser": "deepseek_v3", "have_local_config": false, ...}
```

Override any of these from the env when the auto-pick is wrong:

```bash
MODEL_ALIAS=my-alias TOOL_CALL_PARSER=hermes ./bootstrap.sh run 1
```

When the operator agent drives the run, it makes the same probe call,
recognizes the vLLM error patterns (`The model "X" does not exist`,
`tool choice requires --enable-auto-tool-choice`, `unknown tool parser`),
and proposes a single retry with the right override.

## What this requires

- A GPU box with vLLM (>= 0.19) installed and importable.
- vLLM must support the `--enable-return-routed-experts` flag (the
  RoutedExpertsCapturer machinery from the sgl fork). `setup.sh` probes
  this and aborts cleanly if it's missing.
- A MoE model (the default is `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`, which
  fits on a single A100 40GB).
- `docker` accessible to the running user (pier needs it for trial sandboxes).
- Internet access during setup (for installing pier, mini-swe-agent, deep-swe).

## Watchdogs

| N | total budget | per-task budget |
|---|--------------|-----------------|
| 1 | 15 min | 9 min |
| 10 | 90 min | 9 min |
| 20 | 180 min | 9 min |

`run.sh` enforces the total budget with `timeout(1)`; per-task budgets
are baked into the curated dataset's `task.toml` by `prepare_curated.py`.

## Layout

```
bootstrap.sh                   # single entry point
playbook/
  setup.sh                     # idempotent installer + probe
  smoke.sh                     # tiny capture validation
  run.sh                       # boots vllm+proxy, runs pier, builds report
  teardown.sh                  # best-effort cleanup
  probe_vllm.py                # tier A/B/C capture detection
  vllm_compat.py               # flag shim
  routing_proxy.py             # FastAPI proxy that persists routed_experts
  prepare_curated.py           # materializes curated deep-swe subset
  config/
    pier-mini-vllm.yaml        # mini-swe-agent -> proxy wiring
    curated-{1,10,20}.txt      # frozen task IDs
  analysis/
    build_report.py            # 3 graphs + summary.json
    describe_with_mmx.sh       # natural-language graph descriptions
  opencode/agent/router-bench.md   # operator agent
secrets/                       # MiniMax key goes here (gitignored)
work/runs/<job>/               # per-run artifacts
  routing/turn*_req_*.npz      # captured top-k expert IDs per request
  routing/turn*_req_*.json     # sidecar metadata
  pier-jobs/                   # mini-swe trial directories
  logs/                        # vllm, proxy, pier logs
  report/                      # graph1/2/3 + summary
```
