#!/usr/bin/env python3
"""Aggregate every captured .npz under a job dir into the three canonical
graphs (heatmap, coverage CDF, coverage-threshold bars) plus a small
machine-readable summary.

Style intentionally tracks the three reference charts in the workspace:
  graph1.png — log-scale per-(layer, expert) routed-count heatmap
  graph2.png — coverage CDF, fanned over block sizes B ∈ {1,2,4,8,16,32,64,128}
  graph3.png — bar chart of experts required at coverage targets {50,80,90,95,99}%
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np


BLOCK_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
COVERAGE_TARGETS = [0.50, 0.80, 0.90, 0.95, 0.99]


def _load_all(job_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Return stacked (total_tokens, num_layers, top_k) and list of sidecar dicts.

    Concatenates per-task to preserve token order within each task; tasks
    are concatenated in lexicographic order. Returns an empty array if
    nothing was captured.
    """
    npzs = sorted(job_dir.rglob("routing/*.npz"))
    if not npzs:
        return np.zeros((0, 0, 0), dtype=np.int32), []
    arrs: list[np.ndarray] = []
    metas: list[dict] = []
    for p in npzs:
        try:
            arr = np.load(p)["routed_experts"]
        except Exception as e:
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        sidecar = p.with_suffix(".json")
        meta: dict = {}
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text())
            except Exception:
                pass
        meta["_npz"] = str(p)
        meta["_shape"] = list(arr.shape)
        arrs.append(arr.astype(np.int32))
        metas.append(meta)
    if not arrs:
        return np.zeros((0, 0, 0), dtype=np.int32), metas
    # Pad to common (layers, top_k) — they should already match but be safe.
    L = max(a.shape[1] for a in arrs)
    K = max(a.shape[2] for a in arrs)
    fixed = []
    for a in arrs:
        if a.shape[1] == L and a.shape[2] == K:
            fixed.append(a)
        else:
            pad = np.zeros((a.shape[0], L, K), dtype=np.int32)
            pad[:, : a.shape[1], : a.shape[2]] = a
            fixed.append(pad)
    return np.concatenate(fixed, axis=0), metas


def _infer_active_layers(stacked: np.ndarray) -> np.ndarray:
    """Indices of layers that actually saw any routing — drops linear-attn /
    non-MoE layers whose entries stay at zero."""
    if stacked.size == 0:
        return np.array([], dtype=np.int64)
    per_layer_nonzero = (stacked.sum(axis=(0, 2)) > 0)
    return np.where(per_layer_nonzero)[0]


def _num_experts(stacked: np.ndarray) -> int:
    if stacked.size == 0:
        return 0
    return int(stacked.max()) + 1


def _per_layer_expert_counts(stacked: np.ndarray, num_experts: int) -> np.ndarray:
    """Shape (num_layers, num_experts), int64 count of routings per cell."""
    if stacked.size == 0:
        return np.zeros((0, num_experts), dtype=np.int64)
    T, L, K = stacked.shape
    flat = stacked.reshape(T, L * K)
    # We bin per layer (each layer's K columns get binned together).
    out = np.zeros((L, num_experts), dtype=np.int64)
    for l in range(L):
        # Tokens in this layer come from columns [l*K : (l+1)*K] of flat.
        col_start = l * K
        ids = stacked[:, l, :].reshape(-1)
        # Mask any sentinel zeros from layers that don't actually route
        # (we'll filter those layers out before plotting).
        bc = np.bincount(ids, minlength=num_experts)
        out[l] = bc[:num_experts]
    return out


def _coverage_curves(stacked: np.ndarray, num_experts: int,
                      active_layers: np.ndarray) -> dict[int, np.ndarray]:
    """For each block size B, return cumulative fraction of routing mass
    covered by the top-N experts (N along x-axis), aggregated globally
    across all active layers.

    "Block of size B" means: average the K one-hot expert assignments over
    a contiguous window of B tokens (within each layer) before ranking.
    This matches the reference plot's interpretation — at B=1 the mass is
    per-token, at B=128 it's almost uniform.
    """
    out: dict[int, np.ndarray] = {}
    if stacked.size == 0 or len(active_layers) == 0:
        return {b: np.zeros(num_experts) for b in BLOCK_SIZES}

    T = stacked.shape[0]
    # Build a per-token "soft" usage histogram = sum of one-hot over K
    # (so each token contributes K/total to its experts). Then aggregate
    # by block-mean and sum across blocks and active layers.
    K = stacked.shape[2]
    for B in BLOCK_SIZES:
        # Pad T up to multiple of B then group.
        pad = (-T) % B
        if pad:
            padded = np.concatenate(
                [stacked, np.zeros((pad,) + stacked.shape[1:], dtype=stacked.dtype)], axis=0)
        else:
            padded = stacked
        Tp = padded.shape[0]
        nb = Tp // B
        # Build a (nb, L, num_experts) histogram by block.
        hist = np.zeros((nb, num_experts), dtype=np.float64)
        for l_idx in active_layers:
            layer_ids = padded[:, l_idx, :].reshape(nb, B * K)
            # bincount per block; do it via np.add.at for speed
            # block index for each entry:
            block_idx = np.repeat(np.arange(nb), B * K)
            np.add.at(hist, (block_idx, layer_ids.ravel()), 1.0)
        # For each block, rank experts by frequency, build CDF.
        ranked = np.sort(hist, axis=1)[:, ::-1]   # nb x num_experts
        # Normalize each block row to sum=1 (avoid div by zero)
        row_sums = ranked.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        ranked_norm = ranked / row_sums
        cdf = np.cumsum(ranked_norm, axis=1)
        # Average CDFs across blocks
        out[B] = cdf.mean(axis=0)
    return out


def _experts_required_for_coverage(curves: dict[int, np.ndarray]) -> dict[int, dict[float, int]]:
    out: dict[int, dict[float, int]] = {}
    for B, c in curves.items():
        out[B] = {}
        for tgt in COVERAGE_TARGETS:
            idx = int(np.searchsorted(c, tgt) + 1) if len(c) else 0
            out[B][tgt] = min(idx, len(c))
    return out


def plot_heatmap(counts: np.ndarray, active_layers: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    # Filter to active layers only so we don't waste rows on dead linear-attn layers.
    sub = counts[active_layers] if len(active_layers) else counts
    # +1 so log scale shows zero cells as the darkest color.
    img = ax.imshow(sub + 1, aspect="auto", origin="lower",
                    norm=__import__("matplotlib").colors.LogNorm(),
                    cmap="plasma")
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("MoE layer index (active layers only)")
    ax.set_title("Routed-count per (layer, expert)")
    # Recompute layer labels so y-axis maps back to true model layer index.
    if len(active_layers):
        step = max(1, len(active_layers) // 12)
        ax.set_yticks(range(0, len(active_layers), step))
        ax.set_yticklabels([str(int(active_layers[i])) for i in range(0, len(active_layers), step)])
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Routed count (+1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cdf(curves: dict[int, np.ndarray], num_experts: int, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = __import__("matplotlib").colormaps["viridis"]
    xs = np.arange(1, num_experts + 1)
    for i, B in enumerate(BLOCK_SIZES):
        c = curves.get(B)
        if c is None:
            continue
        ax.plot(xs[: len(c)], c, label=f"B={B}",
                color=cmap(i / max(1, len(BLOCK_SIZES) - 1)), linewidth=1.8)
    for y in (0.50, 0.80, 0.95):
        ax.axhline(y, linestyle="--", color="grey", linewidth=0.7)
    ax.set_xlabel("Number of experts (ranked by frequency within block)")
    ax.set_ylabel("Fraction of routing mass covered")
    ax.set_title("Coverage CDF across MoE layers (global, by block size)")
    ax.set_xlim(0, num_experts)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bars(req: dict[int, dict[float, int]], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    targets = COVERAGE_TARGETS
    Bs = BLOCK_SIZES
    width = 0.8 / len(Bs)
    cmap = __import__("matplotlib").colormaps["tab10"]
    for i, B in enumerate(Bs):
        ys = [req.get(B, {}).get(t, 0) for t in targets]
        xs = np.arange(len(targets)) + (i - len(Bs) / 2) * width
        ax.bar(xs, ys, width=width, label=f"B={B}", color=cmap(i % 10))
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels([f"{int(t*100)}%" for t in targets])
    ax.set_xlabel("Coverage target")
    ax.set_ylabel("Experts required")
    ax.set_title("Experts required for coverage (global, all MoE layers)")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    job_dir = Path(args.job_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stacked, metas = _load_all(job_dir)
    print(f"loaded {len(metas)} requests; stacked shape={stacked.shape}")

    summary: dict = {
        "job_dir": str(job_dir),
        "n_requests": len(metas),
        "n_tokens": int(stacked.shape[0]) if stacked.size else 0,
        "n_layers": int(stacked.shape[1]) if stacked.size else 0,
        "top_k": int(stacked.shape[2]) if stacked.size else 0,
    }

    if stacked.size == 0:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print("no routing data captured; wrote empty summary only")
        return 0

    num_experts = _num_experts(stacked)
    active = _infer_active_layers(stacked)
    summary["num_experts_seen"] = num_experts
    summary["active_moe_layers"] = active.tolist()

    counts = _per_layer_expert_counts(stacked, num_experts)
    curves = _coverage_curves(stacked, num_experts, active)
    req = _experts_required_for_coverage(curves)
    summary["experts_required_for_coverage"] = {
        str(B): {f"{int(t*100)}%": v for t, v in d.items()}
        for B, d in req.items()
    }

    plot_heatmap(counts, active, out_dir / "graph1_heatmap.png")
    plot_cdf(curves, num_experts, out_dir / "graph2_cdf.png")
    plot_bars(req, out_dir / "graph3_bars.png")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote graphs and summary to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
