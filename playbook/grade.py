#!/usr/bin/env python3
"""
Deferred grader for local-runner trials.

Walks ``<job>/pier-jobs/<job>/<trial>/`` directories looking for trials with
a ``verifier/.pending`` sentinel and no existing ``verifier/reward.json``.
For each, builds the task's verifier docker image from ``<task>/tests/``,
runs it with our captured ``artifacts/model.patch`` mounted in, and
collects ``reward.json`` + supporting files back out.

This intentionally re-uses each deep-swe task's own grading recipe
(``tests/Dockerfile`` + ``tests/test.sh`` + ``tests/grader.py``) instead
of re-implementing pier's verifier. The verifier image already knows how
to apply ``model.patch``, run the test suite, and emit the canonical
reward.json schema.

Idempotent: re-running skips trials that already have reward.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def log(msg: str) -> None:
    print(f"[grade] {msg}", flush=True)


def has_docker() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        subprocess.check_call(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def find_dataset_for_job(job_dir: Path) -> Path | None:
    """Find the per-job dataset dir written by prepare_curated.py."""
    candidate = job_dir / "dataset"
    if candidate.exists():
        return candidate
    return None


def find_task_dir(dataset_dir: Path, task_id: str) -> Path | None:
    candidate = dataset_dir / task_id
    if (candidate / "task.toml").exists() and (candidate / "tests" / "Dockerfile").exists():
        return candidate
    return None


def task_id_from_trial(trial: Path) -> str:
    # local_runner.py writes ``<task_id>__local<8hex>``
    name = trial.name
    return name.rsplit("__local", 1)[0]


def build_verifier_image(task_dir: Path, tag: str) -> bool:
    log(f"docker build -t {tag} {task_dir}/tests")
    try:
        subprocess.check_call(
            [
                "docker", "build",
                "--quiet",
                "-t", tag,
                str(task_dir / "tests"),
            ]
        )
        return True
    except subprocess.CalledProcessError as e:
        log(f"build failed: {e}")
        return False


def run_verifier(tag: str, trial: Path, timeout_sec: int) -> tuple[int, str]:
    """
    Execute the verifier image, mounting our patch and verifier outputs.
    Returns (exit_code, container_stdout_path).
    """
    artifacts_dir = trial / "artifacts"
    verifier_dir = trial / "verifier"
    verifier_dir.mkdir(exist_ok=True)
    # Ensure mountable empty subdir so verifier can write reports/
    (verifier_dir / "reports").mkdir(exist_ok=True)

    stdout_path = verifier_dir / "run.stdout.txt"
    with stdout_path.open("w") as fp:
        try:
            rc = subprocess.call(
                [
                    "docker", "run", "--rm",
                    "--network", "none",  # verifier should not need internet
                    "-v", f"{artifacts_dir}:/logs/artifacts:ro",
                    "-v", f"{verifier_dir}:/logs/verifier",
                    "--workdir", "/app",
                    "--user", "0:0",  # verifier scripts expect root
                    tag,
                    "bash", "/tests/test.sh",
                ],
                stdout=fp,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            rc = 124
            fp.write(f"\n[grade] verifier timed out after {timeout_sec}s\n")
    return rc, str(stdout_path)


def parse_reward(trial: Path) -> dict | None:
    rj = trial / "verifier" / "reward.json"
    rt = trial / "verifier" / "reward.txt"
    if rj.exists() and rj.stat().st_size > 0:
        try:
            return json.loads(rj.read_text())
        except json.JSONDecodeError:
            return None
    if rt.exists() and rt.stat().st_size > 0:
        try:
            return {"reward": float(rt.read_text().strip())}
        except ValueError:
            return None
    return None


def grade_trial(trial: Path, dataset: Path, verifier_timeout_sec: int) -> dict:
    pending = trial / "verifier" / ".pending"
    reward_path = trial / "verifier" / "reward.json"
    if reward_path.exists() and reward_path.stat().st_size > 0:
        # Already graded
        return {"trial": trial.name, "status": "skipped", "reason": "already_graded"}
    if not pending.exists():
        # Not a local-runner trial (pier-graded path, or something else)
        return {"trial": trial.name, "status": "skipped", "reason": "no_pending_sentinel"}

    task_id = task_id_from_trial(trial)
    task_dir = find_task_dir(dataset, task_id)
    if task_dir is None:
        return {"trial": trial.name, "status": "error", "reason": f"task dir not found for '{task_id}'"}

    patch = trial / "artifacts" / "model.patch"
    if not patch.exists() or patch.stat().st_size == 0:
        # No patch = empty diff = reward 0. Write a minimal reward.json
        # mirroring the pier shape so downstream tooling is happy.
        reward = {
            "reward": 0,
            "f2p_total": 0, "f2p_passed": 0, "p2p_total": 0, "p2p_passed": 0,
            "f2p": 0.0, "p2p": 0.0, "partial": 0.0,
            "skip_reason": "empty_or_missing_patch",
        }
        reward_path.write_text(json.dumps(reward))
        pending.unlink()
        return {"trial": trial.name, "status": "graded", "reward": 0, "skipped": "empty_patch"}

    # Read verifier timeout from task.toml if present
    try:
        with (task_dir / "task.toml").open("rb") as f:
            cfg = tomllib.load(f)
        verifier_timeout_sec = int(
            cfg.get("verifier", {}).get("timeout_sec") or verifier_timeout_sec
        )
    except Exception:
        pass

    # Build verifier image (idempotent — docker caches layers).
    tag = f"er-verifier-{task_id}:local"
    if not build_verifier_image(task_dir, tag):
        return {"trial": trial.name, "status": "error", "reason": "verifier_image_build_failed"}

    rc, stdout_path = run_verifier(tag, trial, verifier_timeout_sec)
    log(f"verifier rc={rc} for {trial.name}")

    reward = parse_reward(trial)
    if reward is None:
        # Verifier ran but didn't produce parseable reward; treat as 0.
        reward = {
            "reward": 0, "f2p_total": 0, "f2p_passed": 0,
            "p2p_total": 0, "p2p_passed": 0,
            "f2p": 0.0, "p2p": 0.0, "partial": 0.0,
            "skip_reason": f"verifier_exit_{rc}_no_reward_file",
        }
        reward_path.write_text(json.dumps(reward))

    # Remove pending sentinel so the trial is "done"
    try:
        pending.unlink()
    except FileNotFoundError:
        pass

    return {
        "trial": trial.name,
        "status": "graded",
        "verifier_exit": rc,
        "reward": reward.get("reward"),
        "partial": reward.get("partial"),
    }


def aggregate_results(job_dir: Path, results: list[dict]) -> None:
    """Update job-level result.json with aggregated reward stats."""
    res_path = job_dir / "result.json"
    if not res_path.exists():
        return
    try:
        data = json.loads(res_path.read_text())
    except json.JSONDecodeError:
        return

    data["graded"] = True
    data["grading_results"] = results
    graded = [r for r in results if r["status"] == "graded"]
    if graded:
        rewards = [r.get("reward") or 0 for r in graded]
        partials = [r.get("partial") or 0 for r in graded]
        data["aggregate"] = {
            "n_graded": len(graded),
            "n_pass": sum(1 for r in rewards if r),
            "mean_reward": sum(rewards) / len(rewards),
            "mean_partial": sum(partials) / len(partials),
        }
    res_path.write_text(json.dumps(data, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir", type=Path, help="path under work/runs/")
    ap.add_argument("--verifier-timeout-sec", type=int, default=1800)
    args = ap.parse_args()

    job_dir = args.job_dir.resolve()
    if not job_dir.exists():
        print(f"GRADE: FAIL job dir not found: {job_dir}")
        return 2
    if not has_docker():
        print("GRADE: BLOCKED reason=no_docker")
        return 3

    dataset = find_dataset_for_job(job_dir)
    if dataset is None:
        print(f"GRADE: FAIL dataset/ subdir not found under {job_dir}")
        return 2

    pier_jobs = job_dir / "pier-jobs"
    # local_runner.py writes pier-jobs/<job_name>/<trial>/. Find the
    # inner job-name dir (only one expected per local capture).
    job_inner_candidates = [p for p in (pier_jobs.iterdir() if pier_jobs.exists() else []) if p.is_dir()]
    if not job_inner_candidates:
        print(f"GRADE: FAIL no inner job dir under {pier_jobs}")
        return 2
    job_inner = job_inner_candidates[0]
    if len(job_inner_candidates) > 1:
        log(f"warning: multiple job dirs under {pier_jobs}; picking {job_inner.name}")

    trials = sorted(p for p in job_inner.iterdir() if p.is_dir())
    log(f"found {len(trials)} trials under {job_inner}")

    results = []
    for t in trials:
        log(f"=== {t.name} ===")
        r = grade_trial(t, dataset, args.verifier_timeout_sec)
        log(f"   -> {r['status']} reward={r.get('reward')}")
        results.append(r)

    aggregate_results(job_inner, results)

    graded = [r for r in results if r["status"] == "graded"]
    print(f"GRADE: OK graded={len(graded)} skipped={len(results) - len(graded)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
