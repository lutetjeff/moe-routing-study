#!/usr/bin/env python3
"""Materialize a curated subset of the deep-swe dataset.

For each task ID in the curated list:
  * If a task.toml [agent].timeout_sec is larger than --task-timeout-sec,
    rewrite to enforce our per-task wall-clock budget (default 9 min, so
    a stuck task can't blow the whole-job watchdog).
  * Either symlink unchanged files or copy the task dir (we copy because
    pier walks the dataset dir and we don't want the agent confused by
    symlinks that go outside the dataset root).

Output is a dataset dir pier can be pointed at with `-p <out>`. We also
write a manifest snippet so the analysis stage can recover original
metadata (language, repo, category).
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import sys
from pathlib import Path


def patch_task_toml(text: str, agent_timeout: int) -> str:
    """Patch a deep-swe ``task.toml`` for our run conditions.

    Two changes scoped by [section] header so we don't clobber unrelated
    fields:

      * ``[agent].timeout_sec`` → ``agent_timeout`` (default 9 min). This is
        the per-task wall clock the framework enforces.
      * ``[environment].allow_internet`` → ``true``. The default is
        ``false``, which makes pier route the agent container through a
        squid egress proxy that only allows HTTP/HTTPS on the default
        ports (80/443). Our routing proxy lives on the host at :8001 —
        squid rejects that as ``ERR_ACCESS_DENIED`` (port not in
        Safe_ports). Flipping this to ``true`` skips squid and gives the
        container direct bridge access, which is what we need to reach
        ``172.17.0.1:8001``. The verifier section is left untouched.
    """
    out: list[str] = []
    section: str = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            out.append(line)
            continue
        if section == "agent" and re.match(r"\s*timeout_sec\s*=", line):
            out.append(f"timeout_sec = {agent_timeout}.0")
            continue
        if section == "environment" and re.match(r"\s*allow_internet\s*=", line):
            out.append("allow_internet = true")
            continue
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="deep-swe/tasks dir")
    ap.add_argument("--list", required=True, help="curated task-id list")
    ap.add_argument("--out", required=True, help="output dataset dir")
    ap.add_argument("--task-timeout-sec", type=int, default=540,
                    help="per-task wall clock cap (default 540s = 9 min)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    ids: list[str] = []
    for line in Path(args.list).read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            ids.append(s)

    manifest_in = src / "manifest.json"
    if manifest_in.exists():
        m = json.loads(manifest_in.read_text())
        wanted = {t["task_id"]: t for t in m.get("tasks", [])}
    else:
        wanted = {}

    selected: list[dict] = []
    missing: list[str] = []
    for tid in ids:
        tdir = src / tid
        if not tdir.is_dir():
            missing.append(tid)
            continue
        dst = out / tid
        shutil.copytree(tdir, dst)
        toml_path = dst / "task.toml"
        if toml_path.exists():
            toml_path.write_text(patch_task_toml(
                toml_path.read_text(), args.task_timeout_sec))
        meta = wanted.get(tid, {"task_id": tid})
        selected.append(meta)
        print(f"prepared: {tid}")

    out_manifest = {
        "schema_version": "1.0",
        "dataset": "deep-swe-curated",
        "task_count": len(selected),
        "task_timeout_sec": args.task_timeout_sec,
        "tasks": selected,
    }
    (out / "manifest.json").write_text(json.dumps(out_manifest, indent=2))

    print(f"\nwrote {len(selected)} tasks to {out}")
    if missing:
        print(f"MISSING from {src}: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
