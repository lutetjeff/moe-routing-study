#!/usr/bin/env python3
"""
Docker-free SWE-task runner. Sequential, one task at a time.

For each curated task: clone the upstream repo at the recorded base commit,
spawn ``mini-swe-agent`` as the unprivileged ``mse-runner`` user against a
fresh per-trial work copy, then capture (trajectory, model.patch) plus a
verifier/.pending sentinel under a pier-compatible trial directory.

The actual grading happens later via ``playbook/grade.py`` on a docker host
(the same machine, vast.ai-after, or any other) by building each task's
``tests/Dockerfile`` and feeding it our captured ``artifacts/model.patch``.

Design choices baked into this runner:

* **Mirror pier's mini-swe-agent invocation verbatim.** Same env vars
  (``LITELLM_LOCAL_MODEL_COST_MAP``, ``MSWEA_CONFIGURED``,
  ``MSWEA_COST_TRACKING``), same CLI flags
  (``--yolo --model --task --output -c mini.yaml -c agent.cost_limit=0
  -c <custom>.yaml -c model.model_class=litellm --exit-immediately``),
  same per-call ``extra_headers`` (so the routing proxy keeps bucketing
  by trial/task). The only thing we don't run is pier's sandbox.
* **Privilege drop.** Agent runs as a system user with no home dir, no
  shell. Trial workdir chowned to that user. Reduces blast radius of
  any ``rm -rf`` the agent might decide to execute.
* **Dumb sequential.** No retry, no concurrency. The operator agent at
  the top of bootstrap.sh handles retries.
* **Cached clones.** First task that needs a repo populates the cache
  under ``$WORK_DIR/clones/<task_id>/``; subsequent trials of the same
  task rsync from there into a fresh per-trial workdir.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path


def log(msg: str) -> None:
    print(f"[local_runner] {msg}", flush=True)


def slug_suffix() -> str:
    return secrets.token_hex(4)


def parse_task_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def read_instruction(task_dir: Path) -> str:
    """Read the task's instruction prompt (pier sends this verbatim)."""
    md = task_dir / "instruction.md"
    if md.exists():
        return md.read_text()
    # Fallback: some legacy tasks use instruction.txt
    txt = task_dir / "instruction.txt"
    if txt.exists():
        return txt.read_text()
    raise FileNotFoundError(f"no instruction.{{md,txt}} under {task_dir}")


def ensure_clone(repo_url: str, base_commit: str, cache_dir: Path) -> Path:
    """
    Fast-cache a shallow clone of ``repo_url`` at ``base_commit`` under
    ``cache_dir``. Re-uses an existing checkout when the commit matches.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    # cache key: <last url segment>-<commit short>
    repo_name = repo_url.rstrip("/").split("/")[-1]
    repo_name = re.sub(r"\.git$", "", repo_name)
    key = f"{repo_name}-{base_commit[:12]}"
    clone = cache_dir / key

    if clone.exists():
        try:
            head = subprocess.check_output(
                ["git", "-C", str(clone), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if head.startswith(base_commit) or base_commit.startswith(head):
                return clone
            log(f"clone cache stale at {clone} (HEAD={head[:12]}, want={base_commit[:12]}); refreshing")
        except subprocess.CalledProcessError:
            pass

    if clone.exists():
        shutil.rmtree(clone)

    clone.mkdir(parents=True)
    # Shallow fetch of just the base commit. Some hosts disallow this
    # for unadvertised commits; fall back to a deepening clone in that
    # case.
    subprocess.check_call(["git", "init", "-q", str(clone)])
    subprocess.check_call(
        ["git", "-C", str(clone), "remote", "add", "origin", repo_url]
    )
    try:
        subprocess.check_call(
            ["git", "-C", str(clone), "fetch", "--depth=1", "origin", base_commit],
            stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(["git", "-C", str(clone), "checkout", "-q", base_commit])
    except subprocess.CalledProcessError:
        log(f"shallow fetch failed for {repo_url} @ {base_commit[:12]}; doing full clone")
        shutil.rmtree(clone)
        subprocess.check_call(["git", "clone", "-q", repo_url, str(clone)])
        subprocess.check_call(["git", "-C", str(clone), "checkout", "-q", base_commit])
    return clone


def copy_workdir(src: Path, dst: Path) -> None:
    """Materialize a per-trial workdir from the cache."""
    # rsync is friendlier than copytree for big repos with symlinks.
    if shutil.which("rsync"):
        subprocess.check_call(
            ["rsync", "-a", "--delete", f"{src}/", f"{dst}/"]
        )
    else:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)


def make_world_writable(path: Path) -> None:
    """
    Give group + other read/write/execute on directories and read/write on
    files so the unprivileged ``mse-runner`` user can operate inside the
    per-trial workdir without us needing root to chown. The blast radius
    is bounded by the per-trial path: nothing outside it gets opened up.
    """
    subprocess.check_call(["chmod", "-R", "go+rwX", str(path)])


def build_mini_config(api_base: str, trial_id: str, task_id: str) -> str:
    """
    Build the per-trial mini-swe-agent config that pier would have written.
    We mirror pier's ``-c /tmp/mswea-config/custom.yaml`` content, with the
    correlation headers wired up so the routing proxy can bucket captures.
    """
    return (
        "model:\n"
        f"  litellm_model_args:\n"
        f"    api_base: {api_base}\n"
        f"    api_key: dummy\n"
        f"    extra_headers:\n"
        f"      X-Trial-Id: {trial_id}\n"
        f"      X-Task-Id: {task_id}\n"
        "agent:\n"
        "  step_limit: 50\n"
    )


def render_prompt(instruction: str) -> str:
    """
    Pier wraps the instruction with ``with_prompt_template`` decorator;
    by default that's identity unless a template is configured. For
    parity-with-pier we pass the instruction.md content unchanged.
    """
    return instruction


def run_one_task(
    task_dir: Path,
    job_dir: Path,
    proxy_base_url: str,
    model_id: str,
    model_alias: str,
    model_class: str,
    task_timeout_sec: int,
    clone_cache: Path,
    user: str,
    drop_tool: str,
) -> dict:
    """
    Execute one task end-to-end and return its summary dict. Failures are
    captured into the summary; we never raise out of this function so
    the outer loop can keep going.
    """
    task_id = task_dir.name
    cfg = parse_task_toml(task_dir / "task.toml")
    task_cfg = cfg.get("task", {})
    repo_url = task_cfg.get("repository_url") or cfg.get("metadata", {}).get("repository_url")
    base_commit = task_cfg.get("base_commit_hash") or cfg.get("metadata", {}).get("base_commit_hash")
    if not repo_url or not base_commit:
        return {
            "task_id": task_id,
            "status": "error",
            "reason": "missing repository_url or base_commit_hash in task.toml",
        }

    # Per-task budget: prefer the task.toml override (prepare_curated
    # writes our cap there), fall back to the CLI default.
    task_timeout = int(cfg.get("agent", {}).get("timeout_sec") or task_timeout_sec)

    suffix = slug_suffix()
    trial_id = f"{task_id}__local{suffix}"
    trial_dir = job_dir / trial_id
    workdir = trial_dir / "workdir"
    agent_dir = trial_dir / "agent"
    artifacts_dir = trial_dir / "artifacts"
    verifier_dir = trial_dir / "verifier"
    for d in (agent_dir, artifacts_dir, verifier_dir, workdir.parent):
        d.mkdir(parents=True, exist_ok=True)

    trial_log = trial_dir / "trial.log"
    trial_log_fp = trial_log.open("w")

    def tlog(msg: str) -> None:
        trial_log_fp.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        trial_log_fp.flush()

    tlog(f"task_id={task_id} repo={repo_url} base={base_commit[:12]} timeout={task_timeout}s")

    try:
        # 1. cache clone
        clone = ensure_clone(repo_url, base_commit, clone_cache)
        tlog(f"clone cache: {clone}")

        # 2. materialize per-trial workdir
        copy_workdir(clone, workdir)
        # Make sure the per-trial copy has its own .git pointing at base.
        subprocess.check_call(
            ["git", "-C", str(workdir), "config", "user.email", "mse-runner@localhost"]
        )
        subprocess.check_call(
            ["git", "-C", str(workdir), "config", "user.name", "mse-runner"]
        )
        make_world_writable(workdir)
        tlog(f"workdir prepared: {workdir} (world-writable for {user})")

        # 3. prompt + custom config
        instruction = render_prompt(read_instruction(task_dir))
        prompt_file = trial_dir / "agent" / "task.md"
        prompt_file.write_text(instruction)
        custom_cfg_path = trial_dir / "agent" / "custom.yaml"
        custom_cfg_path.write_text(
            build_mini_config(proxy_base_url, trial_id, task_id)
        )
        # mse-runner needs to read these too
        os.chmod(prompt_file, 0o644)
        os.chmod(custom_cfg_path, 0o644)

        # 4. spawn mini-swe-agent under the unprivileged user
        traj_path = agent_dir / "mini-swe-agent.trajectory.json"
        # Make sure the user can WRITE the trajectory file. Parent dir
        # was created above; broaden perms so mse-runner can create
        # the trajectory output here.
        make_world_writable(agent_dir)

        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": "/tmp",
            "OPENAI_API_KEY": "dummy",
            "OPENAI_BASE_URL": proxy_base_url,
            # Pier-mirrored mini-swe env
            "LITELLM_LOCAL_MODEL_COST_MAP": "true",
            "MSWEA_CONFIGURED": "true",
            "MSWEA_COST_TRACKING": "ignore_errors",
            # Tell LiteLLM not to read its own .env (would break on
            # ~/.config/mini-swe-agent/.env which mse-runner can't see)
            "MINI_SWE_AGENT_NO_GLOBAL_CONFIG": "true",
        }

        # The agent runs as ``openai/<alias>`` so LiteLLM picks the
        # OpenAI-chat-completions adapter against our proxy. We force
        # ``model_class=litellm`` via -c to match pier's default.
        model_for_agent = f"openai/{model_alias}" if model_alias else f"openai/{model_id}"

        # Use the absolute path setup.sh recorded if available — sudo's
        # secure_path strips ~/.local/bin from PATH, so just calling
        # ``mini-swe-agent`` won't resolve under the mse-runner shell.
        mini_bin_file = Path(os.environ.get("WORK_DIR", "")) / ".mini-swe-agent-bin"
        if mini_bin_file.exists():
            mini_bin = mini_bin_file.read_text().strip()
        else:
            mini_bin = shutil.which("mini-swe-agent") or "mini-swe-agent"

        mini_argv = [
            mini_bin,
            "--yolo",
            f"--model={model_for_agent}",
            f"--task={instruction}",
            f"--output={traj_path}",
            "-c", "mini.yaml",
            "-c", "agent.cost_limit=0",
            "-c", str(custom_cfg_path),
            f"-c", f"model.model_class={model_class}",
            "--exit-immediately",
        ]
        # sudo strips most of the environment (secure_path, env_reset).
        # ``--preserve-env=`` honors a list but HOME and other "sensitive"
        # vars are still scrubbed unless explicitly in env_keep. Pass
        # everything through an explicit ``env KEY=VAL ...`` prefix so
        # the values arrive at mini-swe-agent regardless of policy.
        env_prefix = ["env", "-i"] + [f"{k}={v}" for k, v in env.items()]
        if drop_tool == "sudo":
            cmd = ["sudo", "-n", "-u", user] + env_prefix + mini_argv
        else:
            cmd = ["runuser", "-u", user, "--"] + env_prefix + mini_argv

        agent_stdout = (agent_dir / "mini-swe-agent.txt").open("w")
        tlog(f"spawning: {shlex.join(cmd)}")
        t0 = time.time()
        proc = subprocess.Popen(
            cmd,
            stdout=agent_stdout,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(workdir),
            env={**os.environ, **env},
            preexec_fn=os.setsid,  # so we can kill the whole group on timeout
        )
        try:
            rc = proc.wait(timeout=task_timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            tlog(f"timeout after {task_timeout}s; sending SIGTERM")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                tlog("SIGTERM grace expired; SIGKILL")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                rc = proc.wait()
        elapsed = time.time() - t0
        agent_stdout.close()
        tlog(f"agent exit={rc} elapsed={elapsed:.1f}s timed_out={timed_out}")

        # 5. capture diff
        # mse-runner ran as a different user; root reads .git fine.
        # Use --binary so non-text changes survive.
        patch_path = artifacts_dir / "model.patch"
        try:
            diff = subprocess.check_output(
                ["git", "-C", str(workdir), "diff", "--binary", base_commit, "HEAD"],
                text=True,
            )
        except subprocess.CalledProcessError as e:
            tlog(f"git diff failed: {e}; trying against current index")
            diff = subprocess.check_output(
                ["git", "-C", str(workdir), "diff", "--binary"],
                text=True,
            )
        patch_path.write_text(diff)
        tlog(f"wrote {patch_path} ({len(diff)} bytes)")

        # 6. pending sentinel for grade.py
        (verifier_dir / ".pending").write_text(
            json.dumps({"task_id": task_id, "trial_id": trial_id})
        )

        # 7. trial config snapshot (mimic pier)
        (trial_dir / "config.json").write_text(json.dumps(
            {
                "task_id": task_id,
                "trial_id": trial_id,
                "model_id": model_id,
                "model_alias": model_alias,
                "model_class": model_class,
                "proxy_base_url": proxy_base_url,
                "repo_url": repo_url,
                "base_commit": base_commit,
                "task_timeout_sec": task_timeout,
                "agent_exit_code": rc,
                "agent_elapsed_sec": elapsed,
                "agent_timed_out": timed_out,
                "patch_bytes": len(diff),
                "schema_version": 1,
            },
            indent=2,
        ))

        return {
            "task_id": task_id,
            "trial_id": trial_id,
            "status": "captured" if not timed_out else "timed_out",
            "agent_exit": rc,
            "patch_bytes": len(diff),
            "elapsed_sec": elapsed,
        }
    except Exception as e:
        tlog(f"FATAL: {e!r}")
        (trial_dir / "exception.txt").write_text(repr(e))
        return {"task_id": task_id, "trial_id": trial_id, "status": "error", "reason": repr(e)}
    finally:
        trial_log_fp.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--jobs-dir", required=True, type=Path)
    ap.add_argument("--job-name", required=True)
    ap.add_argument("--proxy-base-url", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--model-alias", default="")
    ap.add_argument("--model-class", default="litellm")
    ap.add_argument("--task-timeout-sec", type=int, default=540)
    ap.add_argument("--clone-cache", required=True, type=Path)
    ap.add_argument("--user", required=True)
    ap.add_argument("--drop-tool", choices=["sudo", "runuser"], required=True)
    args = ap.parse_args()

    job_dir = args.jobs_dir / args.job_name
    job_dir.mkdir(parents=True, exist_ok=True)

    # Find tasks: every subdirectory of dataset/ that has a task.toml
    tasks = sorted(
        p for p in args.dataset.iterdir()
        if p.is_dir() and (p / "task.toml").exists()
    )
    log(f"found {len(tasks)} tasks under {args.dataset}")

    results = []
    for i, t in enumerate(tasks, 1):
        log(f"=== [{i}/{len(tasks)}] {t.name} ===")
        r = run_one_task(
            task_dir=t,
            job_dir=job_dir,
            proxy_base_url=args.proxy_base_url,
            model_id=args.model_id,
            model_alias=args.model_alias or args.model_id.split("/")[-1].lower(),
            model_class=args.model_class,
            task_timeout_sec=args.task_timeout_sec,
            clone_cache=args.clone_cache,
            user=args.user,
            drop_tool=args.drop_tool,
        )
        results.append(r)
        log(f"   -> {r.get('status')} {r.get('reason') or ''}")

    # Pier-shaped result.json (informational; rewards filled in later
    # by grade.py).
    (job_dir / "result.json").write_text(json.dumps(
        {
            "id": args.job_name,
            "started_at": None,
            "finished_at": None,
            "n_total_trials": len(results),
            "trials": results,
            "schema_version": 1,
            "graded": False,
        },
        indent=2,
    ))
    log(f"wrote {job_dir/'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
