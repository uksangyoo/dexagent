"""Bound and monitor one locally owned experiment process group.

Polling limits are stop conditions, not kernel memory reservations. Core dumps
are disabled and each output file has an OS-enforced size cap. Never signals
another user's process or changes GPU operating modes.
"""

import argparse
import csv
import fcntl
import io
import json
import os
import resource
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

GIB = 1024**3


@dataclass(frozen=True)
class Limits:
    min_disk_free_gib: float = 20.0
    min_host_available_gib: float = 64.0
    max_process_rss_gib: float = 8.0
    max_gpu_used_gib: float = 12.0
    max_output_gib: float = 2.0
    max_file_mib: int = 256
    max_seconds: float = 3600.0


def available_memory():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].split()[0]) * 1024


def directory_bytes(directory):
    return sum(
        p.stat().st_size for p in Path(directory).rglob("*") if p.is_file() and not p.is_symlink()
    )


def gpu_inventory():
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    return {
        int(row[0]): {
            "uuid": row[1].strip(),
            "used_bytes": int(row[2]) * 1024**2,
            "utilization": int(row[3]),
        }
        for row in csv.reader(io.StringIO(raw))
    }


def group_memory(pgid):
    raw = subprocess.check_output(["ps", "-eo", "pid=,pgid=,rss="], text=True, timeout=10)
    return sum(
        int(rss) * 1024
        for _, group, rss in (line.split() for line in raw.splitlines())
        if int(group) == pgid
    )


def exceeded(sample, limits):
    tests = [
        (sample["disk_free_bytes"] < limits.min_disk_free_gib * GIB, "disk_headroom"),
        (
            sample["host_available_bytes"] < limits.min_host_available_gib * GIB,
            "host_memory_headroom",
        ),
        (sample["process_rss_bytes"] > limits.max_process_rss_gib * GIB, "process_memory_limit"),
        (sample["gpu_used_bytes"] > limits.max_gpu_used_gib * GIB, "gpu_memory_limit"),
        (sample["output_bytes"] > limits.max_output_gib * GIB, "output_size_limit"),
        (sample["elapsed_s"] >= limits.max_seconds, "wall_time_limit"),
    ]
    return [name for bad, name in tests if bad]


def child_limits(max_file_mib):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_mib * 1024**2,) * 2)


def _signal_group(pid, signum):
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass  # The child can exit between poll() and the signal.


def _group_running(pgid):
    raw = subprocess.check_output(["ps", "-eo", "pgid=,stat="], text=True, timeout=10)
    return any(
        int(group) == pgid and not state.startswith("Z")
        for group, state in (line.split() for line in raw.splitlines())
    )


def _stop_group(process, grace_s):
    _signal_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while _group_running(process.pid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.05)
    if _group_running(process.pid):
        _signal_group(process.pid, signal.SIGKILL)
    process.wait(timeout=10)


def supervise(command, output, *, gpu=None, limits=None, poll_s=5.0, grace_s=30.0, workdir=None):
    """Record preflight refusals as well as execution results."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "process.json").exists() or (output / "guard_result.json").exists():
        raise ValueError("Use a fresh output directory; preserve previous run evidence")
    started = time.monotonic()
    try:
        return _supervise(
            command,
            output,
            gpu=gpu,
            limits=limits,
            poll_s=poll_s,
            grace_s=grace_s,
            workdir=workdir,
        )
    except Exception as exc:
        result = {
            "launched": (output / "process.json").exists(),
            "exit_code": None,
            "stop_reasons": [f"supervisor_error: {type(exc).__name__}: {exc}"],
            "elapsed_s": time.monotonic() - started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (output / "guard_result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result


def _supervise(command, output, *, gpu=None, limits=None, poll_s=5.0, grace_s=30.0, workdir=None):
    limits = limits or Limits()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = None
    selected = None
    if gpu is not None:
        selected = gpu_inventory()[gpu]
        lock_dir = Path(__file__).resolve().parents[1] / "runs/locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock = (lock_dir / f"{selected['uuid']}.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        if selected["uuid"] in apps or selected["utilization"] > 5:
            lock.close()
            raise RuntimeError(f"GPU {gpu} is already in use")
    started = time.monotonic()
    process = None

    def sample():
        inventory = gpu_inventory() if selected else {}
        return {
            "elapsed_s": time.monotonic() - started,
            "disk_free_bytes": min(
                shutil.disk_usage(p).free for p in (output, Path("/tmp"), Path("/"))
            ),
            "host_available_bytes": available_memory(),
            "process_rss_bytes": group_memory(process.pid) if process is not None else 0,
            "gpu_used_bytes": inventory[gpu]["used_bytes"] if selected else 0,
            "output_bytes": directory_bytes(output),
        }

    first = sample()
    if failures := exceeded(first, limits):
        if lock is not None:
            lock.close()
        raise RuntimeError(f"Resource preflight failed: {failures}")
    env = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
    )
    if selected:
        env["CUDA_VISIBLE_DEVICES"] = selected["uuid"]
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    (output / "guard_config.json").write_text(
        json.dumps(
            {
                "command": command,
                "gpu": gpu,
                "gpu_uuid": selected["uuid"] if selected else None,
                "limits": asdict(limits),
                "workdir": str(Path(workdir).resolve()) if workdir else str(Path.cwd()),
            },
            indent=2,
        )
        + "\n"
    )
    reasons = []
    interrupted = False

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    previous = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (
            (output / "process.log").open("w") as log,
            (output / "resources.jsonl").open("w", buffering=1) as stats,
        ):
            process = subprocess.Popen(
                command,
                cwd=workdir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                preexec_fn=lambda: child_limits(limits.max_file_mib),
            )
            (output / "process.json").write_text(
                json.dumps({"pid": process.pid, "supervisor_pid": os.getpid()}) + "\n"
            )
            while process.poll() is None or _group_running(process.pid):
                try:
                    current = sample()
                    stats.write(json.dumps(current) + "\n")
                    reasons = exceeded(current, limits)
                except Exception as exc:
                    reasons = [f"monitor_error: {type(exc).__name__}: {exc}"]
                if interrupted:
                    reasons.append("stop_requested")
                if process.poll() is not None and _group_running(process.pid):
                    reasons.append("child_exited_with_live_descendants")
                if reasons:
                    _stop_group(process, grace_s)
                    break
                try:
                    process.wait(timeout=poll_s)
                except subprocess.TimeoutExpired:
                    pass
        result = {
            "launched": True,
            "exit_code": process.returncode,
            "stop_reasons": reasons,
            "elapsed_s": time.monotonic() - started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (output / "guard_result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        if process is not None and (process.poll() is None or _group_running(process.pid)):
            _stop_group(process, grace_s)
        for s, handler in previous.items():
            signal.signal(s, handler)
        if lock is not None:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required after --")
    result = supervise(
        command,
        args.output,
        gpu=args.gpu,
        limits=Limits(max_seconds=args.seconds),
        workdir=args.workdir,
    )
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["exit_code"] == 0 and not result["stop_reasons"] else 1)


if __name__ == "__main__":
    main()
