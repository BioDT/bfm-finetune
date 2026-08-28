"""Machine profile discovery, GPU placement planning, and detached job launch.

A profile describes one machine's hardware (``machines/<hostname>.yaml``); when no file
exists it is auto-detected via ``nvidia-smi`` and ``/proc``, degrading to a CPU-only
profile rather than raising.
"""

import os
import socket
import subprocess
from pathlib import Path
from typing import Any

from bfm_finetune.eumon.common.runner import atomic_path, project_root, utcnow

try:
    import yaml
except ImportError:
    yaml = None

_NVIDIA_SMI_TIMEOUT_S = 10


# YAML: PyYAML when present, else the flat/one-level subset machines/*.yaml uses.

def _strip_comment(line: str) -> str:
    in_quote = None
    for i, ch in enumerate(line):
        if in_quote:
            if ch == in_quote:
                in_quote = None
        elif ch in "\"'":
            in_quote = ch
        elif ch == "#":
            return line[:i]
    return line


def _coerce_scalar(raw: str) -> Any:
    s = raw.strip()
    if not s:
        return None
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("null", "~", "none"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the flat ``key: value`` and one-level ``key:\\n  sub: value`` subset."""
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        key, value = key.strip(), value.strip()
        if indent == 0:
            if value:
                root[key] = _coerce_scalar(value)
                current = None
            else:
                current = {}
                root[key] = current
        elif current is not None:
            current[key] = _coerce_scalar(value)
    return root


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        needs_quote = value == "" or any(c in value for c in ":#\"'")
        return f'"{value}"' if needs_quote else value
    return str(value)


def _dump_simple_yaml(obj: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in obj.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            lines.extend(f"  {sk}: {_format_scalar(sv)}" for sk, sv in value.items())
        else:
            lines.append(f"{key}: {_format_scalar(value)}")
    return "\n".join(lines) + "\n"


def _yaml_load(text: str) -> dict[str, Any]:
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return _parse_simple_yaml(text)


def _yaml_dump(obj: dict[str, Any]) -> str:
    if yaml is not None:
        return yaml.safe_dump(obj, sort_keys=False)
    return _dump_simple_yaml(obj)


# -- nvidia-smi -----------------------------------------------------------------------

def _query_nvidia_smi(fields: str, nounits: bool = False) -> list[str]:
    fmt = "csv,noheader,nounits" if nounits else "csv,noheader"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", f"--format={fmt}"],
            capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [ln for ln in result.stdout.strip().splitlines() if ln.strip()]


def _detect_gpus() -> list[dict[str, Any]]:
    gpus = []
    for line in _query_nvidia_smi("index,name,memory.total"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            index = int(parts[0])
            mem_mib = float(parts[2].split()[0])
        except (ValueError, IndexError):
            continue
        gpus.append({"index": index, "name": parts[1], "memory_gb": round(mem_mib / 1024, 1)})
    return gpus


def _detect_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024 ** 2), 1)
    except OSError:
        pass
    return 0.0


def _detect_numa_nodes() -> int | None:
    node_dir = Path("/sys/devices/system/node")
    if not node_dir.is_dir():
        return None
    nodes = [p for p in node_dir.iterdir() if p.name[4:].isdigit() and p.name.startswith("node")]
    return len(nodes) or None


# -- profile -------------------------------------------------------------------------

def detect_profile() -> dict[str, Any]:
    """Build a profile from local hardware. Never raises: no driver means 0 GPUs."""
    gpus = _detect_gpus()
    names = {g["name"] for g in gpus}
    # Heterogeneous cards are out of scope for this benchmark; report the conservative
    # (minimum) VRAM so placement never assumes more memory than the smallest card has.
    model = names.pop() if len(names) == 1 else ("mixed" if names else None)
    vram_gb = min((g["memory_gb"] for g in gpus), default=0)

    return {
        "name": socket.gethostname(),
        "detected": utcnow(),
        "gpus": {"count": len(gpus), "model": model, "vram_gb": vram_gb},
        "cpu": {
            "cores": os.cpu_count() or 0,
            "ram_gb": _detect_ram_gb(),
            "numa_nodes": _detect_numa_nodes(),
        },
        "policy": {
            "strategy": "single_gpu_per_job",
            "max_concurrent_jobs": len(gpus) or 1,
            "numa_pin": False,
        },
    }


def load_profile(hostname: str | None = None) -> dict[str, Any]:
    """Read ``machines/<hostname>.yaml``; auto-detect the local machine if it is absent."""
    hostname = hostname or socket.gethostname()
    path = project_root() / "machines" / f"{hostname}.yaml"
    if path.exists():
        profile = _yaml_load(path.read_text())
        if isinstance(profile, dict) and profile:
            return profile
    return detect_profile()


def write_profile(profile: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    with atomic_path(path) as tmp:
        tmp.write_text(_yaml_dump(profile))
    return path


# -- placement -------------------------------------------------------------------------

def plan_placement(n_jobs: int, profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Round-robin ``n_jobs`` across the profile's GPUs.

    Only ``single_gpu_per_job`` is implemented: every workload in the experiment matrix
    fits one card.
    """
    gpus = profile.get("gpus") or {}
    gpu_count = int(gpus.get("count") or 0)
    policy = profile.get("policy") or {}
    numa_pin = bool(policy.get("numa_pin", False))
    numa_nodes = (profile.get("cpu") or {}).get("numa_nodes")

    plan = []
    for i in range(n_jobs):
        if gpu_count == 0:
            plan.append({"job_index": i, "gpu": None, "cuda_visible_devices": "", "numa_node": None})
            continue
        gpu = i % gpu_count
        numa_node = None
        if numa_pin and numa_nodes == 2:
            numa_node = 0 if gpu < gpu_count / 2 else 1
        plan.append({
            "job_index": i,
            "gpu": gpu,
            "cuda_visible_devices": str(gpu),
            "numa_node": numa_node,
        })
    return plan


# -- job launch and monitoring ---------------------------------------------------------

def launch_detached(cmd: list[str] | str, log_path: str | Path, gpu: int | None = None,
                     env: dict[str, str] | None = None, cwd: str | Path | None = None) -> dict[str, Any]:
    """Start ``cmd`` detached from this process's session so it outlives an agent disconnect."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = dict(env) if env is not None else dict(os.environ)
    if gpu is not None:
        full_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cuda_visible_devices = full_env.get("CUDA_VISIBLE_DEVICES", "")

    with open(log_path, "a", buffering=1) as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=logf,
            stdin=subprocess.DEVNULL,
            cwd=str(cwd) if cwd else None,
            env=full_env,
            start_new_session=True,
            shell=isinstance(cmd, str),
        )
    # The child inherited its own duplicated fd for logf during Popen(); the handle above
    # can close on context-manager exit without truncating or affecting the child's writes.

    return {
        "pid": proc.pid,
        "log": str(log_path),
        "cmd": cmd,
        "started": utcnow(),
        "cuda_visible_devices": cuda_visible_devices,
    }


def poll(pid: int) -> str:
    """``"running"`` or ``"exited"``. Never raises, even for an unknown or foreign pid."""
    try:
        # Reap if this is our own zombie child; a no-op (ChildProcessError) otherwise --
        # without this, kill(pid, 0) keeps reporting "running" for an unreaped zombie.
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "exited"
    except PermissionError:
        return "running"
    except OSError:
        return "exited"
    return "running"


def gpu_status() -> list[dict[str, Any]]:
    """Live per-GPU utilisation and free memory; empty when nvidia-smi is unavailable."""
    status = []
    for line in _query_nvidia_smi(
        "index,utilization.gpu,memory.used,memory.free,memory.total", nounits=True
    ):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            status.append({
                "index": int(parts[0]),
                "utilization_pct": float(parts[1]),
                "memory_used_mib": float(parts[2]),
                "memory_free_mib": float(parts[3]),
                "memory_total_mib": float(parts[4]),
            })
        except ValueError:
            continue
    return status
