"""Step state machine: idempotent skip, atomic write, heartbeat, resume.

Progress lives on disk, never in a process: any session may die at any moment and a
fresh one must resume from ``runs/<run_id>/state.json`` alone.
"""

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

STALE_HEARTBEAT_S = 15 * 60
HEARTBEAT_PERIOD_S = 60
MAX_STEP_FAILURES = 3


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def project_root() -> Path:
    """Root under which data, weights, artefacts and bfm-model resolve.

    Defaults to this repository's checkout; ``EUMON_ROOT`` points it elsewhere.
    """
    env = os.environ.get("EUMON_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


def artefacts_root() -> Path:
    """Output root for every artefact. ``EUMON_ARTEFACTS`` redirects a whole run."""
    env = os.environ.get("EUMON_ARTEFACTS", "artefacts")
    path = Path(env)
    return path if path.is_absolute() else project_root() / path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@contextmanager
def atomic_path(path: str | Path, suffix: str = ""):
    """Yield a temporary path in the destination directory; publish it on clean exit.

    A killed process must never leave a half-written artefact that looks complete.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=suffix or ".tmp")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        yield tmp_path
        if not tmp_path.exists():
            raise FileNotFoundError(f"step did not write {tmp_path}")
        with open(tmp_path, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_json(path: str | Path, obj: Any) -> str:
    payload = json.dumps(obj, indent=2, sort_keys=False, default=str).encode()
    with atomic_path(path) as tmp:
        tmp.write_bytes(payload)
    return sha256_bytes(payload)


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return default


def _git_sha(repo: Path) -> dict[str, Any] | None:
    if not (repo / ".git").exists():
        return None
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        return {"sha": sha, "dirty": bool(dirty)}
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None


def code_provenance() -> dict[str, Any]:
    """Identify the code that produced an artefact: package digest plus repo SHAs."""
    pkg = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for src in sorted(pkg.rglob("*.py")):
        h.update(str(src.relative_to(pkg)).encode())
        h.update(src.read_bytes())
    repos = {}
    for repo in (pkg.parents[1], project_root() / "bfm-model"):
        rec = _git_sha(repo)
        if rec:
            repos[repo.name] = rec
    return {
        "code_digest": h.hexdigest(),
        "package": str(pkg),
        "repos": repos,
        "python": sys.version.split()[0],
        "host": socket.gethostname(),
    }


class StepFailed(RuntimeError):
    pass


class RunHalted(RuntimeError):
    pass


class Runner:
    """Owns ``runs/<run_id>/state.json`` and the idempotent-step contract."""

    def __init__(self, run_id: str | None = None, phase: str = "unset", root: Path | None = None,
                 resume: bool = True):
        self.root = Path(root) if root else project_root()
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        if run_id is None and resume:
            run_id = self._latest_run_id()
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        self.dir = self.runs_dir / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._hb_stop = threading.Event()
        self._hb_thread: threading.Thread | None = None

        state = read_json(self.state_path)
        if state is None:
            state = {"run_id": self.run_id, "phase": phase, "machine": socket.gethostname(),
                     "created": utcnow(), "heartbeat": utcnow(), "code": code_provenance(),
                     "steps": {}}
        else:
            state["phase"] = phase if phase != "unset" else state.get("phase", "unset")
            state["code"] = code_provenance()
        self._state = state
        self._reap_stale()
        self._save()

    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def _latest_run_id(self) -> str | None:
        candidates = sorted(p.name for p in self.runs_dir.iterdir()
                            if p.is_dir() and (p / "state.json").exists())
        return candidates[-1] if candidates else None

    def _save(self) -> None:
        """Merge this process's steps into the on-disk state under an exclusive lock.

        Concurrent workers share one run directory; a plain overwrite would lose the
        other process's step records.
        """
        import fcntl

        with self._lock:
            self._state["heartbeat"] = utcnow()
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / ".state.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    disk = read_json(self.state_path) or {}
                    merged = {**disk, **{k: v for k, v in self._state.items() if k != "steps"}}
                    steps = dict(disk.get("steps", {}))
                    steps.update(self._state.get("steps", {}))
                    merged["steps"] = steps
                    self._state["steps"] = steps
                    write_json(self.state_path, merged)
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    def _reap_stale(self) -> None:
        hb = self._state.get("heartbeat")
        stale = True
        if hb:
            try:
                age = (datetime.now(timezone.utc)
                       - datetime.strptime(hb, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc))
                stale = age.total_seconds() > STALE_HEARTBEAT_S
            except ValueError:
                stale = True
        for rec in self._state.get("steps", {}).values():
            if rec.get("status") == "running" and stale:
                rec["status"] = "pending"
                rec["failures"] = rec.get("failures", 0) + 1
                rec["last_error"] = "stale heartbeat; presumed dead"

    def _start_heartbeat(self) -> None:
        if self._hb_thread is not None:
            return

        def beat() -> None:
            while not self._hb_stop.wait(HEARTBEAT_PERIOD_S):
                self._save()

        self._hb_thread = threading.Thread(target=beat, name="eumon-heartbeat", daemon=True)
        self._hb_thread.start()

    def close(self) -> None:
        self._hb_stop.set()
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=2)
            self._hb_thread = None
        self._save()

    def record(self, name: str) -> dict[str, Any]:
        return self._state.setdefault("steps", {}).setdefault(name, {"status": "pending"})

    def _outputs_valid(self, rec: dict[str, Any], outputs: Sequence[Path]) -> bool:
        recorded = rec.get("outputs") or {}
        for out in outputs:
            if not out.exists():
                return False
            want = recorded.get(str(out))
            if want and want != sha256_file(out):
                return False
        return bool(outputs) and all(str(o) in recorded for o in outputs)

    def run_step(self, name: str, fn: Callable[[], Any], outputs: Iterable[str | Path] = (),
                 inputs: Iterable[str | Path] = (), force: bool = False,
                 meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute ``fn`` unless its declared outputs already exist and match the manifest."""
        outs = [Path(o) for o in outputs]
        ins = [Path(i) for i in inputs]
        rec = self.record(name)

        if not force and rec.get("status") == "done" and self._outputs_valid(rec, outs):
            rec["skipped_at"] = utcnow()
            self._save()
            return rec

        if rec.get("failures", 0) >= MAX_STEP_FAILURES:
            raise RunHalted(
                f"step {name!r} failed {rec['failures']} times; last error: {rec.get('last_error')}")

        missing = [str(i) for i in ins if not i.exists()]
        if missing:
            rec.update(status="blocked", last_error=f"missing inputs: {missing}")
            self._save()
            raise StepFailed(f"step {name!r} missing inputs: {missing}")

        rec.update(status="running", started=utcnow(), last_error=None,
                   inputs={str(i): sha256_file(i) for i in ins if i.is_file()})
        if meta:
            rec["meta"] = meta
        self._save()
        self._start_heartbeat()

        t0 = time.time()
        try:
            result = fn()
        except BaseException as exc:
            rec.update(status="failed", failures=rec.get("failures", 0) + 1,
                       last_error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc()[-4000:], ended=utcnow())
            self._save()
            raise

        absent = [str(o) for o in outs if not o.exists()]
        if absent:
            rec.update(status="failed", failures=rec.get("failures", 0) + 1,
                       last_error=f"declared outputs not written: {absent}", ended=utcnow())
            self._save()
            raise StepFailed(f"step {name!r} did not write {absent}")

        rec.update(status="done", ended=utcnow(), wall_s=round(time.time() - t0, 3),
                   failures=0, outputs={str(o): sha256_file(o) for o in outs})
        if isinstance(result, dict):
            rec["result"] = result
        self._save()
        self._append_manifest(name, rec)
        return rec

    def _append_manifest(self, name: str, rec: dict[str, Any]) -> None:
        import fcntl

        with open(self.dir / ".state.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self._append_manifest_locked(name, rec)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _append_manifest_locked(self, name: str, rec: dict[str, Any]) -> None:
        manifest = read_json(self.manifest_path, default={"run_id": self.run_id, "entries": {}})
        manifest["code"] = self._state["code"]
        manifest["machine"] = self._state["machine"]
        manifest["updated"] = utcnow()
        manifest["entries"][name] = {k: rec.get(k) for k in
                                     ("status", "started", "ended", "wall_s", "inputs", "outputs",
                                      "meta", "result")}
        write_json(self.manifest_path, manifest)

    def summary(self) -> str:
        lines = [f"run {self.run_id}  phase={self._state.get('phase')}  "
                 f"host={self._state.get('machine')}  heartbeat={self._state.get('heartbeat')}"]
        for name, rec in self._state.get("steps", {}).items():
            extra = ""
            if rec.get("status") in {"failed", "blocked"}:
                extra = f"  <- {rec.get('last_error')}"
            lines.append(f"  {rec.get('status', '?'):8s} {name}{extra}")
        return "\n".join(lines)


def artefact_provenance(path: str | Path, sources: list[dict[str, Any]],
                        extra: dict[str, Any] | None = None) -> Path:
    """Write ``<artefact>.prov.json``: what produced it, from what, when."""
    path = Path(path)
    prov = {"artefact": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "written": utcnow(), "sources": sources, "code": code_provenance()}
    if extra:
        prov.update(extra)
    out = path.with_suffix(path.suffix + ".prov.json")
    write_json(out, prov)
    return out


if __name__ == "__main__":
    print(Runner(run_id=sys.argv[1] if len(sys.argv) > 1 else None).summary())
