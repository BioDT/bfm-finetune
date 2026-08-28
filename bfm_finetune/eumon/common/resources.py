"""Measure what a result cost: wall-clock, GPU utilisation, memory, energy, CO2e.

``Meter`` samples the device in a background thread and integrates power draw over the
run. Two honesty constraints are built in: NVML reports utilisation and power per GPU,
never per process, so on a shared card the figures are an upper bound and ``shared_gpu``
records that; and the carbon figure is a stated grid-intensity assumption, reported with
the factor attached.
"""

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# kg CO2e per kWh; a typical European grid.
GRID_INTENSITY_KG_PER_KWH = 0.3
SAMPLE_SECONDS = 1.0

# RAPL is root-only on most hosts, so CPU energy is an ESTIMATE from measured core-seconds
# at a stated watts-per-core (280 W TDP / 32 cores, AMD EPYC 9354), never a measurement.
CPU_WATTS_PER_CORE = 280.0 / 32.0


def _nvidia_smi(query: str, extra: list[str] | None = None) -> list[list[str]]:
    if not shutil.which("nvidia-smi"):
        return []
    cmd = ["nvidia-smi", f"--query-{'compute-apps' if 'pid' in query else 'gpu'}={query}",
           "--format=csv,noheader,nounits"] + (extra or [])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return []
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def gpu_inventory() -> list[dict[str, Any]]:
    rows = _nvidia_smi("index,name,memory.total,power.limit,driver_version")
    out = []
    for r in rows:
        try:
            out.append({"index": int(r[0]), "name": r[1], "memory_total_mib": float(r[2]),
                        "power_limit_w": float(r[3]), "driver": r[4]})
        except (ValueError, IndexError):
            continue
    return out


class GPUBusy(RuntimeError):
    pass


def gpu_occupants(index: int) -> list[dict[str, Any]]:
    """Compute processes on one GPU, excluding this one."""
    own = os.getpid()
    out = []
    for r in _nvidia_smi("pid,used_memory", ["-i", str(index)]):
        try:
            pid = int(r[0])
        except (ValueError, IndexError):
            continue
        if pid == own:
            continue
        out.append({"pid": pid, "used_mib": float(r[1]) if len(r) > 1 else None})
    return out


def require_exclusive_gpu(index: int, *, allow_shared: bool = False) -> dict[str, Any]:
    """Refuse to start on an occupied card unless explicitly told otherwise.

    Only the presence of another compute process counts; utilisation is deliberately not
    consulted, since it reads 0% between seeds while memory stays resident.
    """
    occupants = gpu_occupants(index)
    state = {"device": index, "occupants": occupants, "exclusive": not occupants,
             "allow_shared": allow_shared}
    if occupants and not allow_shared:
        detail = ", ".join(f"pid {o['pid']} ({o['used_mib']:.0f} MiB)" for o in occupants)
        raise GPUBusy(
            f"GPU {index} already has {len(occupants)} compute process(es): {detail}. "
            f"Energy and utilisation are per-device, so sharing the card corrupts the cost "
            f"measurement. Choose a free GPU, or pass --allow-shared to accept that this "
            f"run's resource figures are an upper bound.")
    return state


@dataclass
class Meter:
    """Sample a GPU while a block of work runs, and report what it cost.

    Usage::

        with Meter(device=1, label="L3_A_vera_s1") as m:
            ...
        record["resources"] = m.report()
    """

    device: int | None = None
    label: str = ""
    interval: float = SAMPLE_SECONDS
    grid_intensity: float = GRID_INTENSITY_KG_PER_KWH
    _samples: list[tuple[float, float, float, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _t0: float = 0.0
    _t1: float = 0.0
    idle_power_w: float | None = None
    _other_pids: set[int] = field(default_factory=set)
    _cpu0: float = 0.0

    def _sample_once(self) -> None:
        # A CPU-only job has no device; sampling every card would attribute strangers'
        # GPUs to it.
        if self.device is None:
            return
        for r in _nvidia_smi("index,utilization.gpu,memory.used,power.draw"):
            try:
                if int(r[0]) != self.device:
                    continue
                self._samples.append((time.time(), float(r[1]), float(r[2]), float(r[3])))
            except (ValueError, IndexError):
                continue

    def _poll(self) -> None:
        own = os.getpid()
        # Sample before the first wait: energy is integrated between consecutive samples,
        # so a run shorter than one interval would otherwise report 0 Wh.
        self._sample_once()
        while not self._stop.wait(self.interval):
            self._sample_once()
            if self.device is None:
                continue
            for r in _nvidia_smi("pid,used_memory", ["-i", str(self.device)]):
                try:
                    pid = int(r[0])
                    if pid != own:
                        self._other_pids.add(pid)
                except (ValueError, IndexError):
                    continue

    def _measure_idle(self, samples: int = 3) -> float | None:
        """Power drawn by this GPU before the workload starts.

        Total and above-idle are both reported — total for reproducibility of the
        measurement, above-idle as the marginal cost — and neither is presented as the
        other. Returns None for a CPU-only meter.
        """
        if self.device is None:
            return None
        rows = []
        for _ in range(samples):
            for r in _nvidia_smi("index,power.draw"):
                try:
                    if int(r[0]) == self.device:
                        rows.append(float(r[1]))
                except (ValueError, IndexError):
                    continue
            time.sleep(0.2)
        return round(sum(rows) / len(rows), 2) if rows else None

    @staticmethod
    def _cpu_seconds() -> float:
        """User+system CPU time of this process and its reaped children — children matter
        for baselines that fan out with joblib."""
        import resource

        total = 0.0
        for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN):
            r = resource.getrusage(who)
            total += r.ru_utime + r.ru_stime
        return total

    @staticmethod
    def energy_disabled() -> bool:
        """``EUMON_NO_ENERGY=1`` makes this a wall-clock-only stopwatch, for hosts where
        several shards share one GPU and energy would be attributable to nobody."""
        return os.environ.get("EUMON_NO_ENERGY", "").strip().lower() in {"1", "true", "yes"}

    def __enter__(self) -> "Meter":
        self._cpu0 = self._cpu_seconds()
        self._t0 = time.time()
        if self.energy_disabled():
            return self
        self.idle_power_w = self._measure_idle()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, name="eumon-meter", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._t1 = time.time()
        if self.energy_disabled():
            return
        self._stop.set()
        self._sample_once()   # guarantees a closing point for the trapezoid
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval + 1)

    def report(self) -> dict[str, Any]:
        wall = max(self._t1 - self._t0, 0.0) or (time.time() - self._t0)
        util = [s[1] for s in self._samples]
        mem = [s[2] for s in self._samples]
        power = [s[3] for s in self._samples]

        energy_wh = 0.0
        for (t0, _, _, p0), (t1, _, _, p1) in zip(self._samples, self._samples[1:]):
            energy_wh += (p0 + p1) / 2 * (t1 - t0) / 3600.0

        def stat(values: list[float], fn) -> float | None:
            return round(float(fn(values)), 2) if values else None

        peak_torch = None
        try:
            import torch

            if self.device is not None and torch.cuda.is_available():
                peak_torch = round(torch.cuda.max_memory_allocated(self.device) / 2 ** 30, 3)
        except Exception:
            pass

        return {
            "label": self.label, "device": self.device,
            "energy_measured": not self.energy_disabled(),
            "wall_s": round(wall, 2), "gpu_hours": round(wall / 3600.0, 5),
            "n_samples": len(self._samples), "sample_interval_s": self.interval,
            "gpu_util_mean_pct": stat(util, lambda v: sum(v) / len(v)),
            "gpu_util_peak_pct": stat(util, max),
            "gpu_mem_mean_mib": stat(mem, lambda v: sum(v) / len(v)),
            "gpu_mem_peak_mib": stat(mem, max),
            "torch_peak_alloc_gib": peak_torch,
            "power_mean_w": stat(power, lambda v: sum(v) / len(v)),
            "power_peak_w": stat(power, max),
            "energy_wh": round(energy_wh, 4),
            "energy_kwh": round(energy_wh / 1000.0, 6),
            "idle_power_w": self.idle_power_w,
            "energy_wh_above_idle": (None if self.idle_power_w is None else
                                     round(max(energy_wh - self.idle_power_w * wall / 3600.0,
                                               0.0), 4)),
            "co2e_kg": round(energy_wh / 1000.0 * self.grid_intensity, 6),
            "co2e_kg_above_idle": (None if self.idle_power_w is None else
                                   round(max(energy_wh - self.idle_power_w * wall / 3600.0, 0.0)
                                         / 1000.0 * self.grid_intensity, 6)),
            "cpu_core_seconds": round(max(self._cpu_seconds() - self._cpu0, 0.0), 2),
            "cpu_energy_wh_estimate": round(
                max(self._cpu_seconds() - self._cpu0, 0.0) / 3600.0 * CPU_WATTS_PER_CORE, 4),
            "cpu_watts_per_core_assumed": CPU_WATTS_PER_CORE,
            "cpu_energy_is_estimated": True,
            "grid_intensity_kg_per_kwh": self.grid_intensity,
            "shared_gpu": None if self.device is None else len(self._other_pids) > 0,
            "measured_gpu": self.device is not None,
            "other_pids_on_device": sorted(self._other_pids),
            "attribution_note": "utilisation and power are per GPU, not per process; when "
                                "shared_gpu is true these are an upper bound on this job's "
                                "share.",
        }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-run resource reports into the totals a manuscript table needs."""
    usable = [r for r in records if r and r.get("wall_s") is not None]
    if not usable:
        return {"n_runs": 0}
    total_h = sum(r["wall_s"] for r in usable) / 3600.0
    total_kwh = sum(r.get("energy_kwh") or 0.0 for r in usable)
    utils = [r["gpu_util_mean_pct"] for r in usable if r.get("gpu_util_mean_pct") is not None]
    shared = [r["label"] for r in usable if r.get("shared_gpu")]
    return {
        "n_runs": len(usable),
        "total_gpu_hours": round(total_h, 4),
        "total_energy_kwh": round(total_kwh, 5),
        "total_co2e_kg": round(total_kwh * GRID_INTENSITY_KG_PER_KWH, 5),
        "mean_gpu_util_pct": round(sum(utils) / len(utils), 1) if utils else None,
        "peak_gpu_mem_mib": max((r.get("gpu_mem_peak_mib") or 0) for r in usable),
        "median_run_wall_s": round(sorted(r["wall_s"] for r in usable)[len(usable) // 2], 1),
        "runs_on_a_shared_gpu": shared,
        "grid_intensity_kg_per_kwh": GRID_INTENSITY_KG_PER_KWH,
        "hardware": gpu_inventory(),
    }
