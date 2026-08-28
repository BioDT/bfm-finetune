"""BioCube months -> model input, and model fields -> panel cells.

Each task's target accumulates over a fixed field season, so the forecast input is the
``(t-1, t)`` month pair that closes immediately before that season opens — never a pair
that overlaps it. BioCube's species channels are GBIF occurrence density and Tasks A and C
are published into GBIF, so a window-spanning input would hand the model a contemporaneous
observation of the very surveys it is predicting.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .common.runner import project_root
from .panel import GRID, GridSpec

_env_biocube = os.environ.get("EUMON_BIOCUBE")
BIOCUBE_DIR = Path(_env_biocube) if _env_biocube else project_root() / "data" / "batches_28species"
STATS_RELPATH = "bfm-model/batch_statistics/monthly_batches_stats_splitted_channels.json"
FILE_RE = re.compile(r"batch_(\d{4})-(\d{2})-01_to_(\d{4})-(\d{2})-01\.pt$")

ABIOTIC_GROUPS = ("surface_variables", "edaphic_variables", "atmospheric_variables",
                  "climate_variables", "vegetation_variables", "land_variables",
                  "agriculture_variables", "forest_variables", "misc_variables")


def stats_path() -> Path:
    """The scaling-statistics file shipped with ``bfm-model``."""
    return project_root() / STATS_RELPATH


@dataclass(frozen=True)
class SurveyWindow:
    task: str
    first_month: int
    last_month: int
    evidence: str

    @property
    def months(self) -> tuple[int, ...]:
        return tuple(range(self.first_month, self.last_month + 1))

    def input_pair_start(self) -> int:
        """First month of the ``(t-1, t)`` pair that closes before the window opens."""
        return self.first_month - 2


SURVEY_WINDOWS: dict[str, SurveyWindow] = {
    "A": SurveyWindow("A", 5, 7, "event.txt startDayOfYear spans 131-202 (11 May - 21 Jul)"),
    "B": SurveyWindow("B", 5, 9, "event.txt eventDate: 99.5% of 36,435 events in May-September"),
    "C": SurveyWindow("C", 4, 9, "UKBMS protocol: transects walked from the beginning of April "
                                 "to the end of September"),
}


class WindowUnavailable(FileNotFoundError):
    pass


def batch_path(year: int, month: int, biocube_dir: str | Path = BIOCUBE_DIR) -> Path:
    """Path of the file holding the ``(month, month+1)`` pair of ``year``."""
    nxt_y, nxt_m = (year + 1, 1) if month == 12 else (year, month + 1)
    return Path(biocube_dir) / f"batch_{year:04d}-{month:02d}-01_to_{nxt_y:04d}-{nxt_m:02d}-01.pt"


def available_months(biocube_dir: str | Path = BIOCUBE_DIR) -> list[tuple[int, int]]:
    out = []
    for p in sorted(Path(biocube_dir).glob("batch_*.pt")):
        m = FILE_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


def forecast_input(task: str, target_year: int, biocube_dir: str | Path = BIOCUBE_DIR) -> dict[str, Any]:
    """The short-lead input for ``target_year``: the pair closing before the window opens.

    Raises rather than falling back to a later pair, which would overlap the survey window.
    """
    window = SURVEY_WINDOWS[task]
    start = window.input_pair_start()
    if start < 1:
        raise ValueError(f"task {task} window opens in month {window.first_month}; the "
                         "preceding pair falls in the previous year and is not handled")
    path = batch_path(target_year, start, biocube_dir)
    if not path.exists():
        raise WindowUnavailable(
            f"task {task} target {target_year} needs {path.name} — the (t-1, t) pair closing "
            f"before month {window.first_month}; BioCube does not provide it")
    return {
        "task": task, "target_year": target_year, "path": str(path),
        "input_months": [f"{target_year}-{start:02d}", f"{target_year}-{start + 1:02d}"],
        "predicts_month": f"{target_year}-{window.first_month:02d}",
        "window_months": list(window.months),
        "lead_months": 1,
        "note": "single-step short-lead; the decoded field is the first month of the survey "
                "window. Reaching the rest of the window needs roll-out, whose fine-tuned "
                "weights are unavailable.",
    }


def forecast_plan(tasks: Iterable[str] = ("A", "B", "C"),
                  target_years: Iterable[int] = (2019, 2020),
                  biocube_dir: str | Path = BIOCUBE_DIR) -> dict[str, Any]:
    """Resolve every (task, year) input up front, so a missing month fails before GPU time."""
    ok, missing = [], []
    for task in tasks:
        for year in target_years:
            try:
                ok.append(forecast_input(task, year, biocube_dir))
            except (WindowUnavailable, ValueError) as exc:
                missing.append({"task": task, "target_year": year, "error": str(exc)})
    return {"resolved": ok, "missing": missing,
            "biocube_months": len(available_months(biocube_dir))}


def make_dataset(cfg, biocube_dir: str | Path = BIOCUBE_DIR):
    """A ``LargeClimateDataset`` wired to the local batches and local scaling statistics.

    Scaling stays enabled: the 28-species batches carry exactly the keys the statistics
    file defines, so ``scale_batch`` resolves every channel.
    """
    from bfm_model.bfm.dataloader_monthly import LargeClimateDataset

    cfg.data.scaling.stats_path = str(stats_path())
    return LargeClimateDataset(
        data_dir=str(biocube_dir), scaling_settings=cfg.data.scaling,
        num_species=cfg.data.species_number, atmos_levels=cfg.data.atmos_levels,
        model_patch_size=cfg.model.patch_size)


def load_input(path: str | Path, dataset) -> Any:
    return dataset.load_and_process_files(str(path))


def collate_for_model(sample: Any, batch_size: int = 1) -> Any:
    """Batch one sample into the form the encoder actually reads.

    Repairs two metadata-layout disagreements between ``custom_collate`` and the encoder,
    leaving ``bfm-model`` untouched: latitude/longitude must be batched to ``[B, H]``, and
    ``timestamp`` must be a list of per-sample lists.
    """
    import torch
    from bfm_model.bfm.dataloader_monthly import custom_collate

    collated = custom_collate([sample] * batch_size)
    meta = collated.batch_metadata

    lat, lon = meta.latitudes, meta.longitudes
    if getattr(lat, "ndim", 1) == 1:
        lat = torch.as_tensor(lat).unsqueeze(0).repeat(batch_size, 1)
        lon = torch.as_tensor(lon).unsqueeze(0).repeat(batch_size, 1)

    stamps = meta.timestamp
    if stamps and isinstance(stamps[0], str):
        stamps = [list(stamps) for _ in range(batch_size)]

    return collated._replace(batch_metadata=meta._replace(latitudes=lat, longitudes=lon,
                                                          timestamp=stamps))


def sanitise(sample: Any) -> tuple[Any, dict[str, int]]:
    """Replace non-finite inputs with zeros, reporting what was replaced.

    BioCube carries a small number of NaNs outside the vegetation group (the only group
    ``crop_variables`` NaN-handles); they are zeroed rather than masked because the affected
    fraction is negligible, and the count is returned so it is never silent.
    """
    import torch

    replaced: dict[str, int] = {}
    fields = {}
    for name, value in zip(sample._fields, sample):
        if isinstance(value, dict) and name != "batch_metadata":
            clean = {}
            for key, tensor in value.items():
                if hasattr(tensor, "isfinite"):
                    bad = int((~torch.isfinite(tensor)).sum())
                    if bad:
                        replaced[f"{name}.{key}"] = bad
                        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
                clean[key] = tensor
            fields[name] = clean
        else:
            fields[name] = value
    return type(sample)(**fields), replaced


def _as_array(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def read_era5_month(year: int, month: int, group: str, name: str,
                    biocube_dir: str | Path = BIOCUBE_DIR,
                    grid: GridSpec = GRID) -> np.ndarray:
    """One month of a raw BioCube channel, in physical units, on the model grid.

    Read straight off disk rather than through the dataset, which would apply the training
    scaling. The file holding the ``(month, month+1)`` pair carries ``month`` at timestep 0,
    and the model sees ``[..., :H, :W]`` of the 161x281 array, so the same crop applies here.
    """
    import torch

    path = batch_path(year, month, biocube_dir)
    if not path.exists():
        raise WindowUnavailable(f"BioCube has no {path.name}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    stamp = blob["batch_metadata"]["timestamp"][0][:7]
    if stamp != f"{year:04d}-{month:02d}":
        raise ValueError(f"{path.name} carries {stamp} at timestep 0, expected "
                         f"{year:04d}-{month:02d}; the pair convention has changed")
    return _as_array(blob[group][name][0, :grid.H, :grid.W]).astype(np.float32)


def gather_cells(field: Any, cell_i: Sequence[int], cell_j: Sequence[int],
                 grid: GridSpec = GRID) -> np.ndarray:
    """Sample a ``[..., H, W]`` field at panel cells, preserving leading dimensions."""
    arr = _as_array(field)
    if arr.shape[-2:] != (grid.H, grid.W):
        raise ValueError(f"field ends in {arr.shape[-2:]}, expected ({grid.H}, {grid.W})")
    return arr[..., np.asarray(cell_i, dtype=int), np.asarray(cell_j, dtype=int)]


def cell_covariates(batch: Any, cell_i: Sequence[int], cell_j: Sequence[int],
                    groups: Sequence[str] = ABIOTIC_GROUPS, timestep: int = -1,
                    grid: GridSpec = GRID) -> tuple[np.ndarray, list[str]]:
    """Per-cell abiotic covariate matrix, for the classical baselines.

    Species channels are excluded: they are the model's own effort-biased input, and mixing
    them into an environmental baseline would make it a partial passthrough of the target.
    """
    rows, names = [], []
    for group in groups:
        variables = getattr(batch, group, None) if not isinstance(batch, dict) else batch.get(group)
        if not variables:
            continue
        for name, value in sorted(variables.items()):
            arr = _as_array(value)
            if arr.ndim >= 3 and arr.shape[-2:] == (grid.H, grid.W):
                flat = arr.reshape(-1, grid.H, grid.W)[timestep]
                rows.append(flat[np.asarray(cell_i, dtype=int), np.asarray(cell_j, dtype=int)])
                names.append(f"{group}.{name}")
    if not rows:
        raise ValueError("no gridded covariates found on this batch")
    return np.stack(rows, axis=1), names


def describe(batch: Any) -> dict[str, Any]:
    """Group -> variable names and tensor shapes, for provenance."""
    out: dict[str, Any] = {}
    items = batch.items() if isinstance(batch, dict) else zip(batch._fields, batch)
    for group, value in items:
        if isinstance(value, dict) and value:
            first = next(iter(value.values()))
            shape = tuple(_as_array(first).shape) if hasattr(first, "shape") else None
            out[group] = {"n_variables": len(value), "variables": sorted(map(str, value)),
                          "shape": shape}
    return out
