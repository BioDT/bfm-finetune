"""Panel schema, 0.25 degree gridding, and negative reconstruction.

The panel is the single canonical artefact per task; nothing downstream may touch a raw
archive. Its one load-bearing invariant, enforced here rather than trusted to callers:

    observed is False  <=>  value is NaN

A unit-year with no completed visit is masked, never a zero.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

VALUE_TYPES = {"count", "prevalence", "index"}

PANEL_COLUMNS: dict[str, str] = {
    "task": "string",
    "unit_id": "string",
    "lat": "float64",
    "lon": "float64",
    "cell_i": "int32",
    "cell_j": "int32",
    "year": "int32",
    "species": "string",
    "value": "float64",
    "value_type": "string",
    "effort": "float64",
    "observed": "bool",
}


@dataclass(frozen=True)
class GridSpec:
    """The BioAnalyst 0.25 degree grid, as the model actually receives it.

    Taken from a real BioCube batch's metadata: 161x281 points ascending from (32.0, -25.0),
    cropped to 160x280 by dropping the last row and column. Row 0 is the southernmost
    latitude. ``lat_origin``/``lon_origin`` are the coordinates of cell (0, 0); a point is
    assigned to its nearest grid point.
    """

    H: int = 160
    W: int = 280
    res: float = 0.25
    lat_origin: float = 32.0
    lon_origin: float = -25.0
    lat_ascending: bool = True
    origin_ref: Literal["centre", "edge"] = "centre"
    source: str = "biocube batch_metadata (161x281, cropped to 160x280)"

    @property
    def _lat_step(self) -> float:
        return self.res if self.lat_ascending else -self.res

    @property
    def lats(self) -> np.ndarray:
        return self.lat_origin + self._lat_step * np.arange(self.H, dtype=np.float64)

    @property
    def lons(self) -> np.ndarray:
        return self.lon_origin + self.res * np.arange(self.W, dtype=np.float64)

    def bounds(self) -> dict[str, float]:
        half = self.res / 2 if self.origin_ref == "centre" else 0.0
        lats, lons = self.lats, self.lons
        return {"lat_min": float(lats.min() - half), "lat_max": float(lats.max() + half),
                "lon_min": float(lons.min() - half), "lon_max": float(lons.max() + half)}

    def to_cell(self, lat: Any, lon: Any) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised (lat, lon) -> (cell_i, cell_j); -1 for points outside the domain."""
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        half = self.res / 2 if self.origin_ref == "centre" else 0.0
        i = np.floor((lat - (self.lat_origin - half)) / self._lat_step)
        j = np.floor((lon - (self.lon_origin - half)) / self.res)
        i = np.where(np.isfinite(lat), i, -1)
        j = np.where(np.isfinite(lon), j, -1)
        i = np.asarray(i, dtype=np.int64)
        j = np.asarray(j, dtype=np.int64)
        outside = (i < 0) | (i >= self.H) | (j < 0) | (j >= self.W)
        return (np.where(outside, -1, i).astype(np.int32),
                np.where(outside, -1, j).astype(np.int32))

    def cell_centre(self, i: Any, j: Any) -> tuple[np.ndarray, np.ndarray]:
        i = np.asarray(i, dtype=np.int64)
        j = np.asarray(j, dtype=np.int64)
        off = 0.0 if self.origin_ref == "centre" else self.res / 2
        return (self.lat_origin + self._lat_step * i + off * np.sign(self._lat_step),
                self.lon_origin + self.res * j + off)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), **self.bounds()}


GRID = GridSpec()


class PanelContractError(AssertionError):
    pass


def empty_panel() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in PANEL_COLUMNS.items()})


def coerce_panel(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PANEL_COLUMNS if c not in df.columns]
    if missing:
        raise PanelContractError(f"panel missing columns: {missing}")
    extra = [c for c in df.columns if c not in PANEL_COLUMNS]
    if extra:
        raise PanelContractError(f"panel has undeclared columns: {extra}")
    out = df.loc[:, list(PANEL_COLUMNS)].copy()
    for col, dtype in PANEL_COLUMNS.items():
        out[col] = out[col].astype(dtype)
    return out.reset_index(drop=True)


def validate_panel(df: pd.DataFrame, grid: GridSpec = GRID, *, allow_unobserved: bool = True) -> dict[str, Any]:
    """Assert the data contract. Returns summary statistics for the manifest."""
    if list(df.columns) != list(PANEL_COLUMNS):
        raise PanelContractError(f"column order/set differs: {list(df.columns)}")
    for col, dtype in PANEL_COLUMNS.items():
        if str(df[col].dtype) != dtype:
            raise PanelContractError(f"{col}: dtype {df[col].dtype} != {dtype}")
    if len(df) == 0:
        raise PanelContractError("panel is empty")

    tasks = set(df["task"].unique())
    if len(tasks) != 1 or not tasks <= {"A", "B", "C"}:
        raise PanelContractError(f"panel must cover exactly one task in A/B/C, got {tasks}")
    vtypes = set(df["value_type"].unique())
    if len(vtypes) != 1 or not vtypes <= VALUE_TYPES:
        raise PanelContractError(f"panel must carry exactly one known value_type, got {vtypes}")

    observed = df["observed"].to_numpy()
    value = df["value"].to_numpy()

    masked_with_value = int(np.count_nonzero(~observed & np.isfinite(value)))
    if masked_with_value:
        raise PanelContractError(
            f"{masked_with_value} rows have observed=False but a non-null value. "
            "'not surveyed' must never carry a number, least of all a zero.")
    observed_without_value = int(np.count_nonzero(observed & ~np.isfinite(value)))
    if observed_without_value:
        raise PanelContractError(f"{observed_without_value} rows have observed=True but a null value")
    negative = int(np.count_nonzero(observed & (value < 0)))
    if negative:
        raise PanelContractError(f"{negative} observed rows have a negative value")
    if not allow_unobserved and not observed.all():
        raise PanelContractError("unobserved rows present where none were expected")

    dup = df.duplicated(subset=["task", "unit_id", "species", "year"]).sum()
    if dup:
        raise PanelContractError(f"{dup} duplicate (unit_id, species, year) rows")

    if not np.isfinite(df["lat"].to_numpy()).all() or not np.isfinite(df["lon"].to_numpy()).all():
        raise PanelContractError("non-finite coordinates")
    off_grid = int(np.count_nonzero((df["cell_i"] < 0) | (df["cell_j"] < 0)))
    if off_grid:
        raise PanelContractError(f"{off_grid} rows fall outside the {grid.H}x{grid.W} model grid")
    ci, cj = grid.to_cell(df["lat"], df["lon"])
    if not (np.array_equal(ci, df["cell_i"].to_numpy()) and np.array_equal(cj, df["cell_j"].to_numpy())):
        raise PanelContractError("cell_i/cell_j do not match the grid mapping of lat/lon")

    eff = df["effort"].to_numpy()
    bad_effort = int(np.count_nonzero(observed & ~(eff > 0)))
    if bad_effort:
        raise PanelContractError(f"{bad_effort} observed rows have non-positive effort")

    obs = df.loc[df["observed"]]
    unit_years = df.loc[:, ["unit_id", "year"]].drop_duplicates()
    return {
        "rows": int(len(df)),
        "task": next(iter(tasks)),
        "value_type": next(iter(vtypes)),
        "n_units": int(df["unit_id"].nunique()),
        "n_species": int(df["species"].nunique()),
        "n_unit_years": int(len(unit_years)),
        "years": [int(y) for y in sorted(df["year"].unique())],
        "rows_observed": int(observed.sum()),
        "rows_masked": int((~observed).sum()),
        "n_cells": int(df.loc[:, ["cell_i", "cell_j"]].drop_duplicates().shape[0]),
        "value_zero_fraction": float((obs["value"] == 0).mean()) if len(obs) else float("nan"),
        "value_quantiles": {q: float(obs["value"].quantile(q)) for q in (0.5, 0.9, 0.99, 1.0)} if len(obs) else {},
        "grid": grid.as_dict(),
    }


def build_panel(visits: pd.DataFrame, records: pd.DataFrame, *, task: str, value_type: str,
                species: Iterable[str], grid: GridSpec = GRID) -> pd.DataFrame:
    """Cross a visit table with a species list, filling absences only where a visit completed.

    ``visits``  : one row per unit-year — unit_id, year, lat, lon, effort, completed (bool).
    ``records`` : the positives — unit_id, year, species, value.

    Every (unit-year, species) pair with ``completed`` and no record becomes a true zero;
    every pair without a completed visit becomes NaN with ``observed=False``.
    """
    species = pd.Index(pd.unique(pd.Series(list(species), dtype="string")), name="species")
    if species.empty:
        raise PanelContractError("species list is empty")

    required_v = {"unit_id", "year", "lat", "lon", "effort", "completed"}
    if not required_v <= set(visits.columns):
        raise PanelContractError(f"visits missing {required_v - set(visits.columns)}")
    required_r = {"unit_id", "year", "species", "value"}
    if not required_r <= set(records.columns):
        raise PanelContractError(f"records missing {required_r - set(records.columns)}")

    v = visits.drop_duplicates(subset=["unit_id", "year"]).copy()
    v["unit_id"] = v["unit_id"].astype("string")
    v["year"] = v["year"].astype("int32")

    grid_df = v.merge(pd.DataFrame({"species": species}), how="cross")

    r = records.copy()
    r["unit_id"] = r["unit_id"].astype("string")
    r["year"] = r["year"].astype("int32")
    r["species"] = r["species"].astype("string")
    r = r.loc[r["species"].isin(species)]
    if r.duplicated(subset=["unit_id", "year", "species"]).any():
        r = r.groupby(["unit_id", "year", "species"], as_index=False, observed=True)["value"].sum()

    out = grid_df.merge(r, on=["unit_id", "year", "species"], how="left")

    completed = out["completed"].to_numpy(dtype=bool)
    value = out["value"].to_numpy(dtype="float64")
    value = np.where(completed, np.nan_to_num(value, nan=0.0), np.nan)

    ci, cj = grid.to_cell(out["lat"], out["lon"])
    out = out.assign(task=task, value=value, value_type=value_type, observed=completed,
                     cell_i=ci, cell_j=cj)
    out = out.loc[(out["cell_i"] >= 0) & (out["cell_j"] >= 0)]
    return coerce_panel(out.loc[:, list(PANEL_COLUMNS)])


def aggregate_to_cell(panel: pd.DataFrame, grid: GridSpec = GRID) -> pd.DataFrame:
    """Collapse a unit-level panel onto the model grid, one row per (cell, species, year).

    Needed wherever several units share a cell: the model emits one value per cell while
    the nulls read each unit's own history, so scoring a cell-resolution prediction against
    unit-level truth is not like-for-like. ``effort`` becomes the number of contributing
    units and coordinates become the cell centre, so the panel keeps satisfying its own
    grid check.
    """
    obs = panel.loc[panel["observed"]]
    if obs.empty:
        raise PanelContractError("no observed rows to aggregate")

    keys = ["task", "value_type", "cell_i", "cell_j", "year", "species"]
    agg = obs.groupby(keys, observed=True).agg(value=("value", "mean"),
                                               effort=("unit_id", "nunique")).reset_index()
    lat, lon = grid.cell_centre(agg["cell_i"], agg["cell_j"])
    agg = agg.assign(lat=lat, lon=lon, observed=True,
                     unit_id=(agg["cell_i"].astype(str) + "_" + agg["cell_j"].astype(str)))
    return coerce_panel(agg.loc[:, list(PANEL_COLUMNS)])


def to_target(panel: pd.DataFrame, grid: GridSpec = GRID, *,
              agg: Literal["mean", "sum"] = "mean") -> dict[str, Any]:
    """Aggregate the unit-level panel onto the model grid.

    Returns ``y[species, year, H, W]``, ``mask[year, H, W]`` and the index arrays.
    ``mask`` is True only for cell-years containing at least one completed visit; ``y`` is
    NaN everywhere the mask is False.
    """
    obs = panel.loc[panel["observed"]]
    if obs.empty:
        raise PanelContractError("no observed rows to grid")

    species_index = sorted(panel["species"].unique().tolist())
    years = sorted(int(y) for y in panel["year"].unique())
    s_pos = {s: k for k, s in enumerate(species_index)}
    y_pos = {y: k for k, y in enumerate(years)}

    y_arr = np.full((len(species_index), len(years), grid.H, grid.W), np.nan, dtype=np.float32)
    n_arr = np.zeros((len(species_index), len(years), grid.H, grid.W), dtype=np.int32)
    mask = np.zeros((len(years), grid.H, grid.W), dtype=bool)

    grouped = obs.groupby(["species", "year", "cell_i", "cell_j"], observed=True)["value"].agg(["sum", "count"])
    for (sp, yr, i, j), row in grouped.iterrows():
        s, t = s_pos[sp], y_pos[int(yr)]
        n = int(row["count"])
        y_arr[s, t, i, j] = row["sum"] / n if agg == "mean" else row["sum"]
        n_arr[s, t, i, j] = n
        mask[t, i, j] = True

    effort = np.zeros((len(years), grid.H, grid.W), dtype=np.float32)
    eff = obs.drop_duplicates(subset=["unit_id", "year"]).groupby(["year", "cell_i", "cell_j"],
                                                                 observed=True)["effort"].sum()
    for (yr, i, j), val in eff.items():
        effort[y_pos[int(yr)], i, j] = val

    return {"y": y_arr, "mask": mask, "n_units": n_arr, "effort": effort,
            "species_index": species_index, "years": np.asarray(years, dtype=np.int32),
            "value_type": panel["value_type"].iloc[0], "task": panel["task"].iloc[0],
            "agg": agg, "grid": grid.as_dict()}


def write_panel(panel: pd.DataFrame, path: str | Path, grid: GridSpec = GRID) -> dict[str, Any]:
    from .common.runner import atomic_path

    stats = validate_panel(panel, grid)
    with atomic_path(path, suffix=".parquet") as tmp:
        panel.to_parquet(tmp, index=False, compression="zstd")
    return stats


def write_target(target: dict[str, Any], path: str | Path) -> dict[str, Any]:
    from .common.runner import atomic_path, write_json

    path = Path(path)
    with atomic_path(path, suffix=".npz") as tmp:
        np.savez_compressed(tmp, y=target["y"], mask=target["mask"], n_units=target["n_units"],
                            effort=target["effort"], years=target["years"])
    write_json(path.with_name(path.stem + "_species_index.json"),
               {"task": target["task"], "value_type": target["value_type"], "agg": target["agg"],
                "species_index": target["species_index"],
                "years": [int(y) for y in target["years"]], "grid": target["grid"]})
    return {"shape_y": list(target["y"].shape), "shape_mask": list(target["mask"].shape),
            "cells_observed_per_year": {int(y): int(target["mask"][k].sum())
                                        for k, y in enumerate(target["years"])}}
