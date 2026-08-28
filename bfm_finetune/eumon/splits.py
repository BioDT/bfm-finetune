"""Spatial block holdout and chronological head splits.

A random split of units is meaningless on a spatially autocorrelated panel (Roberts et al.
2017): a held-out unit almost always has a training unit nearby, so the head interpolates
rather than generalises. Units are grouped into contiguous lat/lon blocks, whole blocks are
assigned to folds, and an optional buffer drops training units within a given distance of
any test unit. ``verify`` reports the realised minimum train-to-test distance rather than
assuming the buffer worked.
"""

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(np.asarray(lat1, float))[:, None], np.radians(np.asarray(lat2, float))[None, :]
    dlam = np.radians(np.asarray(lon2, float))[None, :] - np.radians(np.asarray(lon1, float))[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@dataclass(frozen=True)
class BlockSpec:
    block_deg: float = 2.0
    n_folds: int = 5
    buffer_km: float = 0.0
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# Task C uses finer blocks because UKBMS sites cluster: at 2 degrees one block holds 28.5%
# of all sites and folds cannot balance; at 1 degree the largest block is 14.3%.
TASK_BLOCKS: dict[str, BlockSpec] = {
    "A": BlockSpec(block_deg=2.0, n_folds=5, buffer_km=50.0),
    "B": BlockSpec(block_deg=2.0, n_folds=5, buffer_km=50.0),
    "C": BlockSpec(block_deg=1.0, n_folds=5, buffer_km=50.0),
}

SEPARATION_EVIDENCE = {
    "note": "median distance from a test unit to its nearest training unit, the quantity "
            "blocking is meant to raise",
    "A": {"random": 24.9, "blocked": 49.8, "blocked_buffered": 89.8},
    "B": {"random": 12.8, "blocked": 30.9, "blocked_buffered": 75.7},
    "C": {"random": 1.5, "blocked": 17.9, "blocked_buffered": 70.0},
}


def spec_for(task: str) -> BlockSpec:
    return TASK_BLOCKS.get(str(task), BlockSpec())


def units_of(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per unit with its coordinates, from observed rows only."""
    obs = panel.loc[panel["observed"], ["unit_id", "lat", "lon"]]
    return obs.drop_duplicates("unit_id").reset_index(drop=True)


def assign_blocks(units: pd.DataFrame, block_deg: float) -> pd.Series:
    """Group units into contiguous lat/lon blocks of ``block_deg`` degrees."""
    bi = np.floor(units["lat"].to_numpy(float) / block_deg).astype(int)
    bj = np.floor(units["lon"].to_numpy(float) / block_deg).astype(int)
    return pd.Series([f"{a}_{b}" for a, b in zip(bi, bj)], index=units.index, name="block")


def spatial_block_folds(panel: pd.DataFrame, spec: BlockSpec = BlockSpec()) -> dict[str, Any]:
    """Assign whole blocks to folds, placing blocks largest-first into the smallest fold."""
    units = units_of(panel)
    if units.empty:
        raise ValueError("panel has no observed units")
    units["block"] = assign_blocks(units, spec.block_deg)

    sizes = units.groupby("block", observed=True).size().sort_values(ascending=False)
    rng = np.random.default_rng(spec.seed)
    order = list(sizes.index)
    rng.shuffle(order)
    order.sort(key=lambda b: -int(sizes[b]))

    load = np.zeros(spec.n_folds, dtype=int)
    block_fold: dict[str, int] = {}
    for block in order:
        k = int(np.argmin(load))
        block_fold[block] = k
        load[k] += int(sizes[block])

    units["fold"] = units["block"].map(block_fold).astype(int)
    return {"units": units, "spec": spec.as_dict(),
            "n_blocks": int(units["block"].nunique()),
            "fold_sizes": units.groupby("fold", observed=True).size().to_dict()}


def fold_masks(assignment: dict[str, Any], fold: int,
               buffer_km: float | None = None) -> dict[str, Any]:
    """Train/test unit ids for one fold, after applying the buffer."""
    units = assignment["units"]
    buffer_km = assignment["spec"]["buffer_km"] if buffer_km is None else buffer_km

    test = units.loc[units["fold"] == fold]
    train = units.loc[units["fold"] != fold]
    dropped = 0
    if buffer_km > 0 and len(test) and len(train):
        d = haversine_km(train["lat"].to_numpy(), train["lon"].to_numpy(),
                         test["lat"].to_numpy(), test["lon"].to_numpy())
        keep = d.min(axis=1) > buffer_km
        dropped = int((~keep).sum())
        train = train.loc[keep]

    return {"fold": fold,
            "train_units": train["unit_id"].tolist(), "test_units": test["unit_id"].tolist(),
            "n_train": int(len(train)), "n_test": int(len(test)),
            "buffer_km": float(buffer_km), "train_units_dropped_to_buffer": dropped}


def verify(assignment: dict[str, Any], fold: int, buffer_km: float | None = None) -> dict[str, Any]:
    """Assert disjointness and report the realised minimum train-to-test distance."""
    units = assignment["units"].set_index("unit_id")
    masks = fold_masks(assignment, fold, buffer_km)
    train, test = set(masks["train_units"]), set(masks["test_units"])
    overlap = train & test
    if overlap:
        raise AssertionError(f"{len(overlap)} units appear in both train and test")

    min_km, median_nearest, q25_nearest = float("nan"), float("nan"), float("nan")
    if train and test:
        tr, te = units.loc[sorted(train)], units.loc[sorted(test)]
        d = haversine_km(tr["lat"].to_numpy(), tr["lon"].to_numpy(),
                         te["lat"].to_numpy(), te["lon"].to_numpy())
        nearest = d.min(axis=0)
        min_km = float(nearest.min())
        median_nearest = float(np.median(nearest))
        q25_nearest = float(np.quantile(nearest, 0.25))

    return {**{k: v for k, v in masks.items() if not k.endswith("_units")},
            "min_train_to_test_km": min_km,
            "median_nearest_train_km": median_nearest,
            "q25_nearest_train_km": q25_nearest,
            "buffer_respected": bool(not np.isfinite(min_km) or min_km > masks["buffer_km"]),
            "n_units_total": int(len(units))}


def apply_fold(panel: pd.DataFrame, assignment: dict[str, Any], fold: int,
               buffer_km: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    masks = fold_masks(assignment, fold, buffer_km)
    obs = panel.loc[panel["observed"]]
    return (obs.loc[obs["unit_id"].isin(masks["train_units"])],
            obs.loc[obs["unit_id"].isin(masks["test_units"])])


def chronological_split(panel: pd.DataFrame, *, train_end: int, val_years: Sequence[int] = (),
                        test_years: Sequence[int] = ()) -> dict[str, Any]:
    """Head-level split in time. Validation must never be drawn from the test years."""
    val, test = set(int(y) for y in val_years), set(int(y) for y in test_years)
    if val & test:
        raise ValueError(f"validation and test years overlap: {sorted(val & test)}")
    if any(y <= train_end for y in test):
        raise ValueError(f"test years must follow train_end={train_end}")

    years = sorted(int(y) for y in panel["year"].unique())
    train = [y for y in years if y <= train_end and y not in val and y not in test]
    obs = panel.loc[panel["observed"]]
    counts = {name: int(obs.loc[obs["year"].isin(group)].shape[0])
              for name, group in (("train", train), ("val", sorted(val)), ("test", sorted(test)))}
    return {"train_years": train, "val_years": sorted(val), "test_years": sorted(test),
            "rows": counts}


def summarize(panel: pd.DataFrame, spec: BlockSpec = BlockSpec()) -> dict[str, Any]:
    """Everything a manifest needs about a spatial split, including realised distances."""
    assignment = spatial_block_folds(panel, spec)
    folds = [verify(assignment, k) for k in range(spec.n_folds)]
    return {"task": str(panel["task"].iloc[0]), "spec": spec.as_dict(),
            "n_blocks": assignment["n_blocks"], "fold_sizes": assignment["fold_sizes"],
            "folds": folds,
            "min_train_to_test_km_across_folds": float(
                np.nanmin([f["min_train_to_test_km"] for f in folds])),
            "median_nearest_train_km_across_folds": float(
                np.nanmedian([f["median_nearest_train_km"] for f in folds])),
            "largest_block_fraction": float(largest_block_fraction(assignment))}


def largest_block_fraction(assignment: dict[str, Any]) -> float:
    """Share of units in the single biggest block; a large value means reduce ``block_deg``."""
    sizes = assignment["units"].groupby("block", observed=True).size()
    return float(sizes.max() / sizes.sum())
