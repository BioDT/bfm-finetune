"""The seven reference forecasts ("nulls") every predictor is scored against.

Every null predicts the same rows a model is scored on: the observed rows of the test
year. Two of them deliberately read data a forecaster would not have — ``spatial_neighbour``
interpolates other units in the test year (Bahn & McGill 2007) and ``distribution_matched``
samples each species' training distribution (Raes & ter Steege 2007) — so they are
diagnostics, not candidate references, and are stated as such wherever reported.
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

NULL_NAMES = ("persistence", "lagged_persistence", "climatology", "species_mean", "site_mean",
              "spatial_neighbour", "distribution_matched")

REFERENCES = ("persistence", "lagged_persistence", "climatology")

EARTH_R_KM = 6371.0


@dataclass(frozen=True)
class Split:
    """Chronological split. ``test_year`` is predicted from ``train_years``."""

    train_years: tuple[int, ...]
    test_year: int

    @property
    def previous_year(self) -> int:
        return self.test_year - 1

    def as_dict(self) -> dict[str, Any]:
        return {"train_years": list(self.train_years), "test_year": self.test_year,
                "previous_year": self.previous_year}


def observed(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.loc[panel["observed"]]


def test_frame(panel: pd.DataFrame, split: Split) -> pd.DataFrame:
    """The rows every predictor is scored on: observed rows of the test year."""
    t = observed(panel).loc[panel["year"] == split.test_year]
    return t.loc[:, ["unit_id", "species", "year", "cell_i", "cell_j", "lat", "lon", "value"]] \
            .rename(columns={"value": "y_true"}).reset_index(drop=True)


def _train(panel: pd.DataFrame, split: Split) -> pd.DataFrame:
    return observed(panel).loc[panel["year"].isin(split.train_years)]


def persistence(panel: pd.DataFrame, split: Split, test: pd.DataFrame) -> np.ndarray:
    """Index at t+1 equals index at t. The dominant reference forecast."""
    prev = observed(panel).loc[panel["year"] == split.previous_year,
                               ["unit_id", "species", "value"]].rename(columns={"value": "p"})
    return test.merge(prev, on=["unit_id", "species"], how="left")["p"].to_numpy(float)


def _last_observation(panel: pd.DataFrame, split: Split) -> pd.DataFrame:
    # Bounded to the training window, like every other null and baseline.
    prior = observed(panel).loc[panel["year"].isin(split.train_years),
                                ["unit_id", "species", "year", "value"]]
    prior = prior.sort_values("year").drop_duplicates(["unit_id", "species"], keep="last")
    return prior.rename(columns={"value": "p", "year": "last_year"})


def lagged_persistence(panel: pd.DataFrame, split: Split, test: pd.DataFrame) -> np.ndarray:
    """The value at this unit the last time it was surveyed, however long ago.

    Classic persistence is undefined wherever a unit was not visited in the previous year;
    on a rotating-panel design that is most of the panel. This is the reference a
    practitioner actually holds, and the lag distribution is reported alongside it.
    """
    prior = _last_observation(panel, split)
    return test.merge(prior, on=["unit_id", "species"], how="left")["p"].to_numpy(float)


def lag_profile(panel: pd.DataFrame, split: Split, test: pd.DataFrame) -> dict[str, Any]:
    prior = _last_observation(panel, split)
    merged = test.merge(prior, on=["unit_id", "species"], how="left")
    lag = (split.test_year - merged["last_year"]).dropna().astype(int)
    if lag.empty:
        return {"n": 0}
    return {"n": int(len(lag)), "coverage": float(len(lag) / len(test)),
            "median_lag_years": float(lag.median()), "mean_lag_years": round(float(lag.mean()), 3),
            "lag_histogram": {int(k): int(v) for k, v in sorted(lag.value_counts().items())[:12]},
            "fraction_lag_1": float((lag == 1).mean())}


def climatology(panel: pd.DataFrame, split: Split, test: pd.DataFrame) -> np.ndarray:
    """Per-species, per-unit mean over the training years."""
    m = _train(panel, split).groupby(["unit_id", "species"], observed=True)["value"].mean() \
        .rename("p").reset_index()
    return test.merge(m, on=["unit_id", "species"], how="left")["p"].to_numpy(float)


def species_mean(panel: pd.DataFrame, split: Split, test: pd.DataFrame) -> np.ndarray:
    m = _train(panel, split).groupby("species", observed=True)["value"].mean().rename("p").reset_index()
    return test.merge(m, on="species", how="left")["p"].to_numpy(float)


def site_mean(panel: pd.DataFrame, split: Split, test: pd.DataFrame) -> np.ndarray:
    m = _train(panel, split).groupby("unit_id", observed=True)["value"].mean().rename("p").reset_index()
    return test.merge(m, on="unit_id", how="left")["p"].to_numpy(float)


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat1)[:, None], np.radians(lat2)[None, :]
    dphi = p2 - p1
    dlam = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def spatial_neighbour(panel: pd.DataFrame, split: Split, test: pd.DataFrame,
                      k: int = 8, power: float = 2.0) -> np.ndarray:
    """Inverse-distance interpolation from the k nearest *other* units in the test year.

    Leave-one-out by construction; ignores environment entirely.
    """
    out = np.full(len(test), np.nan, dtype=float)
    units = test.drop_duplicates("unit_id").set_index("unit_id")[["lat", "lon"]]
    for sp, g in test.groupby("species", observed=True):
        sub = g.drop_duplicates("unit_id")
        if len(sub) < 2:
            continue
        ids = sub["unit_id"].to_numpy()
        lat = units.loc[ids, "lat"].to_numpy(float)
        lon = units.loc[ids, "lon"].to_numpy(float)
        vals = sub.set_index("unit_id")["y_true"].loc[ids].to_numpy(float)

        d = _haversine_km(lat, lon, lat, lon)
        np.fill_diagonal(d, np.inf)
        kk = min(k, len(ids) - 1)
        nn = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        dn = np.take_along_axis(d, nn, axis=1)
        w = 1.0 / np.power(np.clip(dn, 1e-6, None), power)
        pred = (w * vals[nn]).sum(axis=1) / w.sum(axis=1)

        by_unit = pd.Series(pred, index=ids)
        sel = test["species"].to_numpy() == sp
        out[sel] = by_unit.reindex(test.loc[sel, "unit_id"]).to_numpy(float)
    return out


def distribution_matched(panel: pd.DataFrame, split: Split, test: pd.DataFrame,
                         seed: int = 0) -> np.ndarray:
    """Random draws from each species' empirical training distribution."""
    rng = np.random.default_rng(seed)
    train = _train(panel, split)
    out = np.full(len(test), np.nan, dtype=float)
    pools = {sp: g["value"].to_numpy(float) for sp, g in train.groupby("species", observed=True)}
    for sp, idx in test.groupby("species", observed=True).indices.items():
        pool = pools.get(sp)
        if pool is None or pool.size == 0:
            continue
        out[idx] = rng.choice(pool, size=len(idx), replace=True)
    return out


NULLS: dict[str, Callable[..., np.ndarray]] = {
    "persistence": persistence,
    "lagged_persistence": lagged_persistence,
    "climatology": climatology,
    "species_mean": species_mean,
    "site_mean": site_mean,
    "spatial_neighbour": spatial_neighbour,
    "distribution_matched": distribution_matched,
}


def compute_nulls(panel: pd.DataFrame, split: Split, *, seed: int = 0) -> pd.DataFrame:
    """Long frame of every null's prediction on the test rows, plus ``y_true``."""
    test = test_frame(panel, split)
    if test.empty:
        raise ValueError(f"no observed rows in test year {split.test_year}")
    for name, fn in NULLS.items():
        test[name] = fn(panel, split, test, seed=seed) if name == "distribution_matched" \
            else fn(panel, split, test)
    return test


def cell_resolution_ceiling(panel: pd.DataFrame, split: Split, test: pd.DataFrame,
                            refs: dict[str, np.ndarray]) -> dict[str, Any]:
    """The best score any 0.25 degree predictor could reach on this test set.

    An oracle — every unit predicted by the *true* mean of its cell in the test year — so
    it is never a candidate model. Where units share cells, within-cell variance is
    unreachable for a cell-resolution field; if this ceiling sits below zero the benchmark
    cannot be won at this grain, which is a property of the evaluation unit, not of any
    model.
    """
    from . import metrics

    if test.empty:
        return {"n": 0}
    oracle = test.groupby(["species", "cell_i", "cell_j"], observed=True)["y_true"].transform("mean")
    within = test.assign(_c=oracle)
    ss_within = float(((within["y_true"] - within["_c"]) ** 2).sum())
    grand = test.groupby("species", observed=True)["y_true"].transform("mean")
    ss_total = float(((test["y_true"] - grand) ** 2).sum())

    frame = test.loc[:, ["unit_id", "species", "year", "y_true"]].assign(y_pred=oracle.to_numpy())
    out = {
        "definition": "predict each unit by the true mean of its cell in the test year",
        "units_per_cell_mean": float(
            test.drop_duplicates(["unit_id", "cell_i", "cell_j"])
                .groupby(["cell_i", "cell_j"], observed=True).size().mean()),
        "within_cell_variance_fraction": (ss_within / ss_total) if ss_total else float("nan"),
        "spatial_rho": metrics.spatial_spearman(frame)["mean"],
        "skill": {f"vs_{name}": metrics.skill_score(frame["y_true"], frame["y_pred"], vals)
                  for name, vals in refs.items()},
    }
    out["benchmark_winnable_at_this_grain"] = bool(
        out["skill"]["vs_persistence"] > 0 or not np.isfinite(out["skill"]["vs_persistence"]))
    return out


def _score_set(preds: pd.DataFrame, value_type: str, prev: pd.DataFrame,
               refs: dict[str, np.ndarray]) -> dict[str, Any]:
    from . import metrics

    scores: dict[str, Any] = {}
    for name in NULL_NAMES:
        frame = preds.loc[:, ["unit_id", "species", "year", "y_true", name]] \
                     .rename(columns={name: "y_pred"})
        frame = frame.loc[np.isfinite(frame["y_pred"].to_numpy(float))]
        if frame.empty:
            scores[name] = {"n": 0, "note": "no predictions available"}
            continue
        rec = metrics.evaluate(frame, value_type=value_type, previous=prev)
        rec["coverage"] = float(len(frame) / len(preds))
        rec["skill"] = {}
        for ref_name, ref_vals in refs.items():
            sel = ref_vals[frame.index]
            ok = np.isfinite(sel)
            lg = lambda a: np.log1p(np.clip(np.asarray(a, float), 0.0, None))
            rec["skill"][f"vs_{ref_name}"] = {
                "skill_score": metrics.skill_score(frame["y_true"], frame["y_pred"], sel),
                "skill_score_log1p": metrics.skill_score(
                    lg(frame["y_true"]), lg(frame["y_pred"]), lg(sel)),
                "n_scored": int(ok.sum()),
                "rmse_on_reference_rows": metrics.rmse(
                    frame["y_true"].to_numpy(float)[ok], frame["y_pred"].to_numpy(float)[ok]),
            }
        head = headline_skill(rec["skill"])
        if head is not None:
            rec["skill_vs_strongest_null"] = head
        scores[name] = rec
    return scores


def headline_skill(skill: dict[str, Any]) -> dict[str, Any] | None:
    """Skill against whichever legitimate reference is hardest, or None if none is scoreable.

    A minimum can only lower a reported score, so the headline cannot be gamed upward by
    the choice of reference — and the strongest null is not the same one every year.
    ``invariants.headline_is_worst_reference`` asserts this at run time.
    """
    usable = {k: v["skill_score"] for k, v in skill.items()
              if isinstance(v, dict) and isinstance(v.get("skill_score"), float)
              and np.isfinite(v["skill_score"])}
    if not usable:
        return None
    ref = min(usable, key=usable.get)
    return {"reference": ref, "skill_score": usable[ref]}


def score_nulls(panel: pd.DataFrame, split: Split, *, seed: int = 0) -> dict[str, Any]:
    """Score all seven nulls against every candidate reference forecast.

    Skill is reported against persistence, lagged persistence and climatology rather than
    one hard-wired denominator, so the choice of headline reference is made in the open on
    one set of numbers.
    """
    preds = compute_nulls(panel, split, seed=seed)
    value_type = str(panel["value_type"].iloc[0])
    prev = observed(panel).loc[panel["year"] == split.previous_year,
                               ["unit_id", "species", "value"]].rename(columns={"value": "y_true"})
    refs = {name: preds[name].to_numpy(float) for name in REFERENCES}

    scores = _score_set(preds, value_type, prev, refs)

    strict = preds.loc[np.isfinite(refs["persistence"])]
    sensitivity: dict[str, Any] = {
        "definition": "rows where the unit-species was also observed in the immediately "
                      "preceding year, i.e. where classic persistence is defined",
        "n_rows": int(len(strict)), "n_units": int(strict["unit_id"].nunique()),
        "fraction_of_test": float(len(strict) / len(preds)),
        "selection_bias_warning": "on a rotating panel these are the fast-revisited units and "
                                  "are not a random sample of the domain",
    }
    if len(strict):
        sensitivity["scores"] = _score_set(
            strict.reset_index(drop=True), value_type, prev,
            {"persistence": refs["persistence"][strict.index]})

    return {"task": str(panel["task"].iloc[0]), "value_type": value_type,
            "split": split.as_dict(), "seed": seed, "n_test_rows": int(len(preds)),
            "n_test_units": int(preds["unit_id"].nunique()),
            "n_test_species": int(preds["species"].nunique()),
            "reference_coverage": {k: float(np.isfinite(v).mean()) for k, v in refs.items()},
            "lag_profile": lag_profile(panel, split, preds),
            "cell_resolution_ceiling": cell_resolution_ceiling(panel, split, preds, refs),
            "comparability_note": "each null's raw RMSE is over its own coverage; the entries "
                                  "under 'skill' are computed on the rows where that reference "
                                  "exists and are the comparable figures.",
            "scores": scores,
            "strict_previous_year_subset": sensitivity}
