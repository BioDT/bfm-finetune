"""L0 zero-shot and L1 calibration
"""

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from . import metrics
from .nulls import REFERENCES, Split, compute_nulls, headline_skill, observed, test_frame

SPECIES_TABLE = Path(__file__).resolve().parents[1] / "bfm_species.json"
CALIBRATION_MIN_ROWS = 20


def species_channels(path: Path = SPECIES_TABLE) -> dict[str, int]:
    """Scientific name -> decoder channel index, from the config-derived table."""
    spec = json.loads(path.read_text())
    return {row["scientific_name"]: int(row["i"]) for row in spec["decoder_species_vars"]}


def overlapping_species(panel: pd.DataFrame, channels: dict[str, int] | None = None, *,
                        min_detections: int | None = 30,
                        test_years: Sequence[int] = (2019, 2020)) -> dict[str, int]:
    """Decoder species present in this task, after G0's minimum-sample filter.

    A species carried almost entirely by reconstructed zeros has no ranks to correlate;
    ``min_detections=None`` returns the raw overlap, which is what the gate itself reports.
    """
    channels = channels if channels is not None else species_channels()
    present = {name: idx for name, idx in channels.items()
               if name in set(panel["species"].unique())}
    if min_detections is None:
        return present

    obs = panel.loc[panel["observed"] & panel["year"].isin(list(test_years))]
    detected = obs.loc[obs["value"] > 0].groupby("species", observed=True).size()
    return {name: idx for name, idx in present.items()
            if int(detected.get(name, 0)) >= min_detections}


def gather_predictions(field: np.ndarray, frame: pd.DataFrame, channels: dict[str, int]) -> np.ndarray:
    """Sample ``field[channel, H, W]`` at each row's cell and species.

    The channel mapping comes from the config-derived table rather than field order — a
    wrong gather returns plausible numbers from the wrong cells.
    """
    field = np.asarray(field)
    if field.ndim != 3:
        raise ValueError(f"expected [species, H, W], got shape {field.shape}")
    out = np.full(len(frame), np.nan, dtype=float)
    species = frame["species"].to_numpy()
    ci = frame["cell_i"].to_numpy(dtype=int)
    cj = frame["cell_j"].to_numpy(dtype=int)
    for name, channel in channels.items():
        sel = species == name
        if sel.any():
            out[sel] = field[channel, ci[sel], cj[sel]]
    return out


def l0_zero_shot(panel: pd.DataFrame, split: Split, field: np.ndarray,
                 channels: dict[str, int] | None = None) -> dict[str, Any]:
    """Rank the decoded species field against observed values. Nothing is fitted."""
    channels = channels if channels is not None else overlapping_species(panel)
    if not channels:
        return {"rung": "L0", "n_species": 0,
                "note": "no overlap between the decoder's species and this task; L0 is not "
                        "defined here and the task starts at L2"}

    test = test_frame(panel, split)
    test = test.loc[test["species"].isin(channels)]
    preds = gather_predictions(field, test, channels)
    frame = test.assign(y_pred=preds)
    frame = frame.loc[np.isfinite(frame["y_pred"].to_numpy(float))]

    nulls = compute_nulls(panel, split)
    refs = _reference_columns(nulls, frame)
    out = metrics.evaluate(frame, value_type=str(panel["value_type"].iloc[0]))
    for key in ("rmse", "rmse_log"):
        out.pop(key, None)
    out.update({"rung": "L0", "fitted_parameters": 0,
                "species": sorted(channels), "n_species": len(channels),
                "skill": {f"vs_{k}": None for k in refs},
                "per_species": metrics.per_species(frame, value_type=str(panel["value_type"].iloc[0])),
                "note": "no parameters fitted. Skill scores and RMSE are deliberately not "
                        "reported: the decoded field is a normalised occurrence density and "
                        "the target a standardised count, so a squared-error comparison "
                        "measures the unit mismatch, not the model. L0 is a rank claim and "
                        "Spearman is the only honest metric for it."})
    return out


def _reference_columns(nulls: pd.DataFrame, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    key = ["unit_id", "species", "year"]
    merged = frame.loc[:, key].merge(nulls.loc[:, key + list(REFERENCES)], on=key, how="left")
    return {name: merged[name].to_numpy(float) for name in REFERENCES}


def _fit_poisson(x: np.ndarray, y: np.ndarray) -> tuple[float, float, bool]:
    """Two-parameter Poisson GLM with a log link: ``E[y] = exp(a + b·x)``."""
    import statsmodels.api as sm

    design = sm.add_constant(np.asarray(x, float).reshape(-1, 1), has_constant="add")
    try:
        fit = sm.GLM(np.asarray(y, float), design, family=sm.families.Poisson()).fit()
        return float(fit.params[0]), float(fit.params[1]), bool(fit.converged)
    except Exception:
        return float("nan"), float("nan"), False


def l1_calibrate(panel: pd.DataFrame, split: Split, fields: dict[int, np.ndarray],
                 channels: dict[str, int] | None = None) -> dict[str, Any]:
    """Learn only the link from the model's field to the observed scale, per species.

    ``fields`` maps year -> decoded species field. The link is fitted on the training
    years and applied to the test year; nothing about the test year informs the fit.
    """
    channels = channels if channels is not None else overlapping_species(panel)
    if not channels:
        return {"rung": "L1", "n_species": 0,
                "note": "no species overlap; L1 is not defined for this task"}

    obs = observed(panel)
    train_years = [y for y in split.train_years if y in fields]
    train = obs.loc[obs["year"].isin(train_years) & obs["species"].isin(channels)].copy()
    if train.empty:
        raise ValueError("no training rows with a matching decoded field")

    # Sort before the grouped gather so the concatenated result aligns row-for-row: pandas
    # emits groups in sorted key order and preserves within-group row order.
    train = train.sort_values("year", kind="stable").reset_index(drop=True)
    train["x"] = np.concatenate([gather_predictions(fields[y], g, channels)
                                 for y, g in train.groupby("year", observed=True)])

    params, diagnostics = {}, {}
    for name, g in train.groupby("species", observed=True):
        g = g.loc[np.isfinite(g["x"].to_numpy(float))]
        if len(g) < CALIBRATION_MIN_ROWS:
            diagnostics[str(name)] = {"status": "too few training rows", "n": int(len(g))}
            continue
        a, b, ok = _fit_poisson(g["x"].to_numpy(float), g["value"].to_numpy(float))
        params[str(name)] = {"intercept": a, "slope": b, "converged": ok, "n_train": int(len(g))}
        diagnostics[str(name)] = {"status": "fitted" if ok else "did not converge"}

    test = test_frame(panel, split)
    test = test.loc[test["species"].isin(channels)]
    x = gather_predictions(fields[split.test_year], test, channels)
    pred = np.full(len(test), np.nan, dtype=float)
    for name, p in params.items():
        sel = (test["species"].to_numpy() == name) & np.isfinite(x)
        if sel.any() and np.isfinite(p["intercept"]) and np.isfinite(p["slope"]):
            pred[sel] = np.exp(p["intercept"] + p["slope"] * x[sel])

    frame = test.assign(y_pred=pred)
    frame = frame.loc[np.isfinite(frame["y_pred"].to_numpy(float))]
    nulls = compute_nulls(panel, split)
    refs = _reference_columns(nulls, frame)

    out = metrics.evaluate(frame, value_type=str(panel["value_type"].iloc[0]))
    out.update({"rung": "L1", "fitted_parameters": 2 * len(params),
                "n_species": len(params), "train_years": train_years,
                "parameters": params, "diagnostics": diagnostics,
                "skill": {f"vs_{k}": {"skill_score": metrics.skill_score(
                              frame["y_true"], frame["y_pred"], v),
                          "n_scored": int(np.isfinite(v).sum())}
                          for k, v in refs.items()},
                "in_sample": False,
                "note": "link fitted on training years only and applied to the held-out test "
                        "year; two parameters per species"})
    # L1 is on the target's scale, so a squared-error headline is meaningful here.
    head = headline_skill(out["skill"])
    if head is not None:
        out["skill_vs_strongest_null"] = head
    return out


def run_ladder(panel: pd.DataFrame, splits: Iterable[Split], fields: dict[int, np.ndarray],
               out_dir: str | Path, runner: Any = None) -> dict[str, Any]:
    from ..common.invariants import ladder_l1_preserves_ranking
    from ..common.runner import write_json

    task = str(panel["task"].iloc[0])
    channels = overlapping_species(panel)
    results = []
    for split in splits:
        rec = {"split": split.as_dict(),
               "L0": l0_zero_shot(panel, split, fields[split.test_year], channels),
               "L1": l1_calibrate(panel, split, fields, channels)}
        ladder_l1_preserves_ranking(rec["L0"], rec["L1"], len(channels))
        results.append(rec)

    record = {"task": task, "n_overlapping_species": len(channels),
              "species": sorted(channels), "results": results}
    write_json(Path(out_dir) / f"ladder_{task}.json", record)
    return record
