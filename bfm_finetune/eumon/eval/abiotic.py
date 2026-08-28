"""The abiotic (CHELSA) task: nulls first, then the probe.

Six nulls, each ruling out a specific trivial explanation: ``month_of_year`` (per-cell
calendar-month mean over training years — the bar any abiotic result must clear),
``cell_climatology`` (geography with no seasonality), ``persistence`` (last month),
``lagged_year`` (same month one year earlier), ``model_passthrough`` (the model's own
decoded field, unfitted) and ``era5_observed`` (the observed ERA5 field for the target
month — the transfer ceiling, since CHELSA v2.1 is downscaled ERA5 and block-averaging it
back to 0.25 degrees returns ERA5 to 0.26 K RMSE on ``tas``).

Scores are reported per cell and on the domain mean, the latter purely to reproduce the
published setup and show what it hides. They are never merged.
"""

import calendar
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

NULL_NAMES = ("month_of_year", "cell_climatology", "persistence", "lagged_year",
              "model_passthrough", "era5_observed")

# The BioCube channel that is each CHELSA variable's counterpart, and how its units map.
# ERA5 `tp` is a daily accumulation in metres, so metres/day x 1000 mm/m x days = mm/month,
# which is CHELSA's unit. The chain is checked, not assumed: it puts the 2017-2019 domain
# mean at 67.5 against CHELSA's 67.1 mm/month.
ERA5_CHANNEL = {"tas": ("surface_variables", "t2m"), "pr": ("climate_variables", "tp")}


def to_chelsa_units(field: np.ndarray, variable: str,
                    index: Sequence[tuple[int, int]]) -> np.ndarray:
    """Convert an ERA5-unit field ``[n_months, H, W]`` into CHELSA's units."""
    if variable == "tas":
        return field
    days = np.array([calendar.monthrange(y, m)[1] for y, m in index], dtype=np.float64)
    return field * 1000.0 * days[:, None, None]


def denormalise(field: np.ndarray, variable: str, stats: dict[str, Any],
                index: Sequence[tuple[int, int]]) -> np.ndarray:
    """Invert the model's training scaling, then convert to CHELSA units.

    The statistics are BioCube's (ERA5's), not CHELSA's, and deliberately so: they are what
    the field was normalised with, and reaching for the target's own statistics would leak
    the target into the prediction. Empirically it is also the correct branch — the config
    labels the mode ``normalize`` but ``scaler.py`` routes that to min-max, which returns
    167-407 K for ``t2m``; mean/std returns 273-303 K against CHELSA's 263-301 K.
    """
    group, name = ERA5_CHANNEL[variable]
    s = stats[group][name]
    return to_chelsa_units(field * s["std"] + s["mean"], variable, index)


@dataclass(frozen=True)
class MonthlySplit:
    """Chronological split over (year, month) pairs."""

    train_years: tuple[int, ...]
    val_years: tuple[int, ...]
    test_years: tuple[int, ...]

    def mask(self, index: Sequence[tuple[int, int]], which: str) -> np.ndarray:
        years = {"train": self.train_years, "val": self.val_years, "test": self.test_years}[which]
        return np.array([y in years for y, _ in index], dtype=bool)

    def as_dict(self) -> dict[str, Any]:
        return {"train_years": list(self.train_years), "val_years": list(self.val_years),
                "test_years": list(self.test_years)}


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination against the mean of the truth, over finite pairs."""
    y = np.asarray(y_true, float).ravel()
    p = np.asarray(y_pred, float).ravel()
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 2:
        return float("nan")
    y, p = y[ok], p[ok]
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, float).ravel()
    p = np.asarray(y_pred, float).ravel()
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2))) if ok.any() else float("nan")


def _month_array(index: Sequence[tuple[int, int]]) -> np.ndarray:
    return np.array([m for _, m in index], dtype=int)


def build_nulls(series: np.ndarray, index: Sequence[tuple[int, int]], split: MonthlySplit,
                model_field: np.ndarray | None = None,
                era5_field: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Predictions for every null, shaped like ``series`` and NaN outside the test months.

    ``series`` is ``[n_months, H, W]``; ``model_field`` is the model's own decoded field and
    ``era5_field`` the observed ERA5 field, both already in the target's units and on the
    same axes.
    """
    months = _month_array(index)
    train = split.mask(index, "train")
    test = split.mask(index, "test")
    out = {name: np.full_like(series, np.nan, dtype=np.float32) for name in NULL_NAMES}

    for m in range(1, 13):
        fit = train & (months == m)
        if not fit.any():
            continue
        out["month_of_year"][test & (months == m)] = np.nanmean(series[fit], axis=0)

    out["cell_climatology"][test] = np.nanmean(series[train], axis=0)

    for k in np.flatnonzero(test):
        if k - 1 >= 0:
            out["persistence"][k] = series[k - 1]
        if k - 12 >= 0:
            out["lagged_year"][k] = series[k - 12]

    for name, field in (("model_passthrough", model_field), ("era5_observed", era5_field)):
        if field is None:
            out.pop(name)
        else:
            out[name][test] = field[test]
    return out


def score(series: np.ndarray, predictions: dict[str, np.ndarray], index: Sequence[tuple[int, int]],
          split: MonthlySplit, variable: str) -> dict[str, Any]:
    """Score every predictor per cell and, separately, on the domain mean.

    The domain-mean column exists only to reproduce the published setup: it is what the
    original experiment measured after collapsing space, and reporting it beside the per-cell
    column is what makes the difference visible.
    """
    test = split.mask(index, "test")
    truth = series[test]
    truth_mean = np.nanmean(truth.reshape(truth.shape[0], -1), axis=1)

    rows = {}
    for name, pred in predictions.items():
        p = pred[test]
        pm = np.nanmean(p.reshape(p.shape[0], -1), axis=1)
        rows[name] = {
            "per_cell_r2": r2(truth, p), "per_cell_rmse": rmse(truth, p),
            "domain_mean_r2": r2(truth_mean, pm), "domain_mean_rmse": rmse(truth_mean, pm),
            "coverage": float(np.isfinite(p).mean()),
        }
    return {"variable": variable, "split": split.as_dict(),
            "n_test_months": int(test.sum()), "n_cells": int(series.shape[1] * series.shape[2]),
            "scores": rows,
            "note": "per_cell is the honest figure. domain_mean reproduces the published "
                    "setup, in which space is averaged away before scoring and the target "
                    "becomes one seasonal series per month."}


def seasonality_share(series: np.ndarray, index: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """How much of the target is explained by the calendar month alone.

    Reported for the domain mean and per cell, because the two differ enormously and the
    published experiment only ever saw the first.
    """
    months = _month_array(index)
    domain = np.nanmean(series.reshape(series.shape[0], -1), axis=1)
    clim = np.array([domain[months == m].mean() for m in months])

    cell_clim = np.stack([np.nanmean(series[months == m], axis=0) for m in range(1, 13)])
    pred = np.stack([cell_clim[m - 1] for m in months])
    return {"domain_mean_r2_month_of_year": r2(domain, clim),
            "per_cell_r2_month_of_year": r2(series, pred),
            "interpretation": "if the domain-mean figure is high, a zero-parameter predictor "
                              "that knows only the calendar month explains the published "
                              "target, and no model result on it is interpretable"}


# The rebuilt probe: a tiny head on frozen decoded fields, scored per cell.

@dataclass(frozen=True)
class ProbeArm:
    """One column of the probe table."""

    name: str
    use_model: bool
    use_month: bool
    use_clim: bool
    note: str


PROBE_ARMS = (
    ProbeArm("month_only", False, True, False,
             "calendar month alone, one head for the whole grid. It carries no spatial "
             "information, so it is a diagnostic of how much of the target is spatial — "
             "not the bar to clear"),
    ProbeArm("climatology_only", False, False, True,
             "each cell's own month-of-year mean over the training years. This reproduces "
             "the strongest null and IS the bar the model has to clear"),
    ProbeArm("model_only", True, False, False,
             "the frozen decoded fields alone"),
    ProbeArm("model_plus_clim", True, False, True,
             "both — the practical predictor. Its gain over climatology_only is the model's "
             "actual contribution"),
)


class AbioticHead:
    """Linear -> GELU -> Linear, the same width as the biotic L2 head.

    Deliberately without the biotic head's leading ``LayerNorm``: normalising across the
    feature axis destroys the low-dimensional arms (with two columns it keeps only which is
    larger). Columns are already standardised on the training rows before the head.
    """

    def __new__(cls, n_features: int, n_targets: int = 2, hidden: int = 64):
        import torch.nn as nn

        return nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(),
                             nn.Linear(hidden, n_targets))


def _month_onehot(index: Sequence[tuple[int, int]], rows: np.ndarray) -> np.ndarray:
    """One-hot calendar month for a set of (month-slot, cell) rows."""
    out = np.zeros((len(rows), 12), dtype=np.float32)
    out[np.arange(len(rows)), [index[t][1] - 1 for t in rows]] = 1.0
    return out


def train_climatology(truth: dict[str, np.ndarray], index: Sequence[tuple[int, int]],
                      split: MonthlySplit, variables: Sequence[str]) -> np.ndarray:
    """Per-cell month-of-year mean over the **training** years, as ``[n_vars, 12, n_cells]``.

    Fitted on training years only; using all years would leak the test period into the very
    baseline the model is measured against.
    """
    months = _month_array(index)
    train = split.mask(index, "train")
    n_cells = truth[variables[0]].shape[1] * truth[variables[0]].shape[2]
    out = np.full((len(variables), 12, n_cells), np.nan, dtype=np.float32)
    for vi, v in enumerate(variables):
        flat = truth[v].reshape(len(index), -1)
        for m in range(1, 13):
            sel = train & (months == m)
            if sel.any():
                out[vi, m - 1] = np.nanmean(flat[sel], axis=0)
    return out


def _gather(features: np.ndarray, arm: ProbeArm, index: Sequence[tuple[int, int]],
            t_rows: np.ndarray, cell_rows: np.ndarray,
            clim: np.ndarray | None = None) -> np.ndarray:
    """Design matrix for the given (month-slot, flat-cell) rows."""
    parts = []
    if arm.use_model:
        flat = features.reshape(features.shape[0], features.shape[1], -1)
        parts.append(flat[t_rows, :, cell_rows].astype(np.float32))
    if arm.use_month:
        parts.append(_month_onehot(index, t_rows))
    if arm.use_clim:
        if clim is None:
            raise ValueError(f"arm {arm.name} needs the training climatology")
        m_rows = np.array([index[t][1] - 1 for t in t_rows])
        parts.append(clim[:, m_rows, cell_rows].T.astype(np.float32))
    return np.concatenate(parts, axis=1)


def _sample_rows(slots: np.ndarray, n_cells: int, n_rows: int, rng) -> tuple[np.ndarray, np.ndarray]:
    t = rng.choice(slots, n_rows, replace=True)
    c = rng.integers(0, n_cells, n_rows)
    return t, c


def fit_abiotic_probe(features: np.ndarray, truth: dict[str, np.ndarray],
                      index: Sequence[tuple[int, int]], split: MonthlySplit, arm: ProbeArm,
                      *, variables: Sequence[str] = ("tas", "pr"), hidden: int = 64,
                      epochs: int = 40, batch: int = 4096, max_train_rows: int = 400_000,
                      max_val_rows: int = 50_000, patience: int = 6, seed: int = 0,
                      device: str = "cpu", lr: float = 1e-3) -> dict[str, Any]:
    """Fit one arm and score it on every test cell.

    Training rows are a seeded sample of (month, cell) pairs — the full training set is
    ~8M rows and the head has ~10k parameters, so sampling costs nothing and the sample size
    is reported. Scoring uses **all** test cells, month by month, never a sample.
    """
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_cells = features.shape[2] * features.shape[3]
    train_slots = np.flatnonzero(split.mask(index, "train"))
    val_slots = np.flatnonzero(split.mask(index, "val"))
    test_slots = np.flatnonzero(split.mask(index, "test"))
    clim = train_climatology(truth, index, split, variables) if arm.use_clim else None

    def design(slots, n_rows):
        t, c = _sample_rows(slots, n_cells, n_rows, rng)
        x = _gather(features, arm, index, t, c, clim)
        y = np.stack([truth[v].reshape(len(index), -1)[t, c] for v in variables], axis=1)
        ok = np.isfinite(x).all(1) & np.isfinite(y).all(1)
        return x[ok], y[ok].astype(np.float32)

    xtr, ytr = design(train_slots, max_train_rows)
    xva, yva = design(val_slots, max_val_rows)
    if len(xtr) < 100 or len(xva) < 10:
        return {"arm": arm.name, "error": "not enough finite training rows",
                "n_train": int(len(xtr))}

    xm, xs = xtr.mean(0), xtr.std(0) + 1e-6
    ym, ys = ytr.mean(0), ytr.std(0) + 1e-6
    dev = torch.device(device)
    Xtr = torch.tensor((xtr - xm) / xs, device=dev)
    Ytr = torch.tensor((ytr - ym) / ys, device=dev)
    Xva = torch.tensor((xva - xm) / xs, device=dev)
    Yva = torch.tensor((yva - ym) / ys, device=dev)

    head = AbioticHead(Xtr.shape[1], len(variables), hidden).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    best, best_state, waited, history = float("inf"), None, 0, []
    for _ in range(epochs):
        head.train()
        order = torch.randperm(len(Xtr), device=dev)
        for k in range(0, len(Xtr), batch):
            sel = order[k:k + batch]
            opt.zero_grad()
            loss_fn(head(Xtr[sel]), Ytr[sel]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            v = float(loss_fn(head(Xva), Yva))
        history.append(v)
        if v < best - 1e-5:
            best, waited = v, 0
            best_state = {k: t.detach().clone() for k, t in head.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state)

    # Score every test cell, one month at a time so the design matrix stays small.
    head.eval()
    pred = {v: np.full((len(index), n_cells), np.nan, dtype=np.float32) for v in variables}
    cells = np.arange(n_cells)
    with torch.no_grad():
        for t in test_slots:
            x = _gather(features, arm, index, np.full(n_cells, t), cells, clim)
            ok = np.isfinite(x).all(1)
            if not ok.any():
                continue
            out = head(torch.tensor((x[ok] - xm) / xs, device=dev)).cpu().numpy() * ys + ym
            for j, v in enumerate(variables):
                pred[v][t, ok] = out[:, j]

    scores = {}
    for v in variables:
        truth_test = truth[v].reshape(len(index), -1)[test_slots]
        pred_test = pred[v][test_slots]
        scores[v] = {
            "per_cell_r2": r2(truth_test, pred_test),
            "per_cell_rmse": rmse(truth_test, pred_test),
            "domain_mean_r2": r2(np.nanmean(truth_test, axis=1), np.nanmean(pred_test, axis=1)),
            "coverage": float(np.isfinite(pred_test).mean()),
        }
    return {"arm": arm.name, "use_model": arm.use_model, "use_month": arm.use_month,
            "note": arm.note, "n_features": int(Xtr.shape[1]),
            "n_head_parameters": int(sum(p.numel() for p in head.parameters())),
            "n_train_rows_used": int(len(xtr)), "n_train_rows_available": int(len(train_slots) * n_cells),
            "n_val_rows_used": int(len(xva)), "n_test_cells_scored": int(len(test_slots) * n_cells),
            "epochs_run": len(history), "early_stopped": len(history) < epochs,
            "best_val_mse": best, "seed": seed, "scores": scores}
