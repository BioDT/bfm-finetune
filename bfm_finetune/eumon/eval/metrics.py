"""Skill score, Spearman, log-RMSE, MCC, AUC-PR.

Every metric consumes a long frame with columns ``unit_id, species, year, y_true, y_pred``
and scores only rows the panel marked observed. Spatial and temporal Spearman are
different claims and are never merged.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EVAL_COLUMNS = ("unit_id", "species", "year", "y_true", "y_pred")


def _clean(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(y) & np.isfinite(p)
    return y[ok], p[ok]


def mse(y_true: Any, y_pred: Any) -> float:
    y, p = _clean(np.asarray(y_true, float), np.asarray(y_pred, float))
    return float(np.mean((y - p) ** 2)) if y.size else float("nan")


def rmse(y_true: Any, y_pred: Any) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def rmse_log(y_true: Any, y_pred: Any) -> float:
    """RMSE on log(y+1); the abundance targets are modelled on that scale."""
    y, p = _clean(np.asarray(y_true, float), np.asarray(y_pred, float))
    if not y.size:
        return float("nan")
    return float(np.sqrt(np.mean((np.log1p(np.clip(y, 0, None)) - np.log1p(np.clip(p, 0, None))) ** 2)))


def skill_score(y_true: Any, y_pred: Any, y_ref: Any) -> float:
    """``SS = 1 - MSE_model / MSE_reference``. Positive means beating the reference."""
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    r = np.asarray(y_ref, float)
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(r)
    if not ok.any():
        return float("nan")
    denom = float(np.mean((y[ok] - r[ok]) ** 2))
    if denom == 0:
        return float("nan")
    return float(1.0 - float(np.mean((y[ok] - p[ok]) ** 2)) / denom)


def _spearman(a: np.ndarray, b: np.ndarray, min_n: int = 5) -> float:
    a, b = _clean(np.asarray(a, float), np.asarray(b, float))
    if a.size < min_n or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    rho = spearmanr(a, b).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def spatial_spearman(df: pd.DataFrame, min_n: int = 5) -> dict[str, Any]:
    """Rank correlation across units, within a (species, year). Spatial skill."""
    per = {}
    for (sp, yr), g in df.groupby(["species", "year"], observed=True):
        per[f"{sp}|{int(yr)}"] = _spearman(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), min_n)
    vals = np.array([v for v in per.values() if np.isfinite(v)], dtype=float)
    return {"mean": float(vals.mean()) if vals.size else float("nan"),
            "sd": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
            "n_groups": int(vals.size), "per_group": per}


def temporal_spearman(df: pd.DataFrame, min_n: int = 5) -> dict[str, Any]:
    """Rank correlation across years, within a (species, unit). Temporal skill."""
    per = {}
    for (sp, unit), g in df.groupby(["species", "unit_id"], observed=True):
        g = g.sort_values("year")
        per[f"{sp}|{unit}"] = _spearman(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), min_n)
    vals = np.array([v for v in per.values() if np.isfinite(v)], dtype=float)
    return {"mean": float(vals.mean()) if vals.size else float("nan"),
            "sd": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
            "n_groups": int(vals.size)}


def sign_coverage(df: pd.DataFrame, previous: pd.DataFrame) -> float:
    """Fraction of unit-species pairs whose predicted direction of change is right.

    ``previous`` supplies ``y_true`` at t-1 for the same unit-species pairs.
    """
    prev = previous.loc[:, ["unit_id", "species", "y_true"]].rename(columns={"y_true": "y_prev"})
    m = df.merge(prev, on=["unit_id", "species"], how="inner")
    if m.empty:
        return float("nan")
    obs = np.sign(m["y_true"].to_numpy(float) - m["y_prev"].to_numpy(float))
    pred = np.sign(m["y_pred"].to_numpy(float) - m["y_prev"].to_numpy(float))
    ok = np.isfinite(obs) & np.isfinite(pred)
    return float(np.mean(obs[ok] == pred[ok])) if ok.any() else float("nan")


def auc_pr(y_bin: Any, score: Any) -> float:
    from sklearn.metrics import average_precision_score

    y, p = _clean(np.asarray(y_bin, float), np.asarray(score, float))
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def mcc(y_bin: Any, pred_bin: Any) -> float:
    """Matthews correlation. A trivial all-absent predictor scores exactly 0."""
    from sklearn.metrics import matthews_corrcoef

    y, p = _clean(np.asarray(y_bin, float), np.asarray(pred_bin, float))
    if y.size == 0:
        return float("nan")
    return float(matthews_corrcoef(y.astype(int), p.astype(int)))


def auc_roc(y_bin: Any, score: Any) -> float:
    from sklearn.metrics import roc_auc_score

    y, p = _clean(np.asarray(y_bin, float), np.asarray(score, float))
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def brier(y_bin: Any, prob: Any) -> float:
    """Mean squared error of a probability forecast — the proper scoring rule for Task B.

    AUC and TSS only rank; Brier penalises being confidently wrong about the level.
    """
    y, p = _clean(np.asarray(y_bin, float), np.asarray(prob, float))
    return float(np.mean((y - np.clip(p, 0.0, 1.0)) ** 2)) if y.size else float("nan")


def tss_at_best_threshold(y_bin: Any, score: Any) -> dict[str, Any]:
    """True Skill Statistic (sensitivity + specificity - 1), maximised over the threshold.

    The standard SDM discrimination statistic (Allouche et al. 2006); unlike a fixed 0.5
    cut it does not depend on prevalence, which matters on a proportion averaging 0.071.
    """
    from sklearn.metrics import roc_curve

    y, p = _clean(np.asarray(y_bin, float), np.asarray(score, float))
    if y.size == 0 or len(np.unique(y)) < 2:
        return {"tss": float("nan"), "threshold": float("nan")}
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    k = int(np.argmax(j))
    return {"tss": float(j[k]), "threshold": float(thr[k]),
            "sensitivity": float(tpr[k]), "specificity": float(1 - fpr[k])}


def calibration(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Slope and intercept of observed regressed on predicted; 1 and 0 mean calibrated.

    A slope below 1 is systematic shrinkage.
    """
    y, p = _clean(np.asarray(y_true, float), np.asarray(y_pred, float))
    if y.size < 3 or np.ptp(p) == 0:
        return {"slope": float("nan"), "intercept": float("nan")}
    slope, intercept = np.polyfit(p, y, 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def poisson_deviance(y_true: Any, y_pred: Any) -> float:
    """Mean Poisson deviance — a proper scoring rule for counts, unlike RMSE, which on a
    heavy-tailed target is decided by a handful of large units."""
    y, p = _clean(np.asarray(y_true, float), np.asarray(y_pred, float))
    if not y.size:
        return float("nan")
    p = np.clip(p, 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / p), 0.0)
    return float(np.mean(2.0 * (term - (y - p))))


def evaluate(df: pd.DataFrame, *, value_type: str, reference: pd.Series | None = None,
             previous: pd.DataFrame | None = None, threshold: float = 0.5) -> dict[str, Any]:
    """Score one prediction set. ``reference`` is the persistence null, for the skill score."""
    missing = [c for c in EVAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"eval frame missing {missing}")
    df = df.loc[np.isfinite(df["y_true"].to_numpy(float))]
    y = df["y_true"].to_numpy(float)
    p = df["y_pred"].to_numpy(float)

    out: dict[str, Any] = {
        "n": int(len(df)), "n_units": int(df["unit_id"].nunique()),
        "n_species": int(df["species"].nunique()),
        "years": [int(v) for v in sorted(df["year"].unique())],
        "rmse": rmse(y, p), "rmse_log": rmse_log(y, p),
        "spatial_rho": spatial_spearman(df),
        "calibration": calibration(y, p),
    }
    out["spatial_rho"].pop("per_group", None)
    # temporal_rho needs several test years per (species, unit); computed only when the
    # frame actually spans enough years rather than emitting a permanent NaN.
    if df["year"].nunique() >= 5:
        out["temporal_rho"] = temporal_spearman(df)
    if value_type != "prevalence":
        out["poisson_deviance"] = poisson_deviance(y, p)
    if reference is not None:
        out["skill_vs_persistence"] = skill_score(y, p, reference.to_numpy(float))
    if previous is not None:
        out["sign_coverage"] = sign_coverage(df, previous)
    if value_type == "prevalence":
        y_bin = (y > 0).astype(int)
        out["prevalence"] = float(y_bin.mean())
        out["auc_pr"] = auc_pr(y_bin, p)
        out["auc_roc"] = auc_roc(y_bin, p)
        out["brier"] = brier(y_bin, p)
        out["tss"] = tss_at_best_threshold(y_bin, p)
        # Kept for continuity with earlier tables; read `tss` instead.
        out["mcc_at_0.5"] = mcc(y_bin, (p >= threshold).astype(int))
    return out


def per_species(df: pd.DataFrame, *, value_type: str,
                reference: pd.Series | None = None) -> dict[str, dict[str, Any]]:
    out = {}
    for sp, g in df.groupby("species", observed=True):
        ref = reference.loc[g.index] if reference is not None else None
        out[str(sp)] = evaluate(g, value_type=value_type, reference=ref)
    return out


def bootstrap_ci(y_true: Any, y_pred: Any, y_ref: Any, n: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for the skill score, resampling rows."""
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    r = np.asarray(y_ref, float)
    rng = np.random.default_rng(seed)
    idx = np.arange(y.size)
    vals = np.empty(n, dtype=float)
    for k in range(n):
        s = rng.choice(idx, size=idx.size, replace=True)
        vals[k] = skill_score(y[s], p[s], r[s])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
