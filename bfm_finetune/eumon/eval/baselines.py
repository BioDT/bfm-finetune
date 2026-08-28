"""Learned baselines scored directly on the benchmark's target panels, with no model inputs.

Five fits are exposed through ``FIT_FUNCTIONS``: per-species trend GLMs with and without a
regional covariate (the UKBMS/TRIM index family), a negative-binomial GAM, a per-species
random-forest SDM, and a ConvLSTM over the gridded target field. Every fit and every score
touches only panel rows with ``observed == True``; missing cells are never zero-filled.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import metrics
from .nulls import REFERENCES, Split, compute_nulls, observed, test_frame
from .. import batch
from ..common.runner import Runner, artefacts_root, write_json
from ..model import load_config

MIN_TRAIN_ROWS = 5       # fewer rows and IRLS is either undetermined or noise
OFFSET_EPS = 1e-3        # keeps log/logit of a zero site-mean finite


@dataclass(frozen=True)
class BaselineFit:
    """One baseline's predictions on a split's test rows, in the shape ``metrics.evaluate``
    and ``nulls`` expect: ``unit_id, species, year, y_true, y_pred``."""

    name: str
    predictions: pd.DataFrame
    diagnostics: dict[str, Any]
    wall_s: float


def _train_rows(panel: pd.DataFrame, split: Split) -> pd.DataFrame:
    # Mirrors ``nulls._train``: null and baseline scores must share one training window.
    return observed(panel).loc[panel["year"].isin(split.train_years)]


def _previous_year_frame(panel: pd.DataFrame, split: Split) -> pd.DataFrame:
    return observed(panel).loc[panel["year"] == split.previous_year,
                               ["unit_id", "species", "value"]].rename(columns={"value": "y_true"})


def fit_glm(panel: pd.DataFrame, split: Split, seed: int = 0, *,
            stratify: bool = False) -> BaselineFit:
    """Per-species Poisson (Binomial for Task B) trend GLM, the UKBMS-index method family.

    Site effects are training-period means passed as a GLM offset (log for Poisson, logit
    for Binomial), with a shared log-linear year term fitted on top — a two-step
    approximation to a site x year fixed-effects design, which does not converge at Task
    C's width. IRLS is deterministic; ``seed`` exists for interface parity only.
    """
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    t0 = time.time()
    test = test_frame(panel, split)
    train = _train_rows(panel, split)
    value_type = str(panel["value_type"].iloc[0])
    is_binomial = value_type == "prevalence"
    year_centre = float(np.mean(split.train_years))

    # Regional strata for the covariate: latitude quartiles of the surveyed units, computed
    # on training rows only.
    units = train.drop_duplicates("unit_id")[["unit_id", "lat"]]
    edges = np.quantile(units["lat"].to_numpy(float), [0.25, 0.5, 0.75])
    stratum_of = pd.Series(np.digitize(units["lat"].to_numpy(float), edges),
                           index=units["unit_id"].to_numpy())

    y_pred = np.full(len(test), np.nan, dtype=float)
    per_species: dict[str, dict[str, Any]] = {}
    counts = {"converged": 0, "no_year_variation": 0, "insufficient_data": 0, "failed": 0,
              "did_not_converge": 0}

    test_idx_by_species = test.groupby("species", observed=True).indices
    train_by_species = {sp: g for sp, g in train.groupby("species", observed=True)}

    for sp, idx in test_idx_by_species.items():
        train_g = train_by_species.get(sp)
        n_rows = 0 if train_g is None else int(len(train_g))
        if train_g is None or n_rows < MIN_TRAIN_ROWS:
            per_species[str(sp)] = {"status": "insufficient_data", "n_train_rows": n_rows,
                                    "n_train_sites": 0, "n_test_rows": int(len(idx))}
            counts["insufficient_data"] += 1
            continue

        site_mean = train_g.groupby("unit_id", observed=True)["value"].mean()
        diag: dict[str, Any] = {"n_train_rows": n_rows, "n_train_sites": int(site_mean.size),
                                "n_test_rows": int(len(idx))}

        n_years = train_g["year"].nunique()
        has_trend = n_years >= 2
        year_c = train_g["year"].to_numpy(float) - year_centre
        # Year effects per stratum: TRIM's documented covariate extension. Without it a
        # site offset times one shared year term never reorders sites.
        strata = train_g["unit_id"].map(stratum_of).fillna(0).to_numpy(int)
        levels = np.unique(strata) if stratify else np.array([0])
        if stratify and has_trend and len(levels) > 1:
            blocks = [np.column_stack([(strata == L).astype(float),
                                       (strata == L).astype(float) * year_c])
                      for L in levels]
            exog = np.column_stack(blocks)
        elif has_trend:
            exog = np.column_stack([np.ones(n_rows), year_c])
        else:
            exog = np.ones((n_rows, 1))

        if is_binomial:
            p = np.clip(train_g["unit_id"].map(site_mean).to_numpy(float), OFFSET_EPS, 1 - OFFSET_EPS)
            offset = np.log(p / (1 - p))
            family = sm.families.Binomial()
            fit_kwargs: dict[str, Any] = {"var_weights": train_g["effort"].to_numpy(float)}
        else:
            offset = np.log(train_g["unit_id"].map(site_mean).to_numpy(float) + OFFSET_EPS)
            family = sm.families.Poisson()
            fit_kwargs = {}

        try:
            endog = train_g["value"].to_numpy(float)
            result = sm.GLM(endog, exog, family=family, offset=offset, **fit_kwargs).fit()
        except (PerfectSeparationError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            diag["status"] = f"failed: {type(exc).__name__}: {exc}"
            counts["failed"] += 1
            per_species[str(sp)] = diag
            continue

        converged = bool(getattr(result, "converged", True))
        diag["converged"] = converged
        diag["year_coef"] = float(result.params[1]) if has_trend else None
        diag["n_strata"] = int(len(levels)) if has_trend else 1
        if not converged:
            diag["status"] = "did_not_converge"
            counts["did_not_converge"] += 1
            per_species[str(sp)] = diag
            continue
        diag["status"] = "converged" if has_trend else "converged_no_trend_single_training_year"
        counts["converged" if has_trend else "no_year_variation"] += 1

        test_site_mean = test.loc[idx, "unit_id"].map(site_mean).to_numpy(float)
        if is_binomial:
            ptest = np.clip(test_site_mean, OFFSET_EPS, 1 - OFFSET_EPS)
            offset_test = np.log(ptest / (1 - ptest))
        else:
            offset_test = np.log(test_site_mean + OFFSET_EPS)
        yc_test = np.full(len(idx), split.test_year - year_centre)
        st_test = test.loc[idx, "unit_id"].map(stratum_of).fillna(0).to_numpy(int)
        if stratify and has_trend and len(levels) > 1:
            exog_test = np.column_stack(
                [np.column_stack([(st_test == L).astype(float),
                                  (st_test == L).astype(float) * yc_test]) for L in levels])
        elif has_trend:
            exog_test = np.column_stack([np.ones(len(idx)), yc_test])
        else:
            exog_test = np.ones((len(idx), 1))
        y_pred[idx] = result.predict(exog=exog_test, offset=offset_test)
        per_species[str(sp)] = diag

    predictions = test.assign(y_pred=y_pred)
    # Keep only rows carrying a prediction, so `n_rows` counts predictions, not attempts.
    predictions = predictions.loc[np.isfinite(predictions["y_pred"].to_numpy(float))]
    diagnostics = {
        "model": ("binomial_logit_site_offset" if is_binomial else "poisson_log_site_offset")
                 + ("_stratum_trend" if stratify else "_shared_trend"),
        "covariate": ("latitude quartile — TRIM's documented stratum covariate" if stratify
                      else "none — TRIM's basic model, one trend shared across sites"),
        "site_effect": "training-period per-site mean, passed as a fixed GLM offset",
        "n_species_in_test": int(test["species"].nunique()),
        "n_species_in_train": int(train["species"].nunique()),
        **{f"n_species_{k}": v for k, v in counts.items()},
        "per_species": per_species,
    }
    return BaselineFit(name="glm_strata" if stratify else "glm", predictions=predictions,
                       diagnostics=diagnostics, wall_s=round(time.time() - t0, 3))


class _ConvLSTMCell(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(in_dim + hidden_dim, 4 * hidden_dim, kernel_size,
                              padding=kernel_size // 2)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
                ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class _SpeciesConvLSTM(nn.Module):
    """K years of a (species, H, W) field predict next year's field.

    Species are compressed to ``latent_dim`` community factors before the convolution. The
    per-timestep observed-cell mask is concatenated as an extra input channel so the
    network can distinguish a filled placeholder from a real value.
    """

    def __init__(self, n_species: int, latent_dim: int = 32, hidden_dim: int = 16,
                 kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(n_species, latent_dim)
        self.cell = _ConvLSTMCell(latent_dim + 1, hidden_dim, kernel_size)
        self.decoder = nn.Linear(hidden_dim, n_species)

    def forward(self, x: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        """x: [B,T,C,H,W] filled log1p values; obs_mask: [B,T,H,W] input validity."""
        b, t_len, _, h, w = x.shape
        z = self.encoder(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)  # [B,T,latent,H,W]
        h_t = x.new_zeros(b, self.hidden_dim, h, w)
        c_t = torch.zeros_like(h_t)
        for t in range(t_len):
            inp = torch.cat([z[:, t], obs_mask[:, t].unsqueeze(1)], dim=1)
            h_t, c_t = self.cell(inp, (h_t, c_t))
        return self.decoder(h_t.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B,C,H,W]


def _masked_log_mse(pred: torch.Tensor, target_log: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE on log1p(value), restricted to ``mask``, which has no species axis and is
    broadcast over the channel dimension."""
    diff2 = (pred - target_log) ** 2
    m = mask.unsqueeze(1).expand_as(diff2).float()
    return (diff2 * m).sum() / m.sum().clamp_min(1.0)


def load_target(task: str, artefacts_dir: str | Path | None = None) -> dict[str, Any]:
    """Read-only load of a pre-built ``target_<task>.npz`` plus its species-index sidecar."""
    d = Path(artefacts_dir) if artefacts_dir is not None else artefacts_root()
    npz = np.load(d / f"target_{task}.npz")
    meta = json.loads((d / f"target_{task}_species_index.json").read_text())
    return {"y": npz["y"], "mask": npz["mask"], "n_units": npz["n_units"], "effort": npz["effort"],
            "years": npz["years"], "species_index": meta["species_index"],
            "value_type": meta["value_type"], "task": meta["task"], "agg": meta["agg"],
            "grid": meta["grid"]}


def fit_convlstm(panel_or_target: pd.DataFrame | dict[str, Any], split: Split, seed: int, *,
                 epochs: int = 30, device: str = "cuda:0", k_history: int = 3,
                 hidden_dim: int = 16, latent_dim: int = 32, lr: float = 1e-3,
                 artefacts_dir: str | Path | None = None) -> BaselineFit:
    """ConvLSTM baseline: the previous ``k_history`` years of the gridded target field
    predict next year's field, trained on the training years and rolled out once.

    Masked cells are zero-filled only so the tensor can be batched, paired with an explicit
    mask channel; the loss never includes them, so this is an input encoding rather than
    imputation. Predictions are gathered back to unit level at each observed test row's own
    ``(cell_i, cell_j)`` — the same cells the nulls and the GLM are scored on.
    """
    t0 = time.time()
    torch.manual_seed(seed)

    if isinstance(panel_or_target, pd.DataFrame):
        panel = panel_or_target
        task = str(panel["task"].iloc[0])
        try:
            target = load_target(task, artefacts_dir)
        except FileNotFoundError:
            from ..panel import to_target
            years_needed = sorted(set(split.train_years) | {split.test_year})
            target = to_target(panel.loc[panel["year"].isin(years_needed)])
    else:
        panel = None
        target = panel_or_target
        task = str(target["task"])

    years = [int(y) for y in target["years"]]
    year_pos = {y: k for k, y in enumerate(years)}
    train_years = [y for y in split.train_years if y in year_pos]
    missing_train = sorted(set(split.train_years) - set(train_years))
    if split.test_year not in year_pos:
        raise ValueError(f"target for task {task} has no year {split.test_year}")
    if len(train_years) <= k_history:
        raise ValueError(f"only {len(train_years)} usable training years, need > k_history={k_history}")

    species_index = list(target["species_index"])
    n_species = len(species_index)
    y_log = np.log1p(np.clip(target["y"], 0, None))
    mask_all = target["mask"]

    t_idx = [year_pos[y] for y in train_years]
    y_seq = np.nan_to_num(y_log[:, t_idx], nan=0.0)      # [C, Ttr, H, W]
    m_seq = mask_all[t_idx]                              # [Ttr, H, W]

    n_samples = len(train_years) - k_history
    windows = np.stack([y_seq[:, s:s + k_history] for s in range(n_samples)], axis=0)   # [N,C,K,H,W]
    x_train = np.transpose(windows, (0, 2, 1, 3, 4))                                    # [N,K,C,H,W]
    xm_train = np.stack([m_seq[s:s + k_history] for s in range(n_samples)], axis=0)     # [N,K,H,W]
    y_train = np.stack([y_seq[:, s + k_history] for s in range(n_samples)], axis=0)     # [N,C,H,W]
    ym_train = np.stack([m_seq[s + k_history] for s in range(n_samples)], axis=0)       # [N,H,W]

    dev = torch.device(device)
    xt = torch.from_numpy(x_train).float().to(dev)
    xmt = torch.from_numpy(xm_train).float().to(dev)
    yt = torch.from_numpy(y_train).float().to(dev)
    ymt = torch.from_numpy(ym_train).bool().to(dev)

    model = _SpeciesConvLSTM(n_species, latent_dim=latent_dim, hidden_dim=hidden_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loss_curve: list[float] = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = _masked_log_mse(model(xt, xmt), yt, ymt)
        loss.backward()
        opt.step()
        loss_curve.append(float(loss.item()))

    last_idx = [year_pos[y] for y in train_years[-k_history:]]
    x_last = np.transpose(np.nan_to_num(y_log[:, last_idx], nan=0.0)[None], (0, 2, 1, 3, 4))
    m_last = mask_all[last_idx][None]
    model.eval()
    with torch.no_grad():
        pred_field = model(torch.from_numpy(x_last).float().to(dev),
                           torch.from_numpy(m_last).float().to(dev))
    pred_field = np.clip(np.expm1(pred_field.squeeze(0).cpu().numpy()), 0.0, None)
    if str(target.get("value_type", "")) == "prevalence":
        pred_field = np.clip(pred_field, 0.0, 1.0)

    if panel is None:
        raise ValueError("fit_convlstm needs the panel (not only the target) to gather unit-level rows")
    test = test_frame(panel, split)
    sp_pos = {s: k for k, s in enumerate(species_index)}
    sp_idx = test["species"].map(sp_pos)
    known = sp_idx.notna().to_numpy()
    y_pred = np.full(len(test), np.nan, dtype=float)
    if known.any():
        si = sp_idx[known].to_numpy(dtype=int)
        ci = test.loc[known, "cell_i"].to_numpy(dtype=int)
        cj = test.loc[known, "cell_j"].to_numpy(dtype=int)
        y_pred[known] = pred_field[si, ci, cj]
    predictions = test.assign(y_pred=y_pred)
    predictions = predictions.loc[np.isfinite(predictions["y_pred"].to_numpy(float))]

    diagnostics = {
        "model": "convlstm_species_latent_log1p",
        "k_history": k_history, "epochs": epochs, "seed": seed, "device": str(dev),
        "hidden_dim": hidden_dim, "latent_dim": latent_dim, "lr": lr,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "n_train_samples": int(n_samples), "train_years_used": train_years,
        "missing_train_years": missing_train, "loss_curve": loss_curve,
        "final_train_loss": loss_curve[-1] if loss_curve else float("nan"),
        "n_species_not_in_target": int((~known).sum()),
        "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated(dev)) if dev.type == "cuda" else 0,
    }
    return BaselineFit(name="convlstm", predictions=predictions, diagnostics=diagnostics,
                       wall_s=round(time.time() - t0, 3))


RF_MIN_TRAIN_ROWS = MIN_TRAIN_ROWS

_RF_DATASET_CACHE: dict[str, Any] = {}
_RF_GRID_CACHE: dict[tuple[str, int], tuple[np.ndarray, list[str]]] = {}


def _rf_dataset(biocube_dir: str | Path) -> Any:
    """One ``LargeClimateDataset`` per ``biocube_dir``, built once per process.

    Scaling is forced off: a random forest splits on raw thresholds and is invariant to
    monotone rescaling, so this costs the baseline nothing.
    """
    key = str(biocube_dir)
    if key not in _RF_DATASET_CACHE:
        cfg = load_config()
        cfg.data.scaling.enabled = False
        _RF_DATASET_CACHE[key] = batch.make_dataset(cfg, biocube_dir=biocube_dir)
    return _RF_DATASET_CACHE[key]


def _environment_features(rows: pd.DataFrame, task: str, dataset: Any,
                          biocube_dir: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Abiotic covariate and lat/lon lookup, one row per distinct ``(unit_id, year)``.

    Covariates come from ``batch.cell_covariates`` with the default ``ABIOTIC_GROUPS``,
    which excludes the species channels. Each ``(task, year)`` BioCube pair is reduced to a
    covariate grid at most once per process (``_RF_GRID_CACHE``).
    """
    uy = rows.drop_duplicates(["unit_id", "year"])[["unit_id", "year", "cell_i", "cell_j", "lat", "lon"]]
    ci_full = np.repeat(np.arange(batch.GRID.H), batch.GRID.W)
    cj_full = np.tile(np.arange(batch.GRID.W), batch.GRID.H)
    names: list[str] | None = None
    feats, index = [], []
    for year, g in uy.groupby("year"):
        key = (task, int(year))
        if key not in _RF_GRID_CACHE:
            info = batch.forecast_input(task, int(year), biocube_dir)
            x = batch.load_input(info["path"], dataset)
            _RF_GRID_CACHE[key] = batch.cell_covariates(x, ci_full, cj_full)
        mat, nm = _RF_GRID_CACHE[key]
        if names is None:
            names = nm
        elif names != nm:
            raise ValueError(f"covariate name/order changed for task {task} year {year}")
        flat = g["cell_i"].to_numpy(int) * batch.GRID.W + g["cell_j"].to_numpy(int)
        feats.append(np.column_stack([mat[flat], g["lat"].to_numpy(float), g["lon"].to_numpy(float)]))
        index.extend(zip(g["unit_id"], g["year"]))
    feature_names = (names or []) + ["lat", "lon"]
    lookup = pd.DataFrame(np.concatenate(feats, axis=0),
                          index=pd.MultiIndex.from_tuples(index, names=["unit_id", "year"]),
                          columns=feature_names)
    return lookup, feature_names


def fit_randomforest(panel: pd.DataFrame, split: Split, seed: int = 0, *,
                     n_estimators: int = 300, n_jobs: int = -1,
                     biocube_dir: str | Path = batch.BIOCUBE_DIR) -> BaselineFit:
    """Per-species ``RandomForestRegressor`` environmental SDM baseline.

    Features are the gridded abiotic covariates plus the unit's lat/lon, read from the same
    short-lead BioCube pair the fine-tuned model is scored against. ``year`` is excluded:
    on a chronological split a tree can split on a literal year value like a row index.
    Forests are per species because a joint fit needs every output species on every
    training row, which Task C does not satisfy. Unlike ``fit_glm``, ``seed`` changes the
    fit.
    """
    from sklearn.ensemble import RandomForestRegressor

    t0 = time.time()
    task = str(panel["task"].iloc[0])
    value_type = str(panel["value_type"].iloc[0])

    test = test_frame(panel, split)
    train = _train_rows(panel, split).reset_index(drop=True)

    dataset = _rf_dataset(biocube_dir)
    env_train, feature_names = _environment_features(train, task, dataset, biocube_dir)
    env_test, _ = _environment_features(test, task, dataset, biocube_dir)

    X_train_all = env_train.reindex(
        pd.MultiIndex.from_arrays([train["unit_id"], train["year"]])).to_numpy(float)
    X_test_all = env_test.reindex(
        pd.MultiIndex.from_arrays([test["unit_id"], test["year"]])).to_numpy(float)
    if np.isnan(X_train_all).any() or np.isnan(X_test_all).any():
        raise ValueError("missing covariate lookup for an observed (unit_id, year) row — "
                         "a BioCube month-pair this split needs was silently unavailable")

    y_pred = np.full(len(test), np.nan, dtype=float)
    per_species: dict[str, dict[str, Any]] = {}
    counts = {"fitted": 0, "insufficient_data": 0, "failed": 0}

    test_idx_by_species = test.groupby("species", observed=True).indices
    train_by_species = {sp: g for sp, g in train.groupby("species", observed=True)}

    for sp, idx in test_idx_by_species.items():
        train_g = train_by_species.get(sp)
        n_rows = 0 if train_g is None else int(len(train_g))
        if train_g is None or n_rows < RF_MIN_TRAIN_ROWS:
            per_species[str(sp)] = {"status": "insufficient_data", "n_train_rows": n_rows,
                                    "n_test_rows": int(len(idx))}
            counts["insufficient_data"] += 1
            continue

        X_tr = X_train_all[train_g.index.to_numpy()]
        y_tr = train_g["value"].to_numpy(float)
        X_te = X_test_all[idx]
        try:
            rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)
            rf.fit(X_tr, y_tr)
            pred = rf.predict(X_te)
        except ValueError as exc:
            per_species[str(sp)] = {"status": f"failed: {type(exc).__name__}: {exc}",
                                    "n_train_rows": n_rows, "n_test_rows": int(len(idx))}
            counts["failed"] += 1
            continue

        if value_type == "prevalence":
            pred = np.clip(pred, 0.0, 1.0)
        elif value_type == "count":
            pred = np.clip(pred, 0.0, None)
        y_pred[idx] = pred
        per_species[str(sp)] = {"status": "fitted", "n_train_rows": n_rows,
                                "n_train_sites": int(train_g["unit_id"].nunique()),
                                "n_test_rows": int(len(idx))}
        counts["fitted"] += 1

    predictions = test.assign(y_pred=y_pred)
    predictions = predictions.loc[np.isfinite(predictions["y_pred"].to_numpy(float))]
    diagnostics = {
        "model": "randomforest_per_species_env_latlon",
        "n_features": len(feature_names), "feature_names": feature_names,
        "n_estimators": n_estimators, "seed": seed,
        "n_species_in_test": int(test["species"].nunique()),
        "n_species_in_train": int(train["species"].nunique()),
        **{f"n_species_{k}": v for k, v in counts.items()},
        "per_species": per_species,
    }
    return BaselineFit(name="randomforest", predictions=predictions, diagnostics=diagnostics,
                       wall_s=round(time.time() - t0, 3))


def _score_predictions(pred: pd.DataFrame, value_type: str, prev: pd.DataFrame,
                       refs: dict[str, np.ndarray]) -> dict[str, Any]:
    """Mirrors ``nulls._score_set`` exactly, so a baseline's numbers sit in the same shape
    as a null's and are directly comparable."""
    frame = pred.loc[:, ["unit_id", "species", "year", "y_true", "y_pred"]]
    frame = frame.loc[np.isfinite(frame["y_pred"].to_numpy(float))]
    if frame.empty:
        return {"n": 0, "note": "no predictions available"}
    rec = metrics.evaluate(frame, value_type=value_type, previous=prev)
    rec["coverage"] = float(len(frame) / len(pred))
    rec["skill"] = {}
    for ref_name, ref_vals in refs.items():
        sel = ref_vals[frame.index]
        ok = np.isfinite(sel)
        # Raw-scale skill on a heavy-tailed target is decided by a handful of rows; the
        # log1p variant is reported beside it, never merged.
        lg = lambda a: np.log1p(np.clip(np.asarray(a, float), 0.0, None))
        rec["skill"][f"vs_{ref_name}"] = {
            "skill_score": metrics.skill_score(frame["y_true"], frame["y_pred"], sel),
            "skill_score_log1p": metrics.skill_score(
                lg(frame["y_true"]), lg(frame["y_pred"]), lg(sel)),
            "n_scored": int(ok.sum()),
            "rmse_on_reference_rows": metrics.rmse(
                frame["y_true"].to_numpy(float)[ok], frame["y_pred"].to_numpy(float)[ok]),
        }
    # The headline is skill against the *hardest* reference; the minimum can only lower a
    # score, so it cannot be gamed upward.
    usable = [v["skill_score"] for v in rec["skill"].values()
              if isinstance(v.get("skill_score"), float) and np.isfinite(v["skill_score"])]
    if usable:
        best_ref = min(rec["skill"], key=lambda k: rec["skill"][k]["skill_score"]
                       if np.isfinite(rec["skill"][k]["skill_score"]) else np.inf)
        rec["skill_vs_strongest_null"] = {"reference": best_ref, "skill_score": min(usable)}
        from ..common.invariants import headline_is_worst_reference

        headline_is_worst_reference(rec)
    return rec


def score_baseline(panel: pd.DataFrame, split: Split, fit: BaselineFit, *, seed: int = 0) -> dict[str, Any]:
    """Score one baseline fit against the same three references ``nulls.score_nulls``
    reports skill against, by reusing ``nulls.compute_nulls``."""
    value_type = str(panel["value_type"].iloc[0])
    null_preds = compute_nulls(panel, split, seed=seed)
    refs = {name: null_preds[name].to_numpy(float) for name in REFERENCES}
    prev = _previous_year_frame(panel, split)
    return _score_predictions(fit.predictions, value_type, prev, refs)


def _summarise_seeds(per_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Mean ± sd of the headline numbers across seeds, per test year."""
    by_year: dict[int, dict[str, list[float]]] = {}
    for rec in per_seed.values():
        for entry in rec.get("splits", []):
            ty = int(entry["split"]["test_year"])
            m = entry["metrics"]
            bucket = by_year.setdefault(ty, {"rmse": [], "rmse_log": [], "skill_vs_persistence": []})
            bucket["rmse"].append(m.get("rmse", float("nan")))
            bucket["rmse_log"].append(m.get("rmse_log", float("nan")))
            bucket["skill_vs_persistence"].append(
                m.get("skill", {}).get("vs_persistence", {}).get("skill_score", float("nan")))
    out: dict[str, Any] = {}
    for ty, bucket in by_year.items():
        out[str(ty)] = {}
        for k, vals in bucket.items():
            arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
            out[str(ty)][k] = {"mean": float(arr.mean()) if arr.size else float("nan"),
                               "sd": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
                               "n_seeds": int(arr.size)}
    return out


GAM_SPLINE_DF = (6, 6, 4)      # lat, lon, year
GAM_N_COMPONENTS = 6           # PCA components retained from the abiotic covariates


def _nbgam_one(endog: np.ndarray, exog_tr: np.ndarray, sm_tr: np.ndarray, exog_te: np.ndarray,
               sm_te: np.ndarray, df: list[int], is_binomial: bool,
               weights: np.ndarray | None, offset_tr: np.ndarray | None = None,
               offset_te: np.ndarray | None = None) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Fit one species' GAM. Module level and array-only so it can be sent to a worker."""
    import statsmodels.api as sm
    from statsmodels.gam.api import BSplines, GLMGam

    # Knots come from the training rows; the year-forward test point is clipped to the
    # training range so the year smooth contributes its last fitted value rather than a
    # cubic run off the end.
    lo, hi = sm_tr.min(axis=0), sm_tr.max(axis=0)
    sm_te = np.clip(sm_te, lo, hi)
    try:
        bs = BSplines(sm_tr, df=df, degree=[3] * len(df), include_intercept=False)
        if is_binomial:
            family, kw = sm.families.Binomial(), {"var_weights": weights}
            alpha = None
        else:
            m_, v_ = float(endog.mean()), float(endog.var())
            alpha = float(np.clip((v_ - m_) / max(m_ ** 2, 1e-9), 1e-6, 10.0))
            family, kw = sm.families.NegativeBinomial(alpha=alpha), {}
        if offset_tr is not None:
            kw["offset"] = offset_tr
        res = GLMGam(endog, exog=exog_tr, smoother=bs, family=family,
                     alpha=[1.0] * len(df), **kw).fit()
        # Build the prediction design by hand: `res.predict(exog=..., exog_smooth=...)` is
        # broken in statsmodels 0.14.x; `bs.transform` re-evaluates the fitted knots.
        design = np.column_stack([exog_te, bs.transform(sm_te)])
        if design.shape[1] != res.params.size:
            raise ValueError(f"design has {design.shape[1]} columns against "
                             f"{res.params.size} fitted parameters")
        eta = design @ res.params
        if offset_te is not None:
            eta = eta + offset_te
        # A log link turns linear extrapolation into exponential nonsense: hold the linear
        # predictor inside the range the fit actually saw.
        eta_tr = np.column_stack([exog_tr, bs.basis]) @ res.params
        if offset_tr is not None:
            eta_tr = eta_tr + offset_tr
        eta = np.clip(eta, eta_tr.min(), eta_tr.max())
        pred = np.asarray(res.model.family.link.inverse(eta), float)
        # And no prediction beyond the response this species was ever observed at.
        pred = np.clip(pred, 0.0, float(endog.max()))
    except Exception as exc:                           # statsmodels raises a wide variety here
        return None, {"status": f"failed: {type(exc).__name__}: {exc}"[:200]}
    return pred, {"status": "converged", "alpha": alpha}


def fit_nbgam(panel: pd.DataFrame, split: Split, seed: int = 0, *,
              biocube_dir: str | Path = batch.BIOCUBE_DIR, n_jobs: int = 32) -> BaselineFit:
    """Per-species negative-binomial GAM: ``site offset + s(lat) + s(lon) + s(year) + environment``.

    The baseline the monitoring literature reaches for when the question is prediction:
    a spatial smooth plus environmental covariates breaks ``fit_glm``'s degeneracy with
    climatology while staying classical. The site term enters as a log site-mean offset,
    the tractable form of ``mgcv``'s ``s(site, bs="re")``. Negative binomial because these
    targets are strongly overdispersed; dispersion is estimated per species by moments.
    Abiotic covariates are reduced to ``GAM_N_COMPONENTS`` principal components fitted on
    training rows only.
    """
    t0 = time.time()
    test = test_frame(panel, split)
    train = _train_rows(panel, split)
    task = str(panel["task"].iloc[0])
    value_type = str(panel["value_type"].iloc[0])
    is_binomial = value_type == "prevalence"

    dataset = _rf_dataset(biocube_dir)
    env_train, names = _environment_features(train, task, dataset, biocube_dir)
    env_test, _ = _environment_features(test, task, dataset, biocube_dir)
    keep = [n for n in names if n not in ("lat", "lon")]

    mu = env_train[keep].mean()
    sd = env_train[keep].std().replace(0.0, 1.0)
    Ztr = ((env_train[keep] - mu) / sd).to_numpy(float)
    _, S, Vt = np.linalg.svd(np.nan_to_num(Ztr), full_matrices=False)
    comps = Vt[:GAM_N_COMPONENTS].T
    proj = lambda df: np.nan_to_num(((df[keep] - mu) / sd).to_numpy(float)) @ comps

    y_pred = np.full(len(test), np.nan, dtype=float)
    per_species: dict[str, dict[str, Any]] = {}
    counts = {"converged": 0, "insufficient_data": 0, "failed": 0}
    test_idx_by_species = test.groupby("species", observed=True).indices
    train_by_species = {sp: g for sp, g in train.groupby("species", observed=True)}

    # Build every species' arrays first, then fit them in parallel with one thread per fit:
    # each IRLS problem is tiny, so BLAS-level parallelism wastes the cores.
    work, order = [], []
    for sp, idx in test_idx_by_species.items():
        g = train_by_species.get(sp)
        n_rows = 0 if g is None else int(len(g))
        if g is None or n_rows < max(MIN_TRAIN_ROWS, 3 * sum(GAM_SPLINE_DF)):
            per_species[str(sp)] = {"status": "insufficient_data", "n_train_rows": n_rows}
            counts["insufficient_data"] += 1
            continue
        gi = env_train.loc[list(zip(g["unit_id"], g["year"]))]
        ti = env_test.loc[list(zip(test.iloc[idx]["unit_id"], test.iloc[idx]["year"]))]
        sm_tr = np.column_stack([gi["lat"].to_numpy(float), gi["lon"].to_numpy(float),
                                 g["year"].to_numpy(float)])
        sm_te = np.column_stack([ti["lat"].to_numpy(float), ti["lon"].to_numpy(float),
                                 test.iloc[idx]["year"].to_numpy(float)])
        var_ok = [k for k in range(3) if np.ptp(sm_tr[:, k]) > 0]
        if len(var_ok) < 2:
            per_species[str(sp)] = {"status": "insufficient_data", "n_train_rows": n_rows,
                                    "reason": "no spatial variation in training rows"}
            counts["insufficient_data"] += 1
            continue
        # Site term: without one this is a pure SDM with no way to tell a poor site from a
        # rich neighbour.
        site_mean = g.groupby("unit_id", observed=True)["value"].mean()
        if is_binomial:
            ptr = np.clip(g["unit_id"].map(site_mean).to_numpy(float), OFFSET_EPS, 1 - OFFSET_EPS)
            pte = np.clip(test.iloc[idx]["unit_id"].map(site_mean).to_numpy(float),
                          OFFSET_EPS, 1 - OFFSET_EPS)
            off_tr, off_te = np.log(ptr / (1 - ptr)), np.log(pte / (1 - pte))
        else:
            off_tr = np.log(g["unit_id"].map(site_mean).to_numpy(float) + OFFSET_EPS)
            off_te = np.log(test.iloc[idx]["unit_id"].map(site_mean)
                            .fillna(float(site_mean.mean())).to_numpy(float) + OFFSET_EPS)
        work.append((g["value"].to_numpy(float),
                     np.column_stack([np.ones(n_rows), proj(gi)]),
                     sm_tr[:, var_ok],
                     np.column_stack([np.ones(len(idx)), proj(ti)]),
                     sm_te[:, var_ok],
                     [GAM_SPLINE_DF[k] for k in var_ok], is_binomial,
                     g["effort"].to_numpy(float) if is_binomial else None,
                     off_tr, off_te))
        order.append((sp, idx, n_rows, var_ok))

    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs)(delayed(_nbgam_one)(*w) for w in work)

    for (sp, idx, n_rows, var_ok), (pred, info) in zip(order, results):
        rec = {"n_train_rows": n_rows, "n_test_rows": int(len(idx)), "smooth_axes": var_ok, **info}
        per_species[str(sp)] = rec
        if pred is None:
            counts["failed"] += 1
            continue
        y_pred[idx] = np.clip(pred, 0.0, 1.0) if is_binomial else np.clip(pred, 0.0, None)
        counts["converged"] += 1

    predictions = test.assign(y_pred=y_pred)
    predictions = predictions.loc[np.isfinite(predictions["y_pred"].to_numpy(float))]
    # A baseline that predicted nothing is a failure, not a result.
    if predictions.empty:
        raise RuntimeError(
            f"nbgam produced no predictions for task {task}: {counts}; first failure: "
            + next((v["status"] for v in per_species.values()
                    if str(v.get("status", "")).startswith("failed")), "none recorded"))
    return BaselineFit(
        name="nbgam", predictions=predictions,
        diagnostics={"model": "nbgam_per_species_site_offset_spatial_smooth_env_pca",
                     "site_term": "log site mean as offset — the tractable form of mgcv s(site, bs='re')",
                     "family": "Binomial/logit" if is_binomial else "NegativeBinomial/log",
                     "spline_df": {"lat_lon_year": list(GAM_SPLINE_DF)},
                     "n_components": GAM_N_COMPONENTS, "n_covariates_in": len(keep),
                     "explained_variance_ratio": float((S[:GAM_N_COMPONENTS] ** 2).sum()
                                                       / max((S ** 2).sum(), 1e-12)),
                     "species": counts, "per_species": per_species, "seed": seed,
                     "note": "spatial smooth + environment, so unlike fit_glm its ranking is "
                             "not climatology's by construction"},
        wall_s=round(time.time() - t0, 1))


def fit_glm_strata(panel: pd.DataFrame, split: Split, seed: int = 0) -> BaselineFit:
    """TRIM's covariate model: year effects estimated per regional stratum.

    Shipped beside the basic model because the comparison is the point: the basic model's
    ranking equals climatology's exactly, and the covariate version breaks that only by
    moving whole strata.
    """
    return fit_glm(panel, split, seed, stratify=True)


FIT_FUNCTIONS: dict[str, Callable[..., BaselineFit]] = {
    "glm": fit_glm, "glm_strata": fit_glm_strata, "nbgam": fit_nbgam,
    "convlstm": fit_convlstm, "randomforest": fit_randomforest}
BASELINE_NAMES = tuple(FIT_FUNCTIONS)


def run_baseline(name: str, panel: pd.DataFrame, splits: Sequence[Split], seeds: Sequence[int],
                 out_dir: str | Path, runner: Runner | None = None, **fit_kwargs: Any) -> dict[str, Any]:
    """Run one baseline over every (split, seed), score it against the null references, and
    write one JSON per seed: ``<out_dir>/<task>_<name>_s<seed>.json``."""
    if name not in FIT_FUNCTIONS:
        raise ValueError(f"unknown baseline {name!r}; choices are {sorted(FIT_FUNCTIONS)}")
    fit_fn = FIT_FUNCTIONS[name]
    task = str(panel["task"].iloc[0])
    value_type = str(panel["value_type"].iloc[0])
    out_dir = Path(out_dir)
    own_runner = runner is None
    runner = runner if runner is not None else Runner(phase=f"baseline_{name}_{task}")

    per_seed: dict[int, dict[str, Any]] = {}
    try:
        for seed in seeds:
            out_path = out_dir / f"{task}_{name}_s{seed}.json"

            def _run(seed: int = seed, out_path: Path = out_path) -> dict[str, Any]:
                from ..common.resources import Meter

                per_split = []
                for split in splits:
                    # Baselines are metered too; ConvLSTM gets the same per-device power
                    # integration the foundation models get.
                    dev = fit_kwargs.get("device")
                    meter = Meter(
                        device=int(str(dev).split(":")[-1]) if dev and ":" in str(dev) else None,
                        label=f"{name}_{task}_s{seed}_{split.test_year}")
                    with meter:
                        fit = fit_fn(panel, split, seed=seed, **fit_kwargs)
                    fit.diagnostics["resources"] = meter.report()
                    score = score_baseline(panel, split, fit, seed=seed)
                    per_split.append({
                        "split": split.as_dict(), "seed": seed,
                        "n_rows": int(len(fit.predictions)),
                        "n_rows_scored": int(score.get("n", 0)),
                        "n_units": int(fit.predictions["unit_id"].nunique()),
                        "n_species": int(fit.predictions["species"].nunique()),
                        "wall_s": fit.wall_s,
                        "diagnostics": fit.diagnostics,
                        "metrics": score,
                    })
                record = {"baseline": name, "task": task, "value_type": value_type, "splits": per_split}
                write_json(out_path, record)
                return record

            rec = runner.run_step(f"base_{task}_{name}_s{seed}", _run, outputs=[out_path],
                                  meta={"baseline": name, "task": task, "seed": seed})
            per_seed[seed] = rec.get("result") or {}
    finally:
        if own_runner:
            runner.close()

    return {"task": task, "baseline": name, "seeds": list(seeds), "per_seed": per_seed,
            "summary": _summarise_seeds(per_seed)}
