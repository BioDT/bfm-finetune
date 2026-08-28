"""L2 — frozen-representation probe: a small MLP over decoded fields, per grid cell.
Reads the model's decoded output fields rather than its encoder latents
"""

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .baselines import BaselineFit, score_baseline
from .nulls import Split, observed, test_frame

# Head shape is pre-registered, not tuned: a width x depth factorial found 256x3 the
# smallest configuration stable on every task, and validation must not select it.
HIDDEN = 256
LAYERS = 3
EPOCHS = 300
LR = 3e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 40


class ProbeHead(nn.Module):
    """Features at a cell -> one value per species.

    Shared by L2 and L3 so both rungs and both backbones stay architecturally identical;
    ``scripts/audit.py`` asserts that sharing.
    """

    def __init__(self, n_features: int, n_species: int, hidden: int = HIDDEN,
                 layers: int = LAYERS):
        super().__init__()
        if layers < 1:
            raise ValueError(f"a head needs at least one hidden block, got layers={layers}")
        mods: list[nn.Module] = [nn.LayerNorm(n_features)]
        width = n_features
        for _ in range(layers):
            mods += [nn.Linear(width, hidden), nn.GELU()]
            width = hidden
        mods.append(nn.Linear(width, n_species))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def flatten_fields(decoded: dict[str, Any], grid_hw: tuple[int, int] = (160, 280)
                   ) -> dict[str, np.ndarray]:
    """``{group: {var: tensor}}`` -> ``{"group.var[.level]": [H, W]}``.

    Atmospheric variables are split into one field per pressure level, so every feature is
    a plain 2-D field and the column order is reproducible.
    """
    out: dict[str, np.ndarray] = {}
    H, W = grid_hw
    for group, variables in decoded.items():
        if not isinstance(variables, dict):
            continue
        for var, tensor in variables.items():
            arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
            arr = arr.reshape(-1, *arr.shape[-2:]) if arr.ndim > 2 else arr[None]
            if arr.shape[-2:] != (H, W):
                continue
            if arr.shape[0] == 1:
                out[f"{group}.{var}"] = arr[0].astype(np.float32)
            else:
                for level in range(arr.shape[0]):
                    out[f"{group}.{var}.{level}"] = arr[level].astype(np.float32)
    return out


def decoded_cell_features(fields: dict[str, np.ndarray], cell_i: Sequence[int],
                          cell_j: Sequence[int]) -> tuple[np.ndarray, list[str]]:
    """Sample every decoded field at the given cells. Names are sorted so order is stable."""
    names = sorted(fields)
    ci = np.asarray(cell_i, dtype=int)
    cj = np.asarray(cell_j, dtype=int)
    cols = []
    for name in names:
        arr = np.asarray(fields[name])
        if arr.ndim != 2:
            raise ValueError(f"field {name!r} must be [H, W], got {arr.shape}")
        cols.append(arr[ci, cj])
    return np.stack(cols, axis=1).astype(np.float32), names


def _design(panel: pd.DataFrame, years: Sequence[int], features_by_year: dict[int, np.ndarray],
            cells_by_year: dict[int, pd.DataFrame], species: list[str], use_prev: bool = False
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build X [n_cellyears, n_features], Y and a mask, both [n_cellyears, n_species].

    Species absent from a cell-year are masked out rather than filled: the panel's
    ``observed`` flag is the only thing that licenses a number.
    """
    obs = observed(panel)
    s_index = {s: k for k, s in enumerate(species)}
    X, Y, M = [], [], []
    for year in years:
        if year not in features_by_year:
            continue
        cells = cells_by_year[year]
        pos = {u: k for k, u in enumerate(cells["unit_id"].tolist())}
        y = np.zeros((len(cells), len(species)), dtype=np.float32)
        m = np.zeros((len(cells), len(species)), dtype=bool)
        sub = obs.loc[obs["year"] == year]
        for unit, sp, val in zip(sub["unit_id"], sub["species"], sub["value"]):
            r, c = pos.get(unit), s_index.get(sp)
            if r is not None and c is not None:
                y[r, c] = val
                m[r, c] = True
        feats = features_by_year[year]
        if use_prev:
            feats = np.concatenate([feats, prev_year_matrix(panel, year, cells, species)], axis=1)
        X.append(feats)
        Y.append(y)
        M.append(m)
    if not X:
        raise ValueError("no years with both features and observations")
    return np.concatenate(X), np.concatenate(Y), np.concatenate(M)


LOSSES = ("log1p_mse", "poisson")


def _masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 mode: str = "log1p_mse") -> torch.Tensor:
    """Training loss over observed entries only.

    ``log1p_mse`` fits the mean of log(1+y), so exponentiating back gives a conditional
    median — a systematic underestimate on a skewed target. ``poisson`` emits a log rate
    and fits E[y] on the raw scale directly; as a quasi-likelihood it is valid for the
    non-integer positive targets here.
    """
    if mode not in LOSSES:
        raise ValueError(f"unknown loss {mode!r}; expected one of {LOSSES}")
    if not mask.any():
        return pred.sum() * 0.0
    if mode == "poisson":
        rate = pred.clamp(-20.0, 20.0)
        d = torch.exp(rate) - target.clamp(min=0) * rate
    else:
        d = (torch.log1p(pred.clamp(min=0)) - torch.log1p(target.clamp(min=0))) ** 2
    return d[mask].mean()


def _to_prediction(raw: torch.Tensor, mode: str, upper: float | None = None) -> torch.Tensor:
    """Map the head's output to the predicted value on the target's own scale."""
    out = torch.exp(raw.clamp(-20.0, 20.0)) if mode == "poisson" else raw.clamp(min=0)
    return out if upper is None else out.clamp(max=upper)


def value_domain_upper(value_type: str) -> float | None:
    """The target's upper bound, where it has one. Proportions are capped at 1."""
    return 1.0 if str(value_type) == "prevalence" else None


def prev_year_matrix(panel: pd.DataFrame, year: int, cells: pd.DataFrame,
                     species: Sequence[str]) -> np.ndarray:
    """``log1p`` of each (cell, species) value in ``year - 1``, zero where unobserved.

    Exactly the information the persistence null uses, and ``year - 1`` is always a
    training year, so this leaks nothing persistence does not already have.
    """
    obs = observed(panel)
    sub = obs.loc[obs["year"] == year - 1]
    pos = {u: k for k, u in enumerate(cells["unit_id"].tolist())}
    s_index = {s: k for k, s in enumerate(species)}
    out = np.zeros((len(cells), len(species)), dtype=np.float32)
    for unit, sp, val in zip(sub["unit_id"], sub["species"], sub["value"]):
        r, c = pos.get(unit), s_index.get(sp)
        if r is not None and c is not None and np.isfinite(val):
            out[r, c] = np.log1p(max(float(val), 0.0))
    return out


def fit_probe(panel: pd.DataFrame, split: Split, seed: int,
              features_by_year: dict[int, np.ndarray], cells_by_year: dict[int, pd.DataFrame],
              *, feature_names: Sequence[str] | None = None, device: str = "cuda:1",
              epochs: int = EPOCHS, hidden: int = HIDDEN, val_year: int | None = None,
              backbone: str = "bfm", save_dir=None, loss: str = "log1p_mse",
              use_prev: bool = False) -> BaselineFit:
    """Fit the L2 head on training years and predict the test year.

    ``val_year`` holds out one training year for early stopping; it must not be the test
    year.
    """
    from ..common.invariants import no_test_year_in_training, predictions_in_domain
    from ..common.resources import Meter

    no_test_year_in_training(split.train_years, split.test_year)
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    meter = Meter(device=int(device.split(":")[-1]) if ":" in str(device) else None,
                  label=f"L2_{backbone}_seed{seed}")
    meter.__enter__()

    species = sorted(observed(panel)["species"].unique().tolist())
    train_years = [y for y in split.train_years if y in features_by_year]
    if val_year is not None:
        if val_year == split.test_year:
            raise ValueError("validation year must not be the test year")
        train_years = [y for y in train_years if y != val_year]

    Xtr, Ytr, Mtr = _design(panel, train_years, features_by_year, cells_by_year, species, use_prev)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    head = ProbeHead(Xtr.shape[1], len(species), hidden).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    xt = torch.from_numpy((Xtr - mu) / sd).to(dev)
    yt = torch.from_numpy(Ytr).to(dev)
    mt = torch.from_numpy(Mtr).to(dev)

    xv = yv = mv = None
    if val_year is not None and val_year in features_by_year:
        Xv, Yv, Mv = _design(panel, [val_year], features_by_year, cells_by_year, species, use_prev)
        xv = torch.from_numpy((Xv - mu) / sd).to(dev)
        yv = torch.from_numpy(Yv).to(dev)
        mv = torch.from_numpy(Mv).to(dev)

    losses, best, best_state, stale = [], float("inf"), None, 0
    for epoch in range(epochs):
        head.train()
        opt.zero_grad()
        step_loss = _masked_loss(head(xt), yt, mt, loss)
        step_loss.backward()
        opt.step()
        losses.append(float(step_loss))
        if xv is not None:
            head.eval()
            with torch.no_grad():
                v = float(_masked_loss(head(xv), yv, mv, loss))
            if v < best - 1e-6:
                best, stale = v, 0
                best_state = {k: t.detach().clone() for k, t in head.state_dict().items()}
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
    if best_state is not None:
        head.load_state_dict(best_state)

    test = test_frame(panel, split)
    cells = cells_by_year[split.test_year]
    pos = {u: k for k, u in enumerate(cells["unit_id"].tolist())}
    s_index = {s: k for k, s in enumerate(species)}
    head.eval()
    with torch.no_grad():
        xq_raw = features_by_year[split.test_year]
        if use_prev:
            xq_raw = np.concatenate(
                [xq_raw, prev_year_matrix(panel, split.test_year, cells, species)], axis=1)
        xq = torch.from_numpy((xq_raw - mu) / sd).to(dev)
        grid = _to_prediction(head(xq), loss,
                              value_domain_upper(panel["value_type"].iloc[0])).cpu().numpy()

    pred = np.full(len(test), np.nan, dtype=float)
    for k, (unit, sp) in enumerate(zip(test["unit_id"], test["species"])):
        r, c = pos.get(unit), s_index.get(sp)
        if r is not None and c is not None:
            pred[k] = grid[r, c]
    frame = test.assign(y_pred=pred)
    frame = frame.loc[np.isfinite(frame["y_pred"].to_numpy(float))]
    predictions_in_domain(frame["y_pred"], str(panel["value_type"].iloc[0]),
                          where=f"L2 {backbone} task {panel['task'].iloc[0]}")

    meter.__exit__(None, None, None)
    saved = None
    if save_dir:
        from ..common.runner import atomic_path, write_json

        out = Path(save_dir) / f"{backbone}_l2_s{seed}_{split.test_year}"
        out.mkdir(parents=True, exist_ok=True)
        with atomic_path(out / "predictions.parquet", suffix=".parquet") as tmp:
            frame.to_parquet(tmp, index=False, compression="zstd")
        with atomic_path(out / "state.pt", suffix=".pt") as tmp:
            torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                        "mu": mu, "sd": sd, "species": species, "seed": seed,
                        "split": split.as_dict(), "backbone": backbone}, tmp)
        write_json(out / "run.json", {"backbone": backbone, "seed": seed,
                                      "split": split.as_dict(), "n_predictions": int(len(frame))})
        saved = {"dir": str(out),
                 "bytes": sum(f.stat().st_size for f in out.iterdir() if f.is_file())}
    return BaselineFit(
        name=f"{backbone}_l2", predictions=frame,
        # Derived from the built module so it cannot drift from the actual shape.
        diagnostics={"backbone": backbone,
                     "head": "-".join(type(m).__name__ for m in head.net),
                     "n_head_parameters": head.n_parameters, "hidden": hidden,
                     "layers": LAYERS,
                     "n_features": int(Xtr.shape[1]), "feature_names": list(feature_names or []),
                     "loss": loss, "use_prev_year_feature": bool(use_prev),
                     "n_species": len(species), "train_years": train_years,
                     "val_year": val_year, "epochs_run": len(losses),
                     "loss_first": losses[0] if losses else None,
                     "loss_last": losses[-1] if losses else None,
                     "best_val_loss": None if best == float("inf") else best,
                     "n_train_cellyears": int(Xtr.shape[0]),
                     "n_train_observations": int(Mtr.sum()),
                     "device": str(dev), "seed": seed, "resources": meter.report(),
                     "saved": saved,
                     "note": "features are the frozen model's decoded fields sampled at each "
                             "cell; the head is the only thing fitted and is identical for "
                             "BioAnalyst and Aurora"},
        wall_s=round(time.time() - t0, 3))


def run_probe(panel: pd.DataFrame, splits: Sequence[Split], seeds: Sequence[int],
              features_by_year: dict[int, np.ndarray], cells_by_year: dict[int, pd.DataFrame],
              out_dir: str | Path, *, backbone: str = "bfm", runner: Any = None,
              **kwargs: Any) -> dict[str, Any]:
    """Run L2 over splits and seeds, scoring against the same references as the nulls."""
    from ..common.runner import write_json

    task = str(panel["task"].iloc[0])
    out_dir = Path(out_dir)
    results = {}
    for seed in seeds:
        record = {"baseline": f"{backbone}_l2", "task": task,
                  "value_type": str(panel["value_type"].iloc[0]), "splits": []}
        for split in splits:
            fit = fit_probe(panel, split, seed, features_by_year, cells_by_year,
                            backbone=backbone, **kwargs)
            # Same envelope as baselines.run_baseline, so every learned predictor lands in
            # one shape and the results tables need no per-model special case.
            record["splits"].append({
                "split": split.as_dict(), "seed": seed,
                "n_rows": int(len(fit.predictions)), "wall_s": fit.wall_s,
                "diagnostics": fit.diagnostics,
                "metrics": score_baseline(panel, split, fit, seed=seed)})
        path = out_dir / f"{task}_{backbone}_l2_s{seed}.json"
        write_json(path, record)
        results[seed] = str(path)
    return {"task": task, "backbone": backbone, "written": results}
