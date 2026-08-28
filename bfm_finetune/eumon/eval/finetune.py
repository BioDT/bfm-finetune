"""L3 — parameter-efficient fine-tuning: VeRA, LoRA, and full fine-tuning.

Adapters are implemented here rather than reused from ``bfm-model``, whose VeRA swaps the
two scaling vectors relative to Kopiczko et al. (2024); ``VeRALinear`` takes ``as_shipped``
so the difference can be measured. Both methods adapt the same modules — the swin blocks'
fused ``qkv`` and ``proj`` — applied identically so it cannot bias the comparison.
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from bfm_model.bfm.dataloader_monthly import batch_to_device

from .baselines import BaselineFit
from .nulls import observed, test_frame
from .probe import (HIDDEN, ProbeHead, _masked_loss, _to_prediction, prev_year_matrix,
                    value_domain_upper)
from .. import batch as B, model as M
from ..common.invariants import (adapters_received_gradient, no_test_year_in_training,
                                 predictions_in_domain)
from ..common.resources import Meter

TARGET_SUFFIXES = ("qkv", "proj")


class LoRALinear(nn.Module):
    """``W₀x + (α/r)·BAx`` with A Gaussian and B zero, so ΔW = 0 at initialisation."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.normal_(self.lora_A, std=1.0 / r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return self.base(x) + self.scaling * delta

    @property
    def n_trainable(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


class VeRALinear(nn.Module):
    """``W₀x + Λ_b·B·Λ_d·A·x`` with A, B frozen random and shared across adapted layers.

    ``lam_d`` is the paper's d (initialised to ``d_initial``) and ``lam_b`` the paper's b
    (initialised to zero). ``as_shipped`` reproduces ``bfm-model``'s swapped initialisation.
    """

    def __init__(self, base: nn.Linear, r: int, shared_A: torch.Tensor, shared_B: torch.Tensor,
                 d_initial: float = 0.1, dropout: float = 0.0, as_shipped: bool = False):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("vera_A", shared_A[:r, :base.in_features], persistent=False)
        self.register_buffer("vera_B", shared_B[:base.out_features, :r], persistent=False)
        if as_shipped:
            self.lam_d = nn.Parameter(torch.ones(r))
            self.lam_b = nn.Parameter(torch.full((base.out_features,), d_initial))
        else:
            self.lam_d = nn.Parameter(torch.full((r,), d_initial))
            self.lam_b = nn.Parameter(torch.zeros(base.out_features))
        self.as_shipped = as_shipped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = (self.dropout(x) @ self.vera_A.T) * self.lam_d
        return self.base(x) + (h @ self.vera_B.T) * self.lam_b

    @property
    def n_trainable(self) -> int:
        return self.lam_d.numel() + self.lam_b.numel()


def _targets(module: nn.Module, suffixes: Sequence[str]) -> list[tuple[nn.Module, str, nn.Linear]]:
    found = []
    for _, child in module.named_modules():
        for attr, sub in list(child.named_children()):
            if isinstance(sub, nn.Linear) and attr.split(".")[-1] in suffixes:
                found.append((child, attr, sub))
    return found


def inject(model: nn.Module, method: str, *, rank: int, alpha: float = 16.0,
           d_initial: float = 0.1, as_shipped: bool = False, dropout: float = 0.0,
           suffixes: Sequence[str] = TARGET_SUFFIXES, backbone_only: bool = True,
           seed: int = 0) -> dict[str, Any]:
    """Replace the targeted linears in place. Returns a record of what was adapted."""
    root = getattr(model, "backbone", model) if backbone_only else model
    targets = _targets(root, suffixes)
    if not targets:
        raise ValueError(f"no linear modules ending in {tuple(suffixes)} found to adapt")

    generator = torch.Generator().manual_seed(seed)
    shared_A = shared_B = None
    if method == "vera":
        max_in = max(t[2].in_features for t in targets)
        max_out = max(t[2].out_features for t in targets)
        shared_A = torch.empty(rank, max_in)
        shared_B = torch.empty(max_out, rank)
        nn.init.kaiming_uniform_(shared_A, a=math.sqrt(5), generator=generator)
        nn.init.kaiming_uniform_(shared_B, a=math.sqrt(5), generator=generator)

    adapted, n_trainable = [], 0
    for parent, attr, linear in targets:
        if method == "lora":
            new = LoRALinear(linear, r=rank, alpha=alpha, dropout=dropout)
        elif method == "vera":
            new = VeRALinear(linear, r=rank, shared_A=shared_A.to(linear.weight.device),
                             shared_B=shared_B.to(linear.weight.device), d_initial=d_initial,
                             dropout=dropout, as_shipped=as_shipped)
        else:
            raise ValueError(f"unknown method {method!r}")
        setattr(parent, attr, new.to(linear.weight.device))
        adapted.append(f"{attr}[{linear.in_features}->{linear.out_features}]")
        n_trainable += new.n_trainable

    return {"method": method, "rank": rank, "alpha": alpha if method == "lora" else None,
            "d_initial": d_initial if method == "vera" else None,
            "as_shipped_init": as_shipped if method == "vera" else None,
            "shared_random_matrices": method == "vera",
            "n_adapted_modules": len(adapted), "n_trainable_adapter": n_trainable,
            "targets": sorted(set(adapted)),
            "delta_w_zero_at_init": (method == "lora") or (method == "vera" and not as_shipped),
            "note": "adapts the fused qkv and proj, a superset of the papers' {Wq, Wv}; "
                    "applied identically to both methods"}


def freeze_backbone(model: nn.Module, method: str) -> dict[str, int]:
    """Freeze everything except the adapters. ``full`` leaves the model trainable."""
    if method == "full":
        for p in model.parameters():
            p.requires_grad = True
        total = sum(p.numel() for p in model.parameters())
        return {"n_trainable_model": total, "n_frozen": 0}

    trainable = 0
    for name, p in model.named_parameters():
        p.requires_grad = any(k in name for k in ("lora_A", "lora_B", "lam_d", "lam_b"))
        if p.requires_grad:
            trainable += p.numel()
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"n_trainable_model": trainable, "n_frozen": frozen}


@dataclass
class L3Config:
    method: str = "vera"            # vera | lora | full
    rank: int = 256
    alpha: float = 16.0
    d_initial: float = 0.1
    as_shipped: bool = False
    lr_adapter: float = 1e-2        # VeRA's paper uses a markedly higher LR than LoRA
    lr_head: float = 4e-3
    lr_full: float = 5e-5
    weight_decay: float = 1e-4
    epochs: int = 20
    val_year: int = 2018
    grad_checkpointing: bool = True
    bf16: bool = True
    loss: str = "log1p_mse"        # log1p_mse | poisson (see probe._masked_loss)
    use_prev: bool = False         # give the head last year's value, as persistence has

    def learning_rate(self) -> float:
        return self.lr_full if self.method == "full" else self.lr_adapter

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in
                ("method", "rank", "alpha", "d_initial", "as_shipped", "lr_adapter", "lr_head",
                 "lr_full", "weight_decay", "epochs", "val_year", "grad_checkpointing", "bf16",
                 "loss", "use_prev")}


PRESETS: dict[str, L3Config] = {
    # Paper-faithful VeRA: rank 256 as used for RoBERTa-large and ViT, delta-W zero.
    "vera": L3Config(method="vera", rank=256, d_initial=0.1, as_shipped=False),
    # The same, with bfm-model's swapped initialisation, so that bug's effect is measured.
    "vera_asshipped": L3Config(method="vera", rank=256, d_initial=0.1, as_shipped=True),
    # LoRA at the repo's stated rank, with alpha corrected to equal r per the paper's rule.
    "lora16": L3Config(method="lora", rank=16, alpha=16.0, lr_adapter=1e-3),
    # Budget-matched arm: r=1 puts LoRA within ~20% of VeRA's trainable count.
    "lora1": L3Config(method="lora", rank=1, alpha=1.0, lr_adapter=1e-3),
    # Default: the Task A rank ablation flattens by r=4.
    "lora4": L3Config(method="lora", rank=4, alpha=4.0, lr_adapter=1e-3),
    "full": L3Config(method="full", lr_full=5e-5, epochs=10),
}

# Head loss per target type, fixed by the target's distribution rather than a validation
# sweep. Task C's overdispersed index needs a log link targeting E[y] directly; log-space
# MSE carries a retransformation bias there.
LOSS_FOR_VALUE_TYPE = {"count": "log1p_mse", "prevalence": "log1p_mse", "index": "poisson"}


def loss_for(value_type: str, requested: str = "auto") -> str:
    """Resolve ``--loss auto`` against the target type; anything else is taken as given."""
    if requested != "auto":
        return requested
    return LOSS_FOR_VALUE_TYPE.get(str(value_type), "log1p_mse")


def parameter_report(model: nn.Module, cfg: L3Config, inject_record: dict[str, Any] | None,
                     head: nn.Module | None = None) -> dict[str, Any]:
    """The number the comparison turns on: trainable parameters per arm."""
    counts = freeze_backbone(model, cfg.method) if cfg.method != "full" else {
        "n_trainable_model": sum(p.numel() for p in model.parameters()), "n_frozen": 0}
    head_params = sum(p.numel() for p in head.parameters()) if head is not None else 0
    total_model = sum(p.numel() for p in model.parameters())
    # The head is fitted, so it belongs in both numerator and denominator; the
    # backbone-only fraction is kept because that is what the PEFT papers quote.
    total_fitted = counts["n_trainable_model"] + head_params
    return {"config": cfg.as_dict(), "inject": inject_record,
            **counts, "n_head": head_params,
            "n_trainable_total": total_fitted,
            "n_model_total": total_model,
            "n_parameters_total": total_model + head_params,
            "trainable_fraction": total_fitted / max(total_model + head_params, 1),
            "trainable_fraction_of_backbone": counts["n_trainable_model"] / max(total_model, 1)}


def _gather_head_inputs(decoded: dict[str, Any], cells, device) -> torch.Tensor:
    """Decoded fields -> [n_cells, n_features], differentiably (no numpy round-trip)."""
    names, cols = [], []
    ci = torch.as_tensor(cells["cell_i"].to_numpy(), dtype=torch.long, device=device)
    cj = torch.as_tensor(cells["cell_j"].to_numpy(), dtype=torch.long, device=device)
    for group in sorted(decoded):
        variables = decoded[group]
        if not isinstance(variables, dict):
            continue
        for var in sorted(variables):
            t = variables[var]
            t = t.reshape(-1, *t.shape[-2:]) if t.dim() > 2 else t[None]
            for level in range(t.shape[0]):
                cols.append(t[level][ci, cj])
                names.append(f"{group}.{var}.{level}")
    return torch.stack(cols, dim=1)


def _bfm_features(model, xb, cells, device, bf16):
    with torch.autocast(device, torch.bfloat16, enabled=bf16):
        dec = M.forward_with_latents(model, xb, batch_size=1)["decoded"]
    return _gather_head_inputs(dec, cells, device)


def _aurora_features(model, xb, cells, device, bf16):
    """Aurora's decoded fields at the panel cells, differentiably.

    ``aurora.forward_fields`` is no-grad and returns numpy for the L2 path; L3 must
    backpropagate, so the same orientation permutation is applied here to live tensors.
    """
    from .aurora import _grid_orientation, _invert_perm
    from ..panel import GRID

    lat_perm, lon_perm = _grid_orientation(GRID)
    inv_lat = torch.as_tensor(_invert_perm(lat_perm), device=device, dtype=torch.long)
    inv_lon = torch.as_tensor(_invert_perm(lon_perm), device=device, dtype=torch.long)
    ci = torch.as_tensor(cells["cell_i"].to_numpy(), dtype=torch.long, device=device)
    cj = torch.as_tensor(cells["cell_j"].to_numpy(), dtype=torch.long, device=device)

    with torch.autocast(device, torch.bfloat16, enabled=bf16):
        pred = model(xb)
    cols = []
    for group in (pred.surf_vars, pred.atmos_vars):
        for name in sorted(group):
            t = group[name]
            t = t.reshape(-1, *t.shape[-2:])
            for k in range(t.shape[0]):
                field = t[k].index_select(0, inv_lat).index_select(1, inv_lon)
                cols.append(field[ci, cj])
    return torch.stack(cols, dim=1)


def _persist(save_dir, frame, model, head, cfg, seed, split, backbone, history, val_history,
             save_full: bool, feat_mu=None, feat_sd=None, arm: str | None = None
             ) -> dict[str, Any]:
    """Write predictions and the trained state so a result can be re-scored without a GPU.

    Predictions are the important half: with them, a new metric or corrected scoring bug is
    a CPU pass over parquet rather than a retrain. Adapters are kept (~1 MB); the full
    fine-tune's ~2.7 GB state is opt-in.
    """
    from ..common.runner import atomic_path, write_json

    # Keyed by the *arm*, not `cfg.method`: lora1, lora4 and lora16 all report method
    # "lora" and would otherwise overwrite each other.
    out = Path(save_dir) / f"{backbone}_l3_{arm or cfg.method}_s{seed}_{split.test_year}"
    out.mkdir(parents=True, exist_ok=True)

    with atomic_path(out / "predictions.parquet", suffix=".parquet") as tmp:
        frame.to_parquet(tmp, index=False, compression="zstd")

    full_unrequested = cfg.method == "full" and not save_full
    trainable = ({} if full_unrequested else
                 {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad})
    state = {"adapters": trainable,
             "adapters_omitted": ("full fine-tune weights not saved; pass --save-full to keep "
                                  "the ~2.7 GB state" if full_unrequested else None),
             "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
             "config": cfg.as_dict(), "seed": seed, "split": split.as_dict(),
             "backbone": backbone,
             # Without these the checkpoint cannot reproduce its own predictions: the head
             # was fitted on standardised inputs.
             "feature_mu": None if feat_mu is None else feat_mu.detach().cpu(),
             "feature_sd": None if feat_sd is None else feat_sd.detach().cpu()}
    if save_full and cfg.method == "full":
        state["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    with atomic_path(out / "state.pt", suffix=".pt") as tmp:
        torch.save(state, tmp)

    write_json(out / "run.json", {"config": cfg.as_dict(), "seed": seed,
                                  "split": split.as_dict(), "backbone": backbone,
                                  "train_history": history, "val_history": val_history,
                                  "n_predictions": int(len(frame)),
                                  "n_trainable_tensors": len(trainable),
                                  "full_model_saved": bool(save_full and cfg.method == "full")})
    return {"dir": str(out), "predictions": "predictions.parquet", "state": "state.pt",
            "bytes": sum(f.stat().st_size for f in out.iterdir() if f.is_file())}


def _augment(model, year, inputs, cells_by_year, device, cfg, featurise, panel, species):
    """Backbone features for a year, with last year's values appended when ``cfg.use_prev``.

    The extra columns are constants with respect to the adapters, so they change what the
    head can condition on without touching the gradient path into the backbone.
    """
    feats = featurise(model, inputs[year], cells_by_year[year], device, cfg.bf16)
    if not cfg.use_prev:
        return feats
    prev = prev_year_matrix(panel, year, cells_by_year[year], species)
    return torch.cat([feats, torch.from_numpy(prev).to(feats.device, feats.dtype)], dim=1)


def fit_l3(panel, split, seed: int, cfg: L3Config, *, device: str = "cuda:0",
           biocube_dir=None, hidden: int | None = None, max_train_years: int | None = None,
           log_every: int = 0, backbone: str = "bfm", patience: int = 25,
           return_internals: bool = False, save_dir=None, save_full: bool = False,
           arm: str | None = None):
    """Adapt the backbone and train the L2 head end to end, early-stop on the validation
    year, then score the held-out test year."""
    no_test_year_in_training(split.train_years, split.test_year)
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    meter = Meter(device=int(device.split(":")[-1]) if ":" in str(device) else None,
                  label=f"L3_{cfg.method}_seed{seed}")
    meter.__enter__()

    task = str(panel["task"].iloc[0])
    obs = observed(panel)
    species = sorted(obs["species"].unique().tolist())
    s_index = {s: k for k, s in enumerate(species)}

    if backbone == "aurora":
        from . import aurora as AU

        model, _ = AU.build(device=device)
        featurise = _aurora_features
        prepare = AU.to_aurora_batch
    else:
        model, _ = M.build(device=device)
        featurise = _bfm_features
        prepare = None

    inject_record = None
    if cfg.method != "full":
        inject_record = inject(model, cfg.method, rank=cfg.rank, alpha=cfg.alpha,
                               d_initial=cfg.d_initial, as_shipped=cfg.as_shipped, seed=seed)
    freeze_backbone(model, cfg.method)

    mcfg = M.load_config()
    # Aurora needs raw ERA5 units; BioAnalyst needs BioCube's scaled channels.
    if backbone == "aurora":
        ds = AU.aurora_dataset(mcfg, biocube_dir or B.BIOCUBE_DIR)
    else:
        ds = B.make_dataset(mcfg, biocube_dir or B.BIOCUBE_DIR)

    def stage(year: int):
        try:
            spec = B.forecast_input(task, year, biocube_dir or B.BIOCUBE_DIR)
        except (B.WindowUnavailable, ValueError):
            return None
        x, _ = B.sanitise(B.load_input(spec["path"], ds))
        return prepare(x) if prepare else batch_to_device(B.collate_for_model(x), device)

    years = [y for y in split.train_years if y != cfg.val_year]
    if max_train_years:
        years = years[-max_train_years:]

    inputs, cells_by_year, target_by_year = {}, {}, {}
    for year in years + [cfg.val_year, split.test_year]:
        xb = stage(year)
        if xb is None:
            continue
        cells = (obs.loc[obs["year"] == year, ["unit_id", "cell_i", "cell_j"]]
                 .drop_duplicates("unit_id").reset_index(drop=True))
        if cells.empty:
            continue
        pos = {u: k for k, u in enumerate(cells["unit_id"].tolist())}
        y = np.zeros((len(cells), len(species)), dtype=np.float32)
        m = np.zeros((len(cells), len(species)), dtype=bool)
        sub = obs.loc[obs["year"] == year]
        for unit, sp, val in zip(sub["unit_id"], sub["species"], sub["value"]):
            r, c = pos.get(unit), s_index.get(sp)
            if r is not None and c is not None:
                y[r, c], m[r, c] = val, True
        inputs[year] = xb
        cells_by_year[year] = cells
        target_by_year[year] = (torch.from_numpy(y).to(device), torch.from_numpy(m).to(device))

    train_years = [y for y in years if y in inputs]
    if not train_years:
        raise ValueError("no usable training years")

    # Standardise head inputs per column exactly as L2 does, or an L2 -> L3 comparison
    # confounds adaptation with preprocessing. Statistics come once from the frozen model
    # and are held fixed.
    with torch.no_grad():
        base = torch.cat([_augment(model, y, inputs, cells_by_year, device, cfg, featurise,
                                   panel, species).float() for y in train_years], dim=0)
        feat_mu = base.mean(dim=0, keepdim=True)
        feat_sd = base.std(dim=0, keepdim=True) + 1e-6
        n_features = base.shape[1]
        del base

    def standardise(x: torch.Tensor) -> torch.Tensor:
        return (x.float() - feat_mu) / feat_sd

    # probe.HIDDEN is the single source of truth; a literal here would let L2 and L3 drift.
    head = ProbeHead(n_features, len(species),
                     HIDDEN if hidden is None else hidden).to(device)

    opt = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg.learning_rate()},
         {"params": list(head.parameters()), "lr": cfg.lr_head}], weight_decay=cfg.weight_decay)

    def loss_on(year: int, grad: bool):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            feats = _augment(model, year, inputs, cells_by_year, device, cfg, featurise,
                             panel, species)
            pred = head(standardise(feats))
            tgt, msk = target_by_year[year]
            return _masked_loss(pred, tgt, msk, cfg.loss)

    history, val_history, step_times = [], [], []
    best, best_state, stale = float("inf"), None, 0
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    for epoch in range(cfg.epochs):
        head.train()
        total = 0.0
        for year in train_years:
            ts = time.time()
            opt.zero_grad(set_to_none=True)
            loss = loss_on(year, grad=True)
            loss.backward()
            if not history and year == train_years[0]:
                # Once, on the first backward of the run: the only check that distinguishes
                # a fine-tune from a head-only fit.
                adapters_received_gradient(model, cfg.method)
            opt.step()
            total += float(loss)
            step_times.append(time.time() - ts)
        history.append(total / len(train_years))

        if cfg.val_year in inputs:
            head.eval()
            v = float(loss_on(cfg.val_year, grad=False))
            val_history.append(v)
            if v < best - 1e-6:
                best, stale = v, 0
                best_state = ({n: p.detach().clone() for n, p in model.named_parameters()
                               if p.requires_grad},
                              {k: t.detach().clone() for k, t in head.state_dict().items()})
            else:
                stale += 1
                if stale >= patience:
                    break
        if log_every and epoch % log_every == 0:
            print(f"    epoch {epoch:3d} train {history[-1]:.5f}"
                  + (f" val {val_history[-1]:.5f}" if val_history else ""), flush=True)

    if best_state is not None:
        adapters, head_state = best_state
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in adapters:
                    p.copy_(adapters[n])
        head.load_state_dict(head_state)

    head.eval()
    with torch.no_grad():
        feats = _augment(model, split.test_year, inputs, cells_by_year, device, cfg,
                         featurise, panel, species)
        grid = _to_prediction(head(standardise(feats)), cfg.loss,
                              value_domain_upper(str(panel["value_type"].iloc[0]))
                              ).float().cpu().numpy()

    test = test_frame(panel, split)
    cells = cells_by_year[split.test_year]
    pos = {u: k for k, u in enumerate(cells["unit_id"].tolist())}
    pred = np.full(len(test), np.nan, dtype=float)
    for k, (unit, sp) in enumerate(zip(test["unit_id"], test["species"])):
        r, c = pos.get(unit), s_index.get(sp)
        if r is not None and c is not None:
            pred[k] = grid[r, c]
    frame = test.assign(y_pred=pred)
    frame = frame.loc[np.isfinite(frame["y_pred"].to_numpy(float))]
    predictions_in_domain(frame["y_pred"], str(panel["value_type"].iloc[0]),
                          where=f"L3 {backbone}/{cfg.method} task {task}")

    meter.__exit__(None, None, None)
    saved = _persist(save_dir, frame, model, head, cfg, seed, split, backbone,
                     history, val_history, save_full, feat_mu, feat_sd,
                     arm=arm) if save_dir else None
    fit = BaselineFit(
        name=f"{backbone}_l3_{cfg.method}", predictions=frame,
        diagnostics={"backbone": backbone, "n_features": int(n_features),
                     "n_trainable_adapter_tensors": len(trainable_names),
                     **parameter_report(model, cfg, inject_record, head),
                     "train_years": train_years, "epochs_run": len(history),
                     "loss_first": history[0], "loss_last": history[-1],
                     "val_first": val_history[0] if val_history else None,
                     "best_val": None if best == float("inf") else best,
                     "early_stopped": len(history) < cfg.epochs,
                     "median_step_s": float(np.median(step_times)),
                     "feature_standardisation": "per-column, from the frozen model on the "
                                                "training years, held fixed",
                     "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                     "seed": seed, "device": device, "resources": meter.report(),
                     "saved": saved},
        wall_s=round(time.time() - t0, 1))
    if return_internals:
        return fit, {"model": model, "head": head, "history": history,
                     "val_history": val_history, "median_step_s": float(np.median(step_times)),
                     "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                     "train_years": train_years}
    return fit
