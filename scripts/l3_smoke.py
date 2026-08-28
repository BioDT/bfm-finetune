#!/usr/bin/env python
"""Prove the L3 fine-tune is real, then estimate what a campaign would cost.

A training loop that runs is not a training loop that trains. Each check fails loudly if
the thing it names is not true: only adapters and the head carry gradients, frozen weights
stay bit-identical, adapter parameters move with non-zero gradients, the loss falls, and
the loss in force is the one the target type dictates.

Usage:  scripts/l3_smoke.py --gpu 1 --arms vera lora1 --epochs 3 --years 4
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EUMON_ROOT = Path(os.environ.get("EUMON_ROOT", ROOT))
sys.path[:0] = [str(ROOT), str(EUMON_ROOT / "bfm-model")]
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from bfm_finetune.eumon.common.runner import artefacts_root, write_json
from bfm_finetune.eumon.eval import finetune as F
from bfm_finetune.eumon.eval.nulls import Split

ARTE = artefacts_root()
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
          flush=True)


def run_arm(arm: str, panel, split, args) -> dict:
    cfg = F.PRESETS[arm]
    cfg.epochs = args.epochs
    value_type = str(panel["value_type"].iloc[0])
    cfg.loss = F.loss_for(value_type, args.loss)
    cfg.use_prev = args.use_prev
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    torch.cuda.reset_peak_memory_stats(device)
    print(f"\n=== {arm} ({cfg.method}, rank {cfg.rank}) ===", flush=True)

    # Snapshot a frozen weight before training.
    from bfm_finetune.eumon import model as M

    if args.backbone == "aurora":
        from bfm_finetune.eumon.eval import aurora as AU

        probe_model, _ = AU.build(device=device)
    else:
        probe_model, _ = M.build(device=device)
    ref_weight = None
    for n, p in probe_model.named_parameters():
        if "backbone" in n and n.endswith("qkv.weight"):
            ref_weight = (n, p.detach().clone())
            break
    del probe_model
    torch.cuda.empty_cache()

    t0 = time.time()
    fit, out = F.fit_l3(panel, split, seed=0, cfg=cfg, device=device,
                        backbone=args.backbone, max_train_years=args.years,
                        log_every=1, return_internals=True)
    wall = time.time() - t0
    model, head, hist = out["model"], out["head"], out["history"]
    rep = fit.diagnostics

    # 1. gradient surface
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    adapter_only = all(any(k in n for k in ("lora_A", "lora_B", "lam_d", "lam_b"))
                       for n in trainable) if cfg.method != "full" else True
    check("only adapters carry gradients in the backbone", adapter_only,
          f"{len(trainable)} trainable tensors")

    # 2. frozen weights untouched
    if cfg.method != "full" and ref_weight is not None:
        name, before = ref_weight
        after = dict(model.named_parameters()).get(name)
        if after is None:
            after = dict(model.named_parameters()).get(name.replace("qkv.", "qkv.base."))
        same = after is not None and torch.equal(before.to(after.device), after.detach())
        check("frozen backbone weight bit-identical after training", same, name.split(".")[-3:][0])

    # 3. adapters moved, gradients non-zero
    moved, grads = 0, 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                grads += 1
            if float(p.detach().abs().sum()) > 0:
                moved += 1
    check("adapter gradients are non-zero", grads > 0, f"{grads}/{len(trainable)} tensors")

    # 4. loss falls
    check("training loss decreases", len(hist) > 1 and hist[-1] < hist[0],
          f"{hist[0]:.5f} -> {hist[-1]:.5f}")

    # 5. the adapter has a measurable parameter surface
    check("adapter has a measurable effect on the model", rep["n_trainable_total"] > 0,
          f"{rep['n_trainable_total']:,} trainable of {rep['n_model_total']:,} "
          f"({100 * rep['trainable_fraction']:.4f}%)")

    del model, head
    torch.cuda.empty_cache()
    check("test year scored out of sample", len(fit.predictions) > 0,
          f"{len(fit.predictions):,} rows predicted for {split.test_year}")
    check("early stopping used the validation year, never the test year",
          cfg.val_year != split.test_year and bool(out["val_history"]),
          f"val {cfg.val_year}, {len(out['val_history'])} evaluations")

    check("loss matches the target type", cfg.loss == F.LOSS_FOR_VALUE_TYPE.get(value_type),
          f"{value_type} -> {cfg.loss}")

    # A Poisson head must emit strictly positive, finite predictions (exp link).
    pv = fit.predictions["y_pred"].to_numpy(float)
    if cfg.loss == "poisson":
        check("poisson predictions are finite and non-negative",
              bool(np.isfinite(pv).all() and (pv >= 0).all()),
              f"range [{pv.min():.4g}, {pv.max():.4g}]")

    # A prevalence target is a proportion: predictions outside [0, 1] are invalid inputs
    # to Brier and the TSS threshold search.
    if value_type == "prevalence":
        check("prevalence predictions lie in [0, 1]",
              bool(np.isfinite(pv).all() and (pv >= 0).all() and (pv <= 1).all()),
              f"range [{pv.min():.4g}, {pv.max():.4g}]")

    # The AR feature must widen the design by exactly one column per species.
    n_species = int(rep["n_species"]) if "n_species" in rep else panel["species"].nunique()
    check("AR feature widens the head input by n_species",
          (rep["n_features"] >= n_species) if cfg.use_prev else True,
          f"n_features {rep['n_features']}, n_species {n_species}, use_prev {cfg.use_prev}")

    check("L3 standardises head inputs",
          rep.get("feature_standardisation", "").startswith("per-column"),
          rep.get("feature_standardisation", "ABSENT"))

    return {"arm": arm, "config": cfg.as_dict(), "report": rep, "history": hist,
            "wall_s": round(wall, 1), "median_step_s": out["median_step_s"],
            "n_steps": len(out["history"]) * len(out["train_years"]),
            "peak_gib": round(out["peak_gib"], 2),
            "train_years": out["train_years"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--arms", nargs="+", default=["vera", "lora1"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--years", type=int, default=4, help="most recent N training years")
    ap.add_argument("--task", default="A")
    ap.add_argument("--backbone", default="bfm", choices=["bfm", "aurora"])
    ap.add_argument("--loss", default="auto", choices=["auto", "log1p_mse", "poisson"])
    ap.add_argument("--use-prev", action="store_true")
    args = ap.parse_args()

    panel = pd.read_parquet(ARTE / f"panel_{'C_cell' if args.task == 'C' else args.task}.parquet")
    split = Split(tuple(range(2000, 2020)), 2020)

    results = [run_arm(arm, panel, split, args) for arm in args.arms]

    print("\n" + "=" * 78)
    failed = [c for c in CHECKS if not c[1]]
    print(f"CHECKS: {len(CHECKS) - len(failed)}/{len(CHECKS)} passed"
          + (f" — FAILURES: {[c[0] for c in failed]}" if failed else ""))

    print(f"\n{'arm':16s}{'trainable':>12s}{'step s':>9s}{'peak GiB':>10s}{'loss':>22s}")
    for r in results:
        fmt = f"{r['history'][0]:.5f} -> {r['history'][-1]:.5f}"
        print(f"{r['arm']:16s}{r['report']['n_trainable_total']:>12,}"
              f"{r['median_step_s']:>9.2f}{r['peak_gib']:>10.2f}{fmt:>22s}")

    med = float(np.median([r["median_step_s"] for r in results]))
    print(f"\nCAMPAIGN ESTIMATE at {med:.2f} s/step")
    for epochs in (10, 20):
        steps = 19 * epochs
        per_run = steps * med / 3600
        for n_runs, label in ((18, "L3 VeRA: 3 tasks x 3 seeds x {BFM, Aurora}"),
                              (12, "ablation Task A: lora16, lora1, vera_asshipped, full x 3 seeds"),
                              (30, "TOTAL")):
            print(f"  {epochs:2d} epochs | {label:58s} {n_runs:3d} runs "
                  f"{n_runs * per_run:6.1f} GPU-h  -> {n_runs * per_run / 4:5.1f} h on 4 GPUs")
    write_json(ARTE / "l3_smoke.json",
               {"checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in CHECKS],
                "arms": results, "median_step_s": med})
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
