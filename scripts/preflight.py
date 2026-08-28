#!/usr/bin/env python
"""End-to-end preflight: every rung, both backbones, every baseline, on a tiny budget.

``audit.py`` checks the code and the data at rest; this runs the actual pipeline once per
predictor and checks that what comes out is well-formed: predictions exist, land on the
rows they were asked for, stay inside the target's domain, carry the metric set the tables
expect, and record what they cost. It is not a performance test — the budgets are tiny and
the numbers it produces are meaningless.

    scripts/preflight.py --gpu 1                 # everything
    scripts/preflight.py --gpu 1 --only l2 l3    # a subset of settings

Settings: nulls, baselines, l0l1, l2, l3. Aurora has no species outputs, so it cannot run L0
or L1 at all — that asymmetry is asserted rather than skipped silently.
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

os.environ.setdefault("EUMON_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import pandas as pd
import torch

from bfm_finetune.eumon.common.runner import artefacts_root
from bfm_finetune.eumon.eval.nulls import Split, test_frame

ARTE = artefacts_root()
PANEL_FOR = {"A": "panel_A", "B": "panel_B", "C": "panel_C_cell"}
DOMAIN_UPPER = {"prevalence": 1.0}
CHECKS: list[tuple[str, bool, str]] = []

# Keys every scored result must carry, or the tables silently render blanks.
REQUIRED_METRICS = ("n", "rmse", "rmse_log", "spatial_rho", "calibration", "skill",
                    "skill_vs_strongest_null")
REQUIRED_SKILL = ("skill_score", "skill_score_log1p", "n_scored")


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
          flush=True)


def panel(task: str) -> pd.DataFrame:
    return pd.read_parquet(ARTE / f"{PANEL_FOR[task]}.parquet")


def verify_predictions(tag: str, frame: pd.DataFrame, split: Split,
                       p: pd.DataFrame) -> None:
    """Shape, domain and alignment of a prediction frame — before any score is read."""
    value_type = str(p["value_type"].iloc[0])
    truth = test_frame(p, split)
    pred = frame["y_pred"].to_numpy(float)

    check(f"{tag}: predictions are finite", bool(np.isfinite(pred).all()),
          f"{int((~np.isfinite(pred)).sum())} non-finite of {len(pred):,}")
    check(f"{tag}: predictions are non-negative", bool((pred >= 0).all()),
          f"min {pred.min():.4g}")
    upper = DOMAIN_UPPER.get(value_type)
    if upper is not None:
        check(f"{tag}: predictions respect the target domain [0, {upper}]",
              bool((pred <= upper).all()), f"max {pred.max():.4g}")
    check(f"{tag}: predicts only rows in the test frame", len(frame) <= len(truth),
          f"{len(frame):,} predicted of {len(truth):,} test rows")
    check(f"{tag}: predicted only the test year",
          set(frame["year"].unique()) == {split.test_year},
          f"years {sorted(frame['year'].unique())}")
    keys = set(zip(frame["unit_id"], frame["species"]))
    tkeys = set(zip(truth["unit_id"], truth["species"]))
    check(f"{tag}: every predicted key exists in the panel", keys <= tkeys,
          f"{len(keys - tkeys)} unknown (unit, species) pairs")
    if "y_true" in frame:
        merged = frame.merge(truth, on=["unit_id", "species"], suffixes=("", "_ref"))
        agree = np.allclose(merged["y_true"].to_numpy(float),
                            merged["y_true_ref"].to_numpy(float), equal_nan=True)
        check(f"{tag}: y_true matches the panel row for row", agree,
              f"{len(merged):,} rows compared")


def verify_metrics(tag: str, score: dict) -> None:
    missing = [k for k in REQUIRED_METRICS if k not in score]
    check(f"{tag}: carries the full metric set", not missing,
          f"missing {missing}" if missing else f"{len(score)} fields")
    sk = score.get("skill") or {}
    for ref, rec in sk.items():
        gaps = [k for k in REQUIRED_SKILL if k not in rec]
        if gaps:
            check(f"{tag}: skill block {ref} complete", False, f"missing {gaps}")
            return
    check(f"{tag}: every skill block is complete", True, f"{len(sk)} references")
    h = score.get("skill_vs_strongest_null") or {}
    vals = [v["skill_score"] for v in sk.values() if np.isfinite(v.get("skill_score", np.nan))]
    check(f"{tag}: headline equals the worst reference",
          bool(vals) and abs(h.get("skill_score", np.nan) - min(vals)) < 1e-9,
          f"headline {h.get('skill_score')} vs min {min(vals) if vals else None} "
          f"({h.get('reference')})")


def verify_resources(tag: str, diag: dict, expect_gpu: bool) -> None:
    r = diag.get("resources")
    if not r:
        check(f"{tag}: cost recorded", False, "no resources block")
        return
    ok = r.get("wall_s", 0) > 0 and r.get("cpu_core_seconds") is not None
    check(f"{tag}: cost recorded", ok,
          f"wall {r.get('wall_s')}s, cpu {r.get('cpu_core_seconds')} core-s, "
          f"gpu energy {r.get('energy_wh')} Wh (above idle {r.get('energy_wh_above_idle')}), "
          f"shared_gpu={r.get('shared_gpu')}")
    if expect_gpu:
        check(f"{tag}: GPU energy integrated", (r.get("energy_wh") or 0) > 0,
              f"{r.get('n_samples')} samples, idle {r.get('idle_power_w')} W")


# ------------------------------------------------------------------ settings

def run_nulls(task: str, split: Split, args) -> None:
    from bfm_finetune.eumon.eval.nulls import score_nulls

    p = panel(task)
    rec = score_nulls(p, split)
    scores = rec.get("scores", rec)
    check(f"nulls/{task}: all seven nulls scored", len(scores) == 7, f"{sorted(scores)}")
    for name in sorted(scores):
        verify_metrics(f"nulls/{task}/{name}", scores[name])
    ref = scores["persistence"]["skill"]["vs_persistence"]["skill_score"]
    check(f"nulls/{task}: persistence against itself is 0", abs(ref) < 1e-12, f"{ref}")


def run_baselines(task: str, split: Split, args) -> None:
    from bfm_finetune.eumon.common.resources import Meter
    from bfm_finetune.eumon.eval.baselines import FIT_FUNCTIONS, score_baseline

    p = panel(task)
    for name in args.baselines:
        kwargs = {}
        expect_gpu = name == "convlstm"
        if expect_gpu:
            kwargs = {"device": f"cuda:{args.gpu}", "epochs": 2}
        if name == "randomforest":
            kwargs = {"n_estimators": 10}
        t0 = time.time()
        meter = Meter(device=args.gpu if expect_gpu else None, label=f"{name}_{task}")
        try:
            with meter:
                fit = FIT_FUNCTIONS[name](p, split, 0, **kwargs)
            fit.diagnostics["resources"] = meter.report()
        except Exception as exc:
            check(f"{name}/{task}: runs", False, f"{type(exc).__name__}: {exc}"[:160])
            continue
        check(f"{name}/{task}: runs", True, f"{len(fit.predictions):,} rows in {time.time() - t0:.0f}s")
        verify_predictions(f"{name}/{task}", fit.predictions, split, p)
        verify_metrics(f"{name}/{task}", score_baseline(p, split, fit, seed=0))
        verify_resources(f"{name}/{task}", fit.diagnostics, expect_gpu)


def run_l0l1(task: str, split: Split, args) -> None:
    """L0 and L1, actually executed — reading a previous run's file off disk would check
    nothing about whether the code still works."""
    from bfm_finetune.eumon.eval import ladder as L

    p = panel(task)
    channels = L.overlapping_species(p)
    if not channels:
        check(f"l0l1/{task}: correctly has no zero-shot rung", True,
              f"{task} shares 0 species with the model's 28 channels, so L0/L1 are undefined")
        return
    check(f"l0l1/{task}: species overlap resolved", True,
          f"{len(channels)} species: {sorted(channels)}")

    fields = {}
    for year in (split.test_year, split.previous_year):
        f = ARTE / "decoded" / f"decoded_species_{task}_{year}.npy"
        if f.exists():
            fields[year] = np.load(f)
    if split.test_year not in fields:
        check(f"l0l1/{task}: decoded species field present", False,
              f"decoded_species_{task}_{split.test_year}.npy missing")
        return

    rec = L.run_ladder(p, [split], fields, out_dir=ARTE / "preflight_ladder")
    r = rec["results"][0]
    for rung in ("L0", "L1"):
        x = r[rung]
        check(f"l0l1/{task}/{rung}: executed", True,
              f"{x['n']} rows, {x['n_species']} species, "
              f"{x['fitted_parameters']} fitted parameters")
        check(f"l0l1/{task}/{rung}: rank metric finite",
              np.isfinite(x["spatial_rho"]["mean"]),
              f"rho {x['spatial_rho']['mean']:+.4f} over {x['spatial_rho']['n_groups']} species")
    check(f"l0l1/{task}: L0 fits exactly zero parameters",
          r["L0"]["fitted_parameters"] == 0, f"{r['L0']['fitted_parameters']}")
    check(f"l0l1/{task}: L1 fits two parameters per species",
          r["L1"]["fitted_parameters"] == 2 * len(channels),
          f"{r['L1']['fitted_parameters']} for {len(channels)} species")
    # L1 is a positive monotone rescale of L0, so Spearman must be identical.
    check(f"l0l1/{task}: L1 leaves the ranking identical to L0",
          abs(r["L0"]["spatial_rho"]["mean"] - r["L1"]["spatial_rho"]["mean"]) < 1e-6,
          f"L0 {r['L0']['spatial_rho']['mean']:+.6f} vs L1 {r['L1']['spatial_rho']['mean']:+.6f}")


def run_l2(task: str, split: Split, args) -> None:
    from bfm_finetune.eumon.eval.baselines import score_baseline
    from bfm_finetune.eumon.eval.finetune import loss_for
    from bfm_finetune.eumon.eval.probe import fit_probe

    p = panel(task)
    for backbone in args.backbones:
        feats, cells = {}, {}
        for year in range(2000, 2021):
            f = ARTE / "features" / f"features_{backbone}_{task}_{year}.npz"
            if not f.exists():
                continue
            d = np.load(f, allow_pickle=True)
            feats[year] = d["X"]
            cells[year] = pd.DataFrame({"unit_id": d["unit_id"].astype(str),
                                        "cell_i": d["cell_i"], "cell_j": d["cell_j"]})
        if split.test_year not in feats:
            check(f"l2/{backbone}/{task}: features present", False,
                  f"no features for {split.test_year}")
            continue
        use_prev = task in ("A", "C")
        try:
            fit = fit_probe(p, split, 0, feats, cells, device=f"cuda:{args.gpu}",
                            epochs=args.epochs, val_year=2018, backbone=backbone,
                            loss=loss_for(str(p["value_type"].iloc[0])), use_prev=use_prev)
        except Exception as exc:
            check(f"l2/{backbone}/{task}: runs", False, f"{type(exc).__name__}: {exc}"[:160])
            continue
        check(f"l2/{backbone}/{task}: runs", True,
              f"{fit.diagnostics['n_features']} features, {len(fit.predictions):,} rows")
        verify_predictions(f"l2/{backbone}/{task}", fit.predictions, split, p)
        verify_metrics(f"l2/{backbone}/{task}", score_baseline(p, split, fit, seed=0))
        verify_resources(f"l2/{backbone}/{task}", fit.diagnostics, expect_gpu=True)
        check(f"l2/{backbone}/{task}: head is the shared ProbeHead",
              "Linear" in str(fit.diagnostics.get("head", "")),
              str(fit.diagnostics.get("head")))


def run_l3(task: str, split: Split, args) -> None:
    from bfm_finetune.eumon.eval import finetune as F
    from bfm_finetune.eumon.eval.baselines import score_baseline

    p = panel(task)
    for backbone, arm in [(b, a) for b in args.backbones for a in args.arms]:
        cfg = F.PRESETS[arm]
        cfg.epochs = args.epochs
        cfg.loss = F.loss_for(str(p["value_type"].iloc[0]))
        cfg.use_prev = task in ("A", "C")
        save_dir = ARTE / "preflight_checkpoints" / task
        try:
            fit = F.fit_l3(p, split, 0, cfg, device=f"cuda:{args.gpu}", backbone=backbone,
                           max_train_years=args.years, save_dir=save_dir)
        except Exception as exc:
            check(f"l3/{backbone}:{arm}/{task}: runs", False, f"{type(exc).__name__}: {exc}"[:200])
            continue
        check(f"l3/{backbone}:{arm}/{task}: runs", True,
              f"{fit.diagnostics['n_trainable_total']:,} trainable, {len(fit.predictions):,} rows")
        verify_predictions(f"l3/{backbone}:{arm}/{task}", fit.predictions, split, p)
        verify_metrics(f"l3/{backbone}:{arm}/{task}", score_baseline(p, split, fit, seed=0))
        verify_resources(f"l3/{backbone}:{arm}/{task}", fit.diagnostics, expect_gpu=True)

        saved = fit.diagnostics.get("saved") or {}
        d = Path(saved.get("dir", "")) if saved else None
        ok = bool(d and (d / "predictions.parquet").exists() and (d / "state.pt").exists())
        check(f"l3/{backbone}:{arm}/{task}: checkpoint written", ok, saved.get("dir", "none"))
        if ok:
            st = torch.load(d / "state.pt", map_location="cpu", weights_only=False)
            need = ("adapters", "head", "feature_mu", "feature_sd", "config", "split")
            gaps = [k for k in need if k not in st]
            check(f"l3/{backbone}:{arm}/{task}: checkpoint is self-sufficient", not gaps,
                  f"missing {gaps}" if gaps else
                  f"{len(st['adapters'])} adapter tensors + head + standardisation statistics")
            check(f"l3/{backbone}:{arm}/{task}: standardisation statistics are usable",
                  st.get("feature_mu") is not None and st.get("feature_sd") is not None
                  and bool(torch.isfinite(st["feature_sd"]).all())
                  and float(st["feature_sd"].min()) > 0,
                  f"sd range [{float(st['feature_sd'].min()):.3g}, "
                  f"{float(st['feature_sd'].max()):.3g}]"
                  if st.get("feature_sd") is not None else "absent")


SETTINGS = {"nulls": run_nulls, "baselines": run_baselines, "l0l1": run_l0l1,
         "l2": run_l2, "l3": run_l3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--tasks", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--only", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    ap.add_argument("--backbones", nargs="+", default=["bfm", "aurora"])
    ap.add_argument("--baselines", nargs="+",
                    default=["glm", "nbgam", "randomforest", "convlstm"])
    ap.add_argument("--arms", nargs="+",
                    default=["lora4", "lora1", "lora16", "vera", "vera_asshipped", "full"],
                    help="L3 presets to exercise; every one is a distinct code path")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--allow-shared", action="store_true")
    args = ap.parse_args()

    from bfm_finetune.eumon.common.resources import GPUBusy, require_exclusive_gpu

    try:
        state = require_exclusive_gpu(args.gpu, allow_shared=args.allow_shared)
    except GPUBusy as exc:
        raise SystemExit(f"preflight refuses to start — {exc}")
    print(f"GPU {args.gpu} preflight: exclusive={state['exclusive']}")

    split = Split(tuple(range(2000, 2020)), 2020)
    for task in args.tasks:
        for rung in args.only:
            print(f"\n=== {rung} | task {task} ===", flush=True)
            try:
                SETTINGS[rung](task, split, args)
            except Exception as exc:
                import traceback

                traceback.print_exc()
                check(f"{rung}/{task}: rung completed", False, f"{type(exc).__name__}: {exc}"[:160])

    failed = [c for c in CHECKS if not c[1]]
    print("\n" + "=" * 78)
    print(f"PREFLIGHT: {len(CHECKS) - len(failed)}/{len(CHECKS)} passed")
    for name, _, detail in failed:
        print(f"  FAIL  {name} — {detail}")
    from bfm_finetune.eumon.common.runner import write_json

    write_json(ARTE / "preflight.json",
               {"checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in CHECKS],
                "n_failed": len(failed), "args": vars(args)})
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
