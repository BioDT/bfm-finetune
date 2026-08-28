#!/usr/bin/env python
"""Re-create every experiment in the benchmark, stage by stage.

Each stage is a ``Runner`` step: it declares its outputs, is skipped when they already
exist with a matching SHA-256, and can be re-run safely after a crash. Nulls are computed
before any model score by construction: ``gates`` precedes ``ladder`` in the stage order.

    scripts/run.py all                  # everything, in dependency order
    scripts/run.py panels gates         # named stages only
    scripts/run.py ladder --gpu 1       # a GPU stage on a chosen device
    scripts/run.py --list               # what stages exist and what they write
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EUMON_ROOT = Path(os.environ.get("EUMON_ROOT", ROOT))
sys.path[:0] = [str(ROOT), str(EUMON_ROOT / "bfm-model")]
warnings.filterwarnings("ignore")

# Cap CPU threading before torch is imported: its defaults oversubscribe a many-core host
# once several jobs run concurrently.
_THREADS = os.environ.setdefault("EUMON_THREADS", "8")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(int(_THREADS))
torch.set_num_interop_threads(int(_THREADS))

from bfm_finetune.eumon.common.runner import (Runner, artefact_provenance, artefacts_root,
                                              atomic_path, project_root, write_json)

ARTE = artefacts_root()
DATA = project_root() / "data" / "raw"
GATES = ARTE / "gates"
BASE = ARTE / "baselines"
TABLES = ARTE / "tables"
FIGS = ARTE / "figures"
DECODED = ARTE / "decoded"

SPLIT_YEARS = ((2019, 2018), (2020, 2019))
LADDER_YEARS = list(range(2000, 2021))
SKILL_REFERENCE = {"A": "persistence", "B": "climatology", "C": "persistence"}

# Whether the head sees last year's value — the information persistence already has.
# Derived from the task's reference, not set by hand, and pre-registered.
USE_PREV_FOR_TASK = {t: ref == "persistence" for t, ref in SKILL_REFERENCE.items()}
PANEL_FOR = {"A": "panel_A", "B": "panel_B", "C": "panel_C_cell"}


def use_prev_for(task: str, args) -> bool:
    """Per-task AR feature, overridable only by an explicit flag."""
    if getattr(args, "force_prev", None) is not None:
        return bool(args.force_prev)
    return USE_PREV_FOR_TASK.get(task, False)


def splits():
    from bfm_finetune.eumon.eval.nulls import Split

    return [Split(tuple(range(2000, test)), test) for test, _ in SPLIT_YEARS]


def panel(task: str) -> pd.DataFrame:
    return pd.read_parquet(ARTE / f"{PANEL_FOR[task]}.parquet")


# -- stages ----------------------------------------------------------------

def stage_panels(runner: Runner, args) -> None:
    from bfm_finetune.eumon.panel import aggregate_to_cell, to_target, write_panel, write_target
    from bfm_finetune.eumon.parsers import (load_ukbms_locations, parse_task_a, parse_task_b,
                                            parse_task_c)

    def build_a():
        p, info = parse_task_a(DATA / "task_a/dwca")
        stats = write_panel(p, ARTE / "panel_A.parquet")
        write_target(to_target(p), ARTE / "target_A.npz")
        artefact_provenance(ARTE / "panel_A.parquet",
                            sources=[{"task": "A", "url": "https://www.gbif.se/ipt/archive.do?r=lu_sft_std",
                                      "licence": "CC0-1.0"}], extra={"parser": info, "stats": stats})
        return {"rows": stats["rows"], "units": stats["n_units"]}

    def build_b():
        p, info = parse_task_b(DATA / "task_b/dwca")
        stats = write_panel(p, ARTE / "panel_B.parquet")
        write_target(to_target(p), ARTE / "target_B.npz")
        artefact_provenance(ARTE / "panel_B.parquet",
                            sources=[{"task": "B", "url": "https://www.gbif.se/ipt/archive.do?r=forestinventory-event",
                                      "licence": "CC0-1.0"}], extra={"parser": info, "stats": stats})
        return {"rows": stats["rows"], "units": stats["n_units"]}

    def build_c():
        locs, linfo = load_ukbms_locations(
            DATA / "ukbms/sitelocs/siteloc_2024_extract/data/ukbmssitelocationdata2024.csv")
        p, info = parse_task_c(DATA / "ukbms/data/ukbmssiteindices2023.csv", locs)
        stats = write_panel(p, ARTE / "panel_C.parquet")
        write_target(to_target(p), ARTE / "target_C.npz")
        artefact_provenance(ARTE / "panel_C.parquet",
                            sources=[{"role": "indices", "doi": "10.5285/04857889-1b09-40ff-a87e-71eb6ac2e998",
                                      "licence": "OGL"},
                                     {"role": "locations", "doi": "10.5285/d7256b49-f2e3-4ae7-907d-af729610c768",
                                      "licence": "OGL"}],
                            extra={"locations": linfo, "parser": info, "stats": stats})
        return {"rows": stats["rows"], "units": stats["n_units"]}

    def build_c_cell():
        cell = aggregate_to_cell(pd.read_parquet(ARTE / "panel_C.parquet"))
        stats = write_panel(cell, ARTE / "panel_C_cell.parquet")
        write_target(to_target(cell), ARTE / "target_C_cell.npz")
        artefact_provenance(ARTE / "panel_C_cell.parquet",
                            sources=[{"derived_from": "artefacts/panel_C.parquet",
                                      "rule": "mean site index per (cell, species, year); effort = n sites"}],
                            extra={"stats": stats})
        return {"rows": stats["rows"], "cells": stats["n_units"]}

    runner.run_step("panel_A", build_a, outputs=[ARTE / "panel_A.parquet", ARTE / "target_A.npz"])
    runner.run_step("panel_B", build_b, outputs=[ARTE / "panel_B.parquet", ARTE / "target_B.npz"])
    runner.run_step("panel_C", build_c, outputs=[ARTE / "panel_C.parquet", ARTE / "target_C.npz"])
    runner.run_step("panel_C_cell", build_c_cell,
                    outputs=[ARTE / "panel_C_cell.parquet", ARTE / "target_C_cell.npz"])


def stage_gates(runner: Runner, args) -> None:
    """G0, G1, G1b, G2. Writes every null score before any model score exists."""
    from bfm_finetune.eumon import gates

    GATES.mkdir(parents=True, exist_ok=True)
    A, C_site = panel("A"), pd.read_parquet(ARTE / "panel_C.parquet")

    runner.run_step("G0_AC", lambda: gates.g0_species_overlap({"A": A, "C": C_site}, out_dir=GATES),
                    outputs=[GATES / "G0_A+C.json"])

    pub_a = pd.read_csv(DATA / "task_a/published_trends/published_index_A.csv")
    runner.run_step("G1_A", lambda: gates.g1_national_trend(A, pub_a, out_dir=GATES),
                    outputs=[GATES / "G1_A.json"])

    pub_c = pd.read_csv(DATA / "ukbms/collated_2021_extract/data/ukbms_collatedindices2021.csv",
                        dtype=str)
    pub_c = pub_c.loc[pub_c["COUNTRY"] == "UK"].rename(
        columns={"SPECIES": "species", "YEAR": "year", "COLLATED_INDEX": "index"})
    pub_c["index"] = pd.to_numeric(pub_c["index"])
    pub_c["year"] = pd.to_numeric(pub_c["year"])
    runner.run_step("G1_C", lambda: gates.g1_national_trend(C_site, pub_c, out_dir=GATES),
                    outputs=[GATES / "G1_C.json"])

    for task in ("A", "B", "C"):
        p = panel(task)
        runner.run_step(f"G1b_{task}", lambda p=p: gates.g1b_survey_coverage(p, out_dir=GATES),
                        outputs=[GATES / f"G1b_{task}.json"])

    # Task C's site-level sensitivity is written first, then archived, so the cell-level
    # run lands last and G2_C.json holds the primary record.
    def g2_c_site():
        gates.g2_nulls(C_site, splits(), out_dir=GATES)
        (GATES / "G2_C_site.json").write_text((GATES / "G2_C.json").read_text())
        return {"written": "G2_C_site.json"}

    runner.run_step("G2_C_site", g2_c_site, outputs=[GATES / "G2_C_site.json"])
    for task in ("A", "B", "C"):
        p = panel(task)
        runner.run_step(f"G2_{task}", lambda p=p: gates.g2_nulls(p, splits(), out_dir=GATES),
                        outputs=[GATES / f"G2_{task}.json"])


def stage_splits(runner: Runner, args) -> None:
    from bfm_finetune.eumon import splits as S

    def build():
        rec = {t: S.summarize(panel(t), S.spec_for(t)) for t in ("A", "B", "C")}
        write_json(ARTE / "spatial_splits.json", rec)
        return {t: rec[t]["median_nearest_train_km_across_folds"] for t in rec}

    runner.run_step("spatial_splits", build, outputs=[ARTE / "spatial_splits.json"])


def stage_baselines(runner: Runner, args) -> None:
    """Learned baselines. Requires `gates` to have written the null block first."""
    from bfm_finetune.eumon.eval.baselines import run_baseline

    BASE.mkdir(parents=True, exist_ok=True)
    if not (GATES / "G2_A.json").exists():
        raise SystemExit("run the `gates` stage first: nulls must be on disk before any model score")
    for task in ("C", "A", "B"):
        p = panel(task)
        run_baseline("glm", p, splits(), seeds=[0], out_dir=BASE, runner=runner)
        run_baseline("randomforest", p, splits(), seeds=args.seeds, out_dir=BASE, runner=runner)
        run_baseline("glm_strata", p, splits(), seeds=[0], out_dir=BASE, runner=runner)
        run_baseline("nbgam", p, splits(), seeds=[0], out_dir=BASE, runner=runner)
        run_baseline("convlstm", p, splits(), seeds=args.seeds, out_dir=BASE, runner=runner,
                     epochs=args.epochs, device=args.device)


def stage_verify_batches(runner: Runner, args) -> None:
    """Accept the BioCube batches: species order, grid axes, dtypes, completeness."""
    from bfm_finetune.eumon import batch as B
    from bfm_finetune.eumon.gates import SPECIES_TABLE

    spec = json.loads(SPECIES_TABLE.read_text())
    cfg28 = [r["gbif_key"] for r in spec["decoder_species_vars"]]
    stats28 = sorted(json.loads(B.stats_path().read_text())["species_variables"])

    def check():
        files = sorted(Path(B.BIOCUBE_DIR).glob("batch_*.pt"))
        bad, groups = [], {}
        for f in files:
            b = torch.load(f, weights_only=False, map_location="cpu")
            m = b["batch_metadata"]
            errs = []
            if [str(s) for s in m["species_list"]] != cfg28:
                errs.append("species_list != config 28 in order")
            if sorted(str(s) for s in m["species_list"]) != stats28:
                errs.append("species set != scaling statistics")
            lat, lon = np.asarray(m["latitudes"]), np.asarray(m["longitudes"])
            if len(lat) != 161 or abs(lat[0] - 32.0) > 1e-6 or lat[1] < lat[0]:
                errs.append("lat axis")
            if len(lon) != 281 or abs(lon[0] + 25.0) > 1e-6 or lon[1] < lon[0]:
                errs.append("lon axis")
            for g, v in b.items():
                if isinstance(v, dict) and g != "batch_metadata" and v:
                    groups.setdefault(g, len(v))
            if errs:
                bad.append({"file": f.name, "errors": errs})
            del b
        rec = {"dir": str(B.BIOCUBE_DIR), "n_files": len(files), "n_failed": len(bad),
               "failures": bad[:20], "variable_groups": groups, "species_order": cfg28}
        write_json(ARTE / "batches_28_verification.json", rec)
        if bad:
            raise SystemExit(f"{len(bad)} batch files failed acceptance; see the record")
        return {"n_files": len(files), "groups": groups}

    runner.run_step("verify_batches_28", check, outputs=[ARTE / "batches_28_verification.json"])


def stage_decode(runner: Runner, args) -> None:
    """One forward pass per (task, year): the pre-window pair -> decoded species field."""
    from bfm_finetune.eumon import batch as B, model as M
    from bfm_model.bfm.dataloader_monthly import batch_to_device

    DECODED.mkdir(parents=True, exist_ok=True)
    cfg = M.load_config()
    species = [str(s) for s in cfg.model.species_vars]

    def build():
        model, info = M.build(device=args.device)
        ds = B.make_dataset(cfg)
        meta = {"device": args.device, "species_order": species, "years": {}, "n_forward": 0,
                "scaling_note": "outputs stay in the statistics' scaled space; min-max scaling is "
                                "monotone per channel, so L0's Spearman is invariant and L1's two "
                                "fitted parameters absorb it"}
        for task in ("A", "C"):
            for year in LADDER_YEARS:
                out_path = DECODED / f"decoded_species_{task}_{year}.npy"
                if out_path.exists():
                    continue
                try:
                    spec_in = B.forecast_input(task, year)
                except (B.WindowUnavailable, ValueError) as exc:
                    meta["years"][f"{task}_{year}"] = {"error": str(exc)[:120]}
                    continue
                x, replaced = B.sanitise(B.load_input(spec_in["path"], ds))
                with torch.no_grad():
                    res = M.forward_with_latents(
                        model, batch_to_device(B.collate_for_model(x), args.device), batch_size=1)
                dec = res["decoded"]["species_variables"]
                arr = np.stack([dec[k][0].float().cpu().numpy() for k in species]).astype(np.float32)
                with atomic_path(out_path, suffix=".npy") as tmp:
                    np.save(tmp, arr)
                meta["years"][f"{task}_{year}"] = {
                    "input_months": spec_in["input_months"], "predicts": spec_in["predicts_month"],
                    "shape": list(arr.shape), "nan_inputs_zeroed": sum(replaced.values()),
                    "finite": bool(np.isfinite(arr).all())}
                meta["n_forward"] += 1
        meta["model"] = {k: info[k] for k in ("patch_size", "embed_dim", "species_num", "n_parameters")}
        write_json(DECODED / "decoded_manifest.json", meta)
        return {"n_forward": meta["n_forward"]}

    runner.run_step("decode_species_fields", build, outputs=[DECODED / "decoded_manifest.json"])


def stage_ladder(runner: Runner, args) -> None:
    """L0 zero-shot and L1 calibration, on Tasks A and C only (Task B has no species overlap)."""
    from bfm_finetune.eumon.eval import ladder as L

    if not (GATES / "G2_A.json").exists():
        raise SystemExit("run the `gates` stage first: nulls must exist before any model score")

    for task in ("A", "C"):
        p = panel(task)
        fields = {}
        for year in LADDER_YEARS:
            f = DECODED / f"decoded_species_{task}_{year}.npy"
            if f.exists():
                fields[year] = np.load(f)
        if not fields:
            raise SystemExit(f"no decoded fields for task {task}; run the `decode` stage")
        runner.run_step(f"ladder_{task}",
                        lambda p=p, fields=fields: L.run_ladder(p, splits(), fields, GATES),
                        outputs=[GATES / f"ladder_{task}.json"])


def stage_features(runner: Runner, args) -> None:
    """GPU: per-cell L2 features — every decoded field sampled at each task's observed cells."""
    from bfm_finetune.eumon import batch as B, model as M
    from bfm_finetune.eumon.eval.probe import decoded_cell_features, flatten_fields
    from bfm_model.bfm.dataloader_monthly import batch_to_device

    FEAT = ARTE / "features"
    FEAT.mkdir(parents=True, exist_ok=True)
    cfg = M.load_config()

    def build():
        model, _ = M.build(device=args.device)
        ds = B.make_dataset(cfg)
        written, names = 0, None
        for task in ("A", "B", "C"):
            p = panel(task)
            obs = p.loc[p["observed"]]
            for year in LADDER_YEARS:
                out_path = FEAT / f"features_bfm_{task}_{year}.npz"
                if out_path.exists():
                    continue
                cells = (obs.loc[obs["year"] == year, ["unit_id", "cell_i", "cell_j"]]
                         .drop_duplicates("unit_id").reset_index(drop=True))
                if cells.empty:
                    continue
                try:
                    spec_in = B.forecast_input(task, year)
                except (B.WindowUnavailable, ValueError):
                    continue
                x, _ = B.sanitise(B.load_input(spec_in["path"], ds))
                with torch.no_grad():
                    res = M.forward_with_latents(
                        model, batch_to_device(B.collate_for_model(x), args.device), batch_size=1)
                fields = flatten_fields(res["decoded"])
                X, names = decoded_cell_features(fields, cells["cell_i"], cells["cell_j"])
                with atomic_path(out_path, suffix=".npz") as tmp:
                    np.savez_compressed(tmp, X=X, unit_id=cells["unit_id"].to_numpy().astype(str),
                                        cell_i=cells["cell_i"].to_numpy(),
                                        cell_j=cells["cell_j"].to_numpy(),
                                        feature_names=np.array(names))
                written += 1
        write_json(FEAT / "features_manifest.json",
                   {"backbone": "bfm", "n_written": written, "n_features": len(names or []),
                    "feature_names": names or [], "device": args.device,
                    "note": "decoded output fields sampled at each task's observed cells; the "
                            "encoder latents are NOT used because G0b showed they carry no "
                            "spatial addressing"})
        return {"written": written, "n_features": len(names or [])}

    runner.run_step("features_bfm", build, outputs=[ARTE / "features/features_manifest.json"])

    if "aurora" in args.backbones:
        from bfm_finetune.eumon.eval import aurora as AU

        def build_aurora():
            model, info = AU.build(device=args.device)
            # Aurora needs raw ERA5 units, not BioCube's scaled channels.
            ds = AU.aurora_dataset(cfg)
            written, names = 0, None
            for task in ("A", "B", "C"):
                p = panel(task)
                obs = p.loc[p["observed"]]
                for year in LADDER_YEARS:
                    out_path = FEAT / f"features_aurora_{task}_{year}.npz"
                    if out_path.exists():
                        continue
                    cells = (obs.loc[obs["year"] == year, ["unit_id", "cell_i", "cell_j"]]
                             .drop_duplicates("unit_id").reset_index(drop=True))
                    if cells.empty:
                        continue
                    try:
                        spec_in = B.forecast_input(task, year)
                    except (B.WindowUnavailable, ValueError):
                        continue
                    x, _ = B.sanitise(B.load_input(spec_in["path"], ds))
                    fields = AU.forward_fields(model, AU.to_aurora_batch(x))
                    X, names = AU.cell_features(fields, cells["cell_i"], cells["cell_j"])
                    with atomic_path(out_path, suffix=".npz") as tmp:
                        np.savez_compressed(tmp, X=X,
                                            unit_id=cells["unit_id"].to_numpy().astype(str),
                                            cell_i=cells["cell_i"].to_numpy(),
                                            cell_j=cells["cell_j"].to_numpy(),
                                            feature_names=np.array(names))
                    written += 1
            write_json(FEAT / "features_aurora_manifest.json",
                       {"backbone": "aurora", "n_written": written,
                        "n_features": len(names or []), "feature_names": names or [],
                        "device": args.device, "aurora": info,
                        "asymmetry": "Aurora decodes 69 fields against BioAnalyst's 124, "
                                     "because it has no species, vegetation, land, agriculture, "
                                     "forest, redlist or edaphic outputs. The head is identical; "
                                     "the feature count is not."})
            return {"written": written, "n_features": len(names or [])}

        runner.run_step("features_aurora", build_aurora,
                        outputs=[ARTE / "features/features_aurora_manifest.json"])


def stage_l2(runner: Runner, args) -> None:
    """L2 frozen probe on the per-cell decoded features, for each backbone available."""
    from bfm_finetune.eumon.eval.finetune import loss_for
    from bfm_finetune.eumon.eval.probe import run_probe

    FEAT = ARTE / "features"
    BASE.mkdir(parents=True, exist_ok=True)
    if not (GATES / "G2_A.json").exists():
        raise SystemExit("run the `gates` stage first: nulls must exist before any model score")

    for backbone in args.backbones:
        for task in ("A", "B", "C"):
            p = panel(task)
            feats, cells = {}, {}
            for year in LADDER_YEARS:
                f = FEAT / f"features_{backbone}_{task}_{year}.npz"
                if not f.exists():
                    continue
                d = np.load(f, allow_pickle=True)
                feats[year] = d["X"]
                cells[year] = pd.DataFrame({"unit_id": d["unit_id"].astype(str),
                                            "cell_i": d["cell_i"], "cell_j": d["cell_j"]})
            if not feats:
                print(f"  no {backbone} features for task {task}; skipping")
                continue
            # `label` is separate from `backbone` on purpose: reassigning the loop variable
            # made the suffix accumulate across tasks.
            label = f"{backbone}-{args.variant}" if args.variant else backbone
            if args.feature_prefix:
                names = list(np.load(FEAT / f"features_{backbone}_{task}_{min(feats)}.npz",
                                     allow_pickle=True)["feature_names"])
                keep = [k for k, n in enumerate(names)
                        if any(str(n).startswith(pre) for pre in args.feature_prefix)]
                feats = {y: X[:, keep] for y, X in feats.items()}
                label = f"{backbone}-matched"
            runner.run_step(
                f"L2_{task}_{label}",
                lambda p=p, feats=feats, cells=cells, label=label: run_probe(
                    p, splits(), args.seeds, feats, cells, BASE, backbone=label,
                    device=args.device, val_year=2018,
                    loss=loss_for(str(p["value_type"].iloc[0]), args.loss),
                    use_prev=use_prev_for(task, args),
                    save_dir=ARTE / "checkpoints" / task),
                outputs=[BASE / f"{task}_{label}_l2_s{s}.json" for s in args.seeds])


def stage_l3(runner: Runner, args) -> None:
    """L3 fine-tunes: ``--base-arms`` on every task and backbone; ``--ablation`` adds the
    PEFT method comparison on Task A."""
    from bfm_finetune.eumon.eval.baselines import score_baseline
    from bfm_finetune.eumon.eval.finetune import PRESETS, fit_l3, loss_for

    BASE.mkdir(parents=True, exist_ok=True)
    if not (GATES / "G2_A.json").exists():
        raise SystemExit("run the `gates` stage first: nulls must exist before any model score")

    tasks = tuple(args.tasks) if args.tasks else ("A", "B", "C")
    # `full` is the ladder's capacity ceiling, not an ablation arm. Both backbones get
    # every arm, or the comparison is not like-for-like.
    jobs = [(t, b, arm) for b in args.backbones for t in tasks for arm in args.base_arms]
    if args.ablation:
        jobs += [("A", "bfm", a) for a in ("lora16", "lora1", "vera", "vera_asshipped")]
    if args.arms_only:
        jobs = [j for j in jobs if j[2] in set(args.arms_only)]
    if not jobs:
        raise SystemExit(f"no L3 jobs selected (tasks={tasks}, arms_only={args.arms_only})")

    for task, backbone, arm in jobs:
        p = panel(task)
        # A variant tag keeps a re-run under different settings from overwriting the
        # results it is meant to be compared against.
        label = f"{backbone}-{args.variant}" if args.variant else backbone
        for seed in args.seeds:
            name = f"{task}_{label}_l3_{arm}_s{seed}"
            out = BASE / f"{name}.json"

            def go(p=p, task=task, backbone=backbone, arm=arm, seed=seed, out=out,
                   label=label):
                cfg = PRESETS[arm]
                cfg.epochs = args.l3_epochs
                cfg.loss = loss_for(str(p["value_type"].iloc[0]), args.loss)
                cfg.use_prev = use_prev_for(task, args)
                record = {"baseline": f"{label}_l3_{arm}", "task": task,
                          "value_type": str(p["value_type"].iloc[0]), "splits": []}
                for split in splits():
                    fit = fit_l3(p, split, seed, cfg, device=args.device, backbone=backbone,
                                 patience=args.l3_patience,
                                 save_dir=ARTE / "checkpoints" / task,
                                 save_full=args.save_full, arm=arm)
                    record["splits"].append({
                        "split": split.as_dict(), "seed": seed, "wall_s": fit.wall_s,
                        "n_rows": int(len(fit.predictions)), "diagnostics": fit.diagnostics,
                        "metrics": score_baseline(p, split, fit, seed=seed)})
                write_json(out, record)
                return {"arm": arm, "task": task}

            runner.run_step(f"L3_{name}", go, outputs=[out])


def stage_abiotic_decode(runner: Runner, args) -> None:
    """GPU: the model's own decoded t2m/tp per month, for the passthrough null and probe.

    Target month m is predicted from the BioCube pair ending at m-1, matching the biotic
    short-lead convention.
    """
    from bfm_finetune.eumon import batch as B, model as M
    from bfm_finetune.eumon.eval.probe import flatten_fields
    from bfm_model.bfm.dataloader_monthly import batch_to_device

    out = ARTE / "abiotic"
    out.mkdir(parents=True, exist_ok=True)
    cfg = M.load_config()
    # CHELSA variable -> the decoded field that is its direct counterpart
    PAIRS = {"tas": "surface_variables.t2m", "pr": "climate_variables.tp"}

    def build():
        model, _ = M.build(device=args.device)
        ds = B.make_dataset(cfg)
        months = [(y, m) for y in range(2000, 2020) for m in range(1, 13)]
        fields = {v: np.full((len(months), 160, 280), np.nan, dtype=np.float32) for v in PAIRS}
        used = 0
        for k, (y, m) in enumerate(months):
            py, pm = (y, m - 2) if m > 2 else (y - 1, m + 10)
            path = B.batch_path(py, pm)
            if not path.exists():
                continue
            x, _ = B.sanitise(B.load_input(path, ds))
            with torch.no_grad():
                dec = M.forward_with_latents(
                    model, batch_to_device(B.collate_for_model(x), args.device), batch_size=1)["decoded"]
            flat = flatten_fields(dec)
            for var, key in PAIRS.items():
                if key in flat:
                    fields[var][k] = flat[key]
            used += 1
        for var, arr in fields.items():
            with atomic_path(out / f"decoded_{var}.npy", suffix=".npy") as tmp:
                np.save(tmp, arr)
        write_json(out / "abiotic_decode_manifest.json",
                   {"months": [f"{y}-{m:02d}" for y, m in months], "n_forward": used,
                    "pairs": PAIRS, "device": args.device,
                    "convention": "target month m decoded from the BioCube pair ending at m-1"})
        return {"n_forward": used}

    runner.run_step("abiotic_decode", build,
                    outputs=[out / "decoded_tas.npy", out / "decoded_pr.npy"])


def stage_abiotic_era5(runner: Runner, args) -> None:
    """CPU: the observed ERA5 t2m/tp per month, in CHELSA units — the transfer ceiling."""
    from bfm_finetune.eumon import batch as B
    from bfm_finetune.eumon.eval import abiotic as AB

    out = ARTE / "abiotic"
    out.mkdir(parents=True, exist_ok=True)

    def build():
        months = [(y, m) for y in range(2000, 2020) for m in range(1, 13)]
        fields = {v: np.full((len(months), 160, 280), np.nan, dtype=np.float32) for v in AB.ERA5_CHANNEL}
        missing = []
        for k, (y, m) in enumerate(months):
            for var, (group, name) in AB.ERA5_CHANNEL.items():
                try:
                    fields[var][k] = B.read_era5_month(y, m, group, name)
                except B.WindowUnavailable:
                    if var == "tas":
                        missing.append(f"{y}-{m:02d}")
        for var, arr in fields.items():
            arr = AB.to_chelsa_units(arr, var, months)
            with atomic_path(out / f"era5_{var}.npy", suffix=".npy") as tmp:
                np.save(tmp, arr.astype(np.float32))
        write_json(out / "abiotic_era5_manifest.json",
                   {"months": [f"{y}-{m:02d}" for y, m in months],
                    "channels": {k: f"{g}.{n}" for k, (g, n) in AB.ERA5_CHANNEL.items()},
                    "missing_months": missing, "units": {"tas": "K", "pr": "mm month-1"},
                    "note": "observed ERA5 at the target month, no model; the transfer ceiling"})
        return {"n_months": len(months) - len(missing)}

    runner.run_step("abiotic_era5", build,
                    outputs=[out / "era5_tas.npy", out / "era5_pr.npy"])


def stage_abiotic(runner: Runner, args) -> None:
    """CHELSA task: null battery first, then (with decoded fields present) the passthrough.

    The common window is 2000-01 to 2019-06: CHELSA v2.1 publishes `tas` to 2019-12 but
    `pr` only to 2019-06.
    """
    from bfm_finetune.eumon import batch as B, chelsa as CH
    from bfm_finetune.eumon.eval import abiotic as AB

    out = ARTE / "abiotic"
    out.mkdir(parents=True, exist_ok=True)
    split = AB.MonthlySplit(train_years=tuple(range(2000, 2015)),
                            val_years=(2015, 2016), test_years=(2017, 2018, 2019))

    def build():
        stats = json.loads(B.stats_path().read_text())
        record = {"split": split.as_dict(), "variables": {}}
        for variable in ("tas", "pr"):
            series, index = CH.load_series(variable, range(2000, 2020), out_dir=ARTE / "chelsa")
            keep = [k for k, (y, m) in enumerate(index)
                    if np.isfinite(series[k]).any()]

            # Both extra fields cover the full month axis, so they must be subset by the
            # same `keep` as the target or months would be silently mispaired.
            full_index = list(index)
            model_field = era5_field = None
            decoded = out / f"decoded_{variable}.npy"
            if decoded.exists():
                model_field = AB.denormalise(np.load(decoded), variable, stats, full_index)[keep]
            era5 = out / f"era5_{variable}.npy"
            if era5.exists():
                era5_field = np.load(era5)[keep]

            series, index = series[keep], [index[k] for k in keep]

            nulls = AB.build_nulls(series, index, split, model_field, era5_field)
            scored = AB.score(series, nulls, index, split, variable)
            scored["seasonality"] = AB.seasonality_share(series, index)
            scored["n_months_available"] = len(index)
            scored["last_month"] = f"{index[-1][0]}-{index[-1][1]:02d}"
            record["variables"][variable] = scored
        write_json(out / "abiotic_nulls.json", record)

        for variable, s in record["variables"].items():
            seas = s["seasonality"]
            print(f"\n  {variable}: {s['n_months_available']} months to {s['last_month']}, "
                  f"{s['n_test_months']} test months")
            print(f"    month-of-year R2  domain-mean {seas['domain_mean_r2_month_of_year']:.4f}"
                  f"   per-cell {seas['per_cell_r2_month_of_year']:.4f}")
            print(f"    {'null':20s}{'per-cell R2':>13s}{'domain-mean R2':>16s}")
            for name, v in s["scores"].items():
                print(f"    {name:20s}{v['per_cell_r2']:13.4f}{v['domain_mean_r2']:16.4f}")
        return {"variables": list(record["variables"])}

    runner.run_step("abiotic_nulls", build, outputs=[out / "abiotic_nulls.json"], force=True)


def stage_abiotic_features(runner: Runner, args) -> None:
    """GPU: every decoded field on the whole grid, per month, for the abiotic probe.

    Stored as float32: float16 silently destroys Aurora, whose geopotential above 400 hPa
    exceeds float16's ceiling. The manifest records which features are not everywhere
    finite, so a range problem surfaces here rather than three stages later.
    """
    from bfm_finetune.eumon import batch as B, model as M
    from bfm_finetune.eumon.eval.probe import flatten_fields
    from bfm_model.bfm.dataloader_monthly import batch_to_device

    out = ARTE / "abiotic"
    out.mkdir(parents=True, exist_ok=True)
    cfg = M.load_config()
    years = list(range(2000, 2020))

    def make(backbone: str):
        def build():
            if backbone == "aurora":
                from bfm_finetune.eumon.eval import aurora as AU

                model, info = AU.build(device=args.device)
                ds = AU.aurora_dataset(cfg)
            else:
                model, info = M.build(device=args.device)
                ds = B.make_dataset(cfg)
            # Seed the column order from a previous run, so a re-run that skips every
            # cached year still writes a manifest with its feature names.
            prior = out / f"features_{backbone}_manifest.json"
            names = json.loads(prior.read_text()).get("feature_names") or None if prior.exists() else None
            written, missing, nonfinite = 0, [], set()
            for year in years:
                path = out / f"features_{backbone}_{year}.npy"
                if path.exists():
                    written += 1
                    continue
                frames = []
                for m in range(1, 13):
                    # Target month m decoded from the pair ending at m-1, the same
                    # short-lead convention the biotic tasks use.
                    py, pm = (year, m - 2) if m > 2 else (year - 1, m + 10)
                    src = B.batch_path(py, pm)
                    if not src.exists():
                        frames.append(None)
                        missing.append(f"{year}-{m:02d}")
                        continue
                    x, _ = B.sanitise(B.load_input(src, ds))
                    if backbone == "aurora":
                        from bfm_finetune.eumon.eval import aurora as AU

                        fields = AU.forward_fields(model, AU.to_aurora_batch(x))
                    else:
                        with torch.no_grad():
                            res = M.forward_with_latents(
                                model, batch_to_device(B.collate_for_model(x), args.device),
                                batch_size=1)
                        fields = flatten_fields(res["decoded"])
                    if names is None:
                        names = sorted(fields)
                    frames.append(np.stack([fields[n] for n in names]).astype(np.float32))
                shape = next((f.shape for f in frames if f is not None), None)
                if shape is None:
                    continue
                stacked = np.stack([np.full(shape, np.nan, np.float32) if f is None else f
                                    for f in frames])
                present = np.array([f is not None for f in frames])
                finite = np.isfinite(stacked[present]).all(axis=(0, 2, 3))
                nonfinite.update(names[i] for i in np.flatnonzero(~finite))
                with atomic_path(path, suffix=".npy") as tmp:
                    np.save(tmp, stacked)
                written += 1
            write_json(out / f"features_{backbone}_manifest.json",
                       {"backbone": backbone, "years": years, "n_years_written": written,
                        "n_features": len(names or []), "feature_names": names or [],
                        "dtype": "float32", "missing_months": missing, "device": args.device,
                        "info": info,
                        "features_not_everywhere_finite": sorted(nonfinite),
                        "convention": "target month m from the BioCube pair ending at m-1"})
            if nonfinite:
                print(f"  WARNING {backbone}: {len(nonfinite)} features are not everywhere "
                      f"finite: {sorted(nonfinite)}", flush=True)
            return {"backbone": backbone, "written": written, "n_features": len(names or []),
                    "n_nonfinite_features": len(nonfinite)}

        return build

    for backbone in args.backbones:
        runner.run_step(f"abiotic_features_{backbone}", make(backbone),
                        outputs=[out / f"features_{backbone}_manifest.json"])


def stage_abiotic_probe(runner: Runner, args) -> None:
    """The rebuilt CHELSA experiment: a tiny head on frozen decoded fields, scored per cell.

    Three arms per backbone: ``climatology_only`` refits the strongest null and is the bar
    to clear; ``model_only`` uses the decoded fields alone; ``model_plus_clim`` uses both,
    and its gain over ``climatology_only`` is the model's actual contribution.
    """
    from bfm_finetune.eumon import chelsa as CH
    from bfm_finetune.eumon.eval import abiotic as AB

    out = ARTE / "abiotic"
    split = AB.MonthlySplit(train_years=tuple(range(2000, 2015)),
                            val_years=(2015, 2016), test_years=(2017, 2018, 2019))
    variables = ("tas", "pr")

    def make(backbone: str):
        def build():
            manifest = json.loads((out / f"features_{backbone}_manifest.json").read_text())
            years = manifest["years"]
            features = np.concatenate([np.load(out / f"features_{backbone}_{y}.npy")
                                       for y in years])
            index = [(y, m) for y in years for m in range(1, 13)]
            truth = {}
            for v in variables:
                series, idx = CH.load_series(v, years, out_dir=ARTE / "chelsa")
                if idx != index:
                    raise ValueError(f"CHELSA {v} month index does not match the feature index")
                truth[v] = series

            # A month is usable only where the features and both targets are all present.
            keep = [k for k in range(len(index))
                    if np.isfinite(features[k]).any()
                    and all(np.isfinite(truth[v][k]).any() for v in variables)]
            features = features[keep]
            truth = {v: truth[v][keep] for v in variables}
            index = [index[k] for k in keep]

            rows = []
            for arm in AB.PROBE_ARMS:
                for seed in args.seeds:
                    r = AB.fit_abiotic_probe(features, truth, index, split, arm,
                                             variables=variables, seed=seed, device="cpu")
                    r["backbone"] = backbone
                    rows.append(r)
                    s = r.get("scores")
                    if not s:
                        print(f"  {backbone:6s} {arm.name:18s} seed {seed}  FAILED: "
                              f"{r.get('error', 'no scores')}", flush=True)
                        continue
                    print(f"  {backbone:6s} {arm.name:18s} seed {seed}  "
                          + "  ".join(f"{v} per-cell {s[v]['per_cell_r2']:+.4f} "
                                      f"domain {s[v]['domain_mean_r2']:+.4f}" for v in variables),
                          flush=True)
            # The climatology arm is handed the month-of-year null as a feature, so a
            # correct head must reproduce that null's score; if not, every other arm's
            # number is suspect.
            check = {"tolerance": 0.05, "checked": False}
            nulls_path = out / "abiotic_nulls.json"
            if nulls_path.exists():
                nulls = json.loads(nulls_path.read_text())["variables"]
                check["checked"] = True
                for v in variables:
                    got = float(np.median([r["scores"][v]["per_cell_r2"] for r in rows
                                           if r["arm"] == "climatology_only" and r.get("scores")]))
                    want = nulls[v]["scores"]["month_of_year"]["per_cell_r2"]
                    check[v] = {"climatology_arm": round(got, 4), "month_of_year_null": round(want, 4),
                                "delta": round(got - want, 4), "passes": bool(got >= want - 0.05)}
                check["passes"] = all(check[v]["passes"] for v in variables)

            record = {"backbone": backbone, "split": split.as_dict(),
                      "n_months": len(index), "n_features": manifest["n_features"],
                      "feature_names": manifest["feature_names"], "arms": rows,
                      "head_sanity_check": check,
                      "note": "per-cell is the honest figure; domain_mean is reported only to "
                              "show what the published setup hid. The model's contribution is "
                              "model_plus_clim minus climatology_only, not the raw R2."}
            if check.get("checked"):
                verdict = "PASS" if check["passes"] else "FAIL"
                print(f"  head sanity check [{verdict}]: " + "  ".join(
                    f"{v} arm {check[v]['climatology_arm']:+.4f} vs null "
                    f"{check[v]['month_of_year_null']:+.4f}" for v in variables), flush=True)
            write_json(out / f"abiotic_probe_{backbone}.json", record)
            return {"backbone": backbone, "n_fits": len(rows)}

        return build

    for backbone in args.backbones:
        runner.run_step(f"abiotic_probe_{backbone}", make(backbone),
                        outputs=[out / f"abiotic_probe_{backbone}.json"], force=True)


def stage_abiotic_forensics(runner: Runner, args) -> None:
    """Re-run the published CHELSA experiment, then repair one defect at a time."""
    from bfm_finetune.eumon import batch as B, chelsa as CH
    from bfm_finetune.eumon.eval import abiotic as AB, abiotic_original as AO

    out = ARTE / "abiotic"

    def build():
        stats = json.loads(B.stats_path().read_text())
        variables = ("tas", "pr")
        truth, index = {}, None
        for v in variables:
            series, idx = CH.load_series(v, range(2000, 2020), out_dir=ARTE / "chelsa")
            truth[v], index = series, idx
        decoded = {v: AB.denormalise(np.load(out / f"decoded_{v}.npy"), v, stats, index)
                   for v in variables}
        era5 = {v: np.load(out / f"era5_{v}.npy") for v in variables}

        # Keep the months where every array is present, so no variant is scored on a month
        # another variant could not see.
        keep = [k for k in range(len(index))
                if all(np.isfinite(d[v][k]).any() for d in (truth, decoded, era5) for v in variables)]
        index = [index[k] for k in keep]
        truth = {v: truth[v][keep] for v in variables}
        decoded = {v: decoded[v][keep] for v in variables}
        era5 = {v: era5[v][keep] for v in variables}

        record = AO.attribution(decoded, era5, truth, index,
                                train_years=range(2000, 2015), test_years=(2017, 2018, 2019))
        record["months"] = {"n": len(index), "first": f"{index[0][0]}-{index[0][1]:02d}",
                            "last": f"{index[-1][0]}-{index[-1][1]:02d}"}
        write_json(out / "abiotic_forensics.json", record)

        print(f"\n  {len(index)} months, {record['months']['first']} to {record['months']['last']}")
        print("  published: " + "  ".join(f"{k} {v:.4f}" for k, v in AO.PUBLISHED.items()))
        print(f"\n  {'variant':22s}{'feat':>5s}{'held out':>10s}{'per cell':>10s}{'R2':>9s}")
        for r in record["variants"]:
            print(f"  {r['variant']:22s}{r['n_features']:5d}{str(r['held_out']):>10s}"
                  f"{str(r['per_cell']):>10s}{r['r2']:9.4f}"
                  + ("   (fit subsampled to "
                     f"{r['n_fit_used']:,} of {r['n_fit_available']:,} rows)" if r["subsampled"] else ""))
        return {"n_variants": len(record["variants"])}

    runner.run_step("abiotic_forensics", build, outputs=[out / "abiotic_forensics.json"],
                    force=True)


def stage_rescore(runner: Runner, args) -> None:
    """Recompute every metric from saved predictions. No GPU, no retraining."""
    from bfm_finetune.eumon.eval import metrics as M
    from bfm_finetune.eumon.eval.nulls import REFERENCES, Split, compute_nulls

    ckpt = ARTE / "checkpoints"
    if not ckpt.exists():
        raise SystemExit("no saved predictions yet; runs must be made with save_dir enabled")

    def build():
        out, cache = [], {}
        for run_dir in sorted(ckpt.glob("*/*/")):
            pred_path = run_dir / "predictions.parquet"
            meta_path = run_dir / "run.json"
            if not (pred_path.exists() and meta_path.exists()):
                continue
            meta = json.loads(meta_path.read_text())
            task = run_dir.parent.name
            frame = pd.read_parquet(pred_path)
            test_year = meta["split"]["test_year"]
            key = (task, test_year)
            if key not in cache:
                cache[key] = compute_nulls(panel(task),
                                           Split(tuple(meta["split"]["train_years"]), test_year))
            nulls = cache[key]
            k = ["unit_id", "species", "year"]
            refs = frame.loc[:, k].merge(nulls.loc[:, k + list(REFERENCES)], on=k, how="left")
            rec = M.evaluate(frame, value_type=str(panel(task)["value_type"].iloc[0]))
            rec["skill"] = {f"vs_{r}": M.skill_score(frame["y_true"], frame["y_pred"],
                                                     refs[r].to_numpy(float)) for r in REFERENCES}
            rec["per_species"] = M.per_species(frame,
                                               value_type=str(panel(task)["value_type"].iloc[0]))
            out.append({"run": run_dir.name, "task": task, **meta, "metrics": rec})
        write_json(ARTE / "rescored.json", {"n_runs": len(out), "runs": out})
        print(f"  re-scored {len(out)} runs from saved predictions, no GPU used")
        return {"n_runs": len(out)}

    runner.run_step("rescore", build, outputs=[ARTE / "rescored.json"], force=True)


def stage_compute(runner: Runner, args) -> None:
    """Aggregate the computational cost of every result into a manuscript-ready table."""
    from bfm_finetune.eumon.common.resources import aggregate, gpu_inventory

    def build():
        per_run, partial = [], []
        for f in sorted(BASE.glob("*.json")) + sorted(GATES.glob("ladder_*.json")):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            for s in d.get("splits", []) or d.get("results", []):
                g = s.get("diagnostics") or {}
                res = g.get("resources")
                label = f"{d.get('task', '?')}_{d.get('baseline', f.stem)}_s{s.get('seed', 0)}_{s.get('split', {}).get('test_year', '')}"
                if res:
                    per_run.append({**res, "label": label})
                elif s.get("wall_s"):
                    # Runs finished before the meter existed still carry wall-clock and
                    # torch peak memory; they contribute GPU-hours but no energy.
                    partial.append({"label": label, "wall_s": s["wall_s"],
                                    "gpu_hours": round(s["wall_s"] / 3600, 5),
                                    "torch_peak_alloc_gib": g.get("peak_gib"),
                                    "energy_kwh": None, "gpu_util_mean_pct": None,
                                    "gpu_mem_peak_mib": None, "shared_gpu": None})
        rec = {"metered": aggregate(per_run),
               "unmetered_but_timed": {"n_runs": len(partial),
                                       "total_gpu_hours": round(sum(r["gpu_hours"] for r in partial), 4)},
               "hardware": gpu_inventory(),
               "per_run": per_run + partial,
               "note": "utilisation and power are per GPU, not per process; runs marked "
                       "shared_gpu ran alongside another process and their energy is an upper "
                       "bound. Runs listed as unmetered contribute wall-clock only."}
        write_json(ARTE / "compute_cost.json", rec)
        m, u = rec["metered"], rec["unmetered_but_timed"]
        print(f"  metered runs {m.get('n_runs', 0)}: {m.get('total_gpu_hours', 0)} GPU-h, "
              f"{m.get('total_energy_kwh', 0)} kWh, {m.get('total_co2e_kg', 0)} kg CO2e, "
              f"mean util {m.get('mean_gpu_util_pct')}%")
        print(f"  timed-only runs {u['n_runs']}: {u['total_gpu_hours']} GPU-h (no energy)")
        return {"metered": m.get("n_runs", 0), "timed_only": u["n_runs"]}

    runner.run_step("compute_cost", build, outputs=[ARTE / "compute_cost.json"], force=True)


def stage_tables(runner: Runner, args) -> None:
    from bfm_finetune.eumon.common import tables as T

    TABLES.mkdir(parents=True, exist_ok=True)
    GLM_NOTE = ("as specified this is climatology rescaled by a fitted per-species year term, so "
                "its spatial ranking is climatology's by construction (delta rho < 1e-5 on all "
                "three tasks) and it differs only in magnitude. Not an independent competitor.")

    def baselines_for(task, year):
        out = {}
        for f in sorted(BASE.glob(f"{task}_*.json")):
            if "superseded" in f.name:
                continue
            d = json.loads(f.read_text())
            for s in d["splits"]:
                if s["split"]["test_year"] == year:
                    out.setdefault(d["baseline"], []).append(s["metrics"])
        return out

    def model_for(task, year):
        out = {}
        # Task B has no ladder file at all (no species overlap); its L2/L3 rows must still
        # be collected, so no early return on that absence.
        f = GATES / f"ladder_{task}.json"
        if f.exists():
            d = json.loads(f.read_text())
            for r in d["results"]:
                if r["split"]["test_year"] == year:
                    for rung, key in (("L0", "bfm_l0"), ("L1", "bfm_l1")):
                        if r.get(rung, {}).get("n_species"):
                            out[key] = [r[rung]]
        for f in sorted(BASE.glob(f"{task}_*_l3_*_s*.json")):
            d = json.loads(f.read_text())
            for s in d["splits"]:
                if s["split"]["test_year"] == year:
                    out.setdefault(d["baseline"], []).append(s["metrics"])
        for backbone in ("bfm", "aurora"):
            recs = []
            for f in sorted(BASE.glob(f"{task}_{backbone}_l2_s*.json")):
                for s in json.loads(f.read_text())["splits"]:
                    if s["split"]["test_year"] == year:
                        recs.append(s["metrics"])
            if recs:
                out[f"{backbone}_l2"] = recs
        return out

    def build():
        built = []
        for task in ("A", "B", "C"):
            rec = json.loads((GATES / f"G2_{task}.json").read_text())
            for s in rec["splits"]:
                year = s["split"]["test_year"]
                blocks = {"nulls": {k: [v] for k, v in s["scores"].items() if v.get("n", 0) > 0},
                          "baselines": baselines_for(task, year),
                          "model": model_for(task, year)}
                # Every predictor on disk must be accounted for: a result not named in a
                # block renders as nothing, silently.
                known = set(T.NULL_BLOCK) | set(T.BASELINE_BLOCK) | set(T.MODEL_BLOCK)
                found = set(blocks["baselines"]) | set(blocks["model"])
                on_disk = {json.loads(p.read_text())["baseline"]
                           for p in BASE.glob(f"{task}_*.json")}
                missing = sorted(n for n in on_disk & known
                                 if n not in found and n in set(T.MODEL_BLOCK))
                if missing:
                    raise SystemExit(
                        f"task {task} {year}: {missing} have result files but produced no "
                        f"table row — a collection step is dropping them.")
                orphan = sorted(found - known)
                if orphan:
                    raise SystemExit(
                        f"task {task} {year}: {orphan} have results on disk but appear in no "
                        f"table block, so the table would omit them silently. Add them to "
                        f"tables.py, or move superseded results out of {BASE}.")
                built.append(T.task_table(task, blocks, reference=SKILL_REFERENCE[task],
                                          ceiling=s.get("cell_resolution_ceiling"),
                                          test_year=year, row_notes={"glm": GLM_NOTE}))
        return {"paths": T.write_tables(built, TABLES), "n_tables": len(built)}

    runner.run_step("tables", build, outputs=[TABLES / "tables_all_tasks.md"], force=True)


def stage_figures(runner: Runner, args) -> None:
    from bfm_finetune.eumon.common import viz

    FIGS.mkdir(parents=True, exist_ok=True)
    panels = {t: panel(t) for t in ("A", "B", "C")}
    src = [{"artefact": str(ARTE / f"{PANEL_FOR[t]}.parquet")} for t in ("A", "B", "C")]
    runner.run_step("figure_G", lambda: viz.figure_g(panels, FIGS, sources=src),
                    outputs=[FIGS / "figure_G_survey_coverage.pdf",
                             FIGS / "figure_G_survey_coverage.png"])
    # Task A alone for the main text; the three-task versions stay in the appendix.
    runner.run_step("figure_G_A",
                    lambda: viz.figure_g({"A": panels["A"]}, FIGS,
                                         sources=src[:1],
                                         name="figure_G_survey_coverage_A"),
                    outputs=[FIGS / "figure_G_survey_coverage_A.pdf",
                             FIGS / "figure_G_survey_coverage_A.png"])

    tables = [json.loads(f.read_text()) for f in sorted(TABLES.glob("table_*.json"))]
    published = viz.load_published()
    for name, fn, out in (
        ("figure_D", lambda: viz.figure_d(tables, FIGS,
                                          sources=[{"from": str(TABLES / "table_*.json")}]),
         "figure_D_null_vs_model"),
        ("figure_D_A", lambda: viz.figure_d(tables, FIGS, tasks=("A",),
                                            name="figure_D_null_vs_model_A",
                                            sources=[{"from": str(TABLES / "table_A_*.json")}]),
         "figure_D_null_vs_model_A"),
        ("figure_E", lambda: viz.figure_e(panels, FIGS), "figure_E_calibration"),
        ("figure_B", lambda: viz.figure_b(panels, FIGS), "figure_B_skill_map"),
        ("figure_F", lambda: viz.figure_f(panels, FIGS), "figure_F_species_skill"),
        ("figure_E_A", lambda: viz.figure_e(panels, FIGS, tasks=("A",),
                                            name="figure_E_calibration_A"),
         "figure_E_calibration_A"),
        ("figure_B_A", lambda: viz.figure_b(panels, FIGS, tasks=("A",),
                                            name="figure_B_skill_map_A"),
         "figure_B_skill_map_A"),
        ("figure_F_A", lambda: viz.figure_f(panels, FIGS, tasks=("A",),
                                            name="figure_F_species_skill_A"),
         "figure_F_species_skill_A"),
        ("figure_C", lambda: viz.figure_c(panels, FIGS, published), "figure_C_published_trend"),
        ("figure_A", lambda: viz.figure_a(panels, FIGS, "A"), "figure_A_trajectories_A"),
    ):
        runner.run_step(name, fn, outputs=[FIGS / f"{out}.pdf"], force=True)


def stage_provenance(runner: Runner, args) -> None:
    from bfm_finetune.eumon import download

    runner.run_step("provenance",
                    lambda: {"path": str(download.write_provenance(artefacts_root=ARTE))},
                    outputs=[ARTE / "provenance.json"], force=True)


STAGES = {
    "panels": (stage_panels, "parse the three archives into panel_{A,B,C}[_cell].parquet"),
    "gates": (stage_gates, "G0, G1, G1b, G2 — writes all null scores before any model score"),
    "splits": (stage_splits, "spatial block holdout assignment and separation diagnostics"),
    "baselines": (stage_baselines, "TRIM GLM, TRIM+covariate, NB-GAM, RandomForest, ConvLSTM "
                                   "(needs gates)"),
    "verify_batches": (stage_verify_batches, "accept the 28-species BioCube batches"),
    "decode": (stage_decode, "GPU: one forward pass per (task, year) -> decoded species fields"),
    "ladder": (stage_ladder, "L0 zero-shot and L1 calibration (needs decode + gates)"),
    "features": (stage_features, "GPU: per-cell L2 features from the decoded fields"),
    "l2": (stage_l2, "L2 frozen probe, per backbone (needs features + gates)"),
    "l3": (stage_l3, "L3 fine-tunes: --base-arms on every task and backbone, plus "
                     "--ablation for the PEFT method comparison on Task A"),
    "abiotic_decode": (stage_abiotic_decode, "GPU: decoded t2m/tp per month for the CHELSA task"),
    "abiotic_era5": (stage_abiotic_era5, "CPU: observed ERA5 t2m/tp per month, the transfer ceiling"),
    "abiotic": (stage_abiotic, "CHELSA task: null battery (CPU) and the passthrough null"),
    "abiotic_features": (stage_abiotic_features,
                         "GPU: whole-grid decoded fields per month, the abiotic probe's features"),
    "abiotic_probe": (stage_abiotic_probe,
                      "CHELSA: the rebuilt per-cell probe on frozen decoded fields"),
    "abiotic_forensics": (stage_abiotic_forensics,
                          "CHELSA: re-run the published experiment and attribute its number"),
    "rescore": (stage_rescore, "recompute all metrics from saved predictions (no GPU)"),
    "compute": (stage_compute, "aggregate GPU-hours, energy and CO2e for the manuscript"),
    "tables": (stage_tables, "per-task result tables; refuses a pooled number"),
    "figures": (stage_figures, "result figures"),
    "provenance": (stage_provenance, "consolidated provenance.json for the artefacts root"),
}
# Stages that put work on a GPU; each is gated on the card being free.
GPU_STAGES = {"baselines", "decode", "features", "l2", "l3", "abiotic_decode",
              "abiotic_features"}

ORDER = ["panels", "gates", "splits", "baselines", "verify_batches", "decode", "ladder",
         "features", "l2", "l3", "abiotic_decode", "abiotic_era5", "abiotic_features",
         "abiotic", "abiotic_probe", "abiotic_forensics", "rescore", "compute", "tables",
         "figures", "provenance"]

# A stage present in STAGES but missing from ORDER could never be selected; fail at import.
if set(ORDER) != set(STAGES):
    raise RuntimeError(f"ORDER and STAGES disagree: only in ORDER {sorted(set(ORDER) - set(STAGES))}, "
                       f"only in STAGES {sorted(set(STAGES) - set(ORDER))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", help="stage names, or 'all'")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--run-id", default="benchmark", help="runs/<id>/state.json to resume")
    ap.add_argument("--gpu", type=int, default=1,
                    help="CUDA device index for GPU stages (default 1)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=200, help="ConvLSTM epochs")
    ap.add_argument("--feature-prefix", nargs="+", default=None,
                    help="restrict L2 features to these prefixes, e.g. surface_ atmospheric_ "
                         "(the matched-variable control against Aurora)")
    ap.add_argument("--l3-epochs", type=int, default=150)
    ap.add_argument("--l3-patience", type=int, default=25,
                    help="epochs without validation improvement before stopping; scale it "
                         "with --l3-epochs")
    ap.add_argument("--ablation", action="store_true",
                    help="add the Task A PEFT method ablation (lora1, lora16, vera, "
                         "vera_asshipped); `full` is a ladder rung, not an ablation arm")
    ap.add_argument("--tasks", nargs="+", default=None, help="restrict L3 to these tasks")
    ap.add_argument("--save-full", action="store_true",
                    help="also store the full ~2.7 GB state dict for the full fine-tune arm")
    ap.add_argument("--arms-only", nargs="+", default=None,
                    help="restrict L3 to these arms, e.g. lora16 lora1 vera_asshipped full")
    ap.add_argument("--allow-shared", action="store_true",
                    help="run even if another process holds the GPU; resource figures "
                         "then become an upper bound and are flagged as such")
    ap.add_argument("--base-arms", nargs="+", default=["vera"],
                    help="L3 arms run on every task, e.g. lora4 full")
    ap.add_argument("--variant", default="",
                    help="tag appended to the backbone label so a re-run under new "
                         "settings does not overwrite what it is compared against")
    ap.add_argument("--loss", default="auto", choices=["auto", "log1p_mse", "poisson"],
                    help="L2/L3 head loss; auto picks from the target type")
    ap.add_argument("--use-prev", dest="force_prev", action="store_true", default=None,
                    help="force the AR feature on for every task, overriding USE_PREV_FOR_TASK")
    ap.add_argument("--no-use-prev", dest="force_prev", action="store_false",
                    help="force the AR feature off for every task")
    ap.add_argument("--backbones", nargs="+", default=["bfm", "aurora"],
                    help="backbones to run at L2/L3")
    args = ap.parse_args()
    args.device = f"cuda:{args.gpu}"

    if args.list or not args.stages:
        print(f"{'stage':16s} description")
        for name in ORDER:
            print(f"  {name:14s} {STAGES[name][1]}")
        print(f"\nGPU stages default to cuda:{args.gpu}. Stages are idempotent; re-running is cheap.")
        return

    wanted = ORDER if "all" in args.stages else [s for s in ORDER if s in args.stages]
    unknown = set(args.stages) - set(STAGES) - {"all"}
    if unknown:
        raise SystemExit(f"unknown stage(s): {sorted(unknown)}")

    gpu_wanted = sorted(set(wanted) & GPU_STAGES)
    if gpu_wanted:
        from bfm_finetune.eumon.common.resources import GPUBusy, require_exclusive_gpu
        try:
            state = require_exclusive_gpu(args.gpu, allow_shared=args.allow_shared)
        except GPUBusy as exc:
            raise SystemExit(f"refusing to start {gpu_wanted} — {exc}")
        print(f"GPU {args.gpu} preflight: exclusive={state['exclusive']}"
              + (f" (sharing accepted: {state['occupants']})" if state["occupants"] else ""))

    runner = Runner(run_id=args.run_id, phase="benchmark")
    try:
        for name in wanted:
            print(f"\n=== {name} ===", flush=True)
            STAGES[name][0](runner, args)
        print("\n" + runner.summary())
    finally:
        runner.close()


if __name__ == "__main__":
    main()
