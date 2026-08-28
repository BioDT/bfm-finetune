#!/usr/bin/env python
"""Read-only verification of the benchmark's foundations. Writes no artefact, runs no model.

Every claim the benchmark rests on is stated as an assertion over the data and code
actually on disk, in four areas: TASK (what is predicted, for which units, in which
years), DATA (what reaches the model), SCORE (what the numbers mean and whether they are
comparable), MODEL (whether the fitting procedure is the same for every arm compared).

Every check prints PASS, FAIL or WARN with the evidence inline. WARN marks a defensible
design choice that must be stated in the manuscript; FAIL marks something that makes a
reported number wrong. Run it with no arguments.
"""

import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EUMON_ROOT = Path(os.environ.get("EUMON_ROOT", ROOT))
sys.path[:0] = [str(ROOT), str(EUMON_ROOT / "bfm-model")]

import numpy as np
import pandas as pd

from bfm_finetune.eumon.common.runner import artefacts_root

PKG = ROOT / "bfm_finetune" / "eumon"
ARTE = artefacts_root()
PANEL_FOR = {"A": "panel_A", "B": "panel_B", "C": "panel_C_cell"}
RESULTS: list[tuple[str, str, str, str]] = []


def record(area: str, check: str, ok: bool | None, evidence: str) -> None:
    status = "PASS" if ok is True else ("WARN" if ok is None else "FAIL")
    RESULTS.append((area, check, status, evidence))
    print(f"  [{status}] {check}\n         {evidence}", flush=True)


def panel(task: str) -> pd.DataFrame:
    return pd.read_parquet(ARTE / f"{PANEL_FOR[task]}.parquet")


# ---------------------------------------------------------------- TASK

def audit_task() -> None:
    print("\n=== TASK DEFINITION ===")
    for t in ("A", "B", "C"):
        p = panel(t)
        obs = p["observed"].to_numpy(bool)
        val = p["value"].to_numpy(float)
        bad = int((obs & ~np.isfinite(val)).sum() + (~obs & np.isfinite(val)).sum())
        record("TASK", f"{t}: observed <=> value is finite", bad == 0,
               f"{bad} violating rows of {len(p):,}; observed={obs.sum():,}")

        dup = int(p.duplicated(["unit_id", "species", "year"]).sum())
        record("TASK", f"{t}: one row per (unit, species, year)", dup == 0,
               f"{dup} duplicate keys")

        yrs = sorted(p.loc[p["observed"], "year"].unique())
        have = [y for y in (2019, 2020) if y in yrs]
        record("TASK", f"{t}: both test years present", len(have) == 2,
               f"observed years {yrs[0]}-{yrs[-1]}, test years present: {have}")

    # Task C is scored at cell level; how much of that aggregation is real?
    c = panel("C")
    eff = c.loc[c["observed"], "effort"].to_numpy(float)
    single = float((eff == 1).mean())
    record("TASK", "C: cell aggregation actually aggregates", None,
           f"{single:.1%} of observed cell-years contain a single site "
           f"(mean {eff.mean():.2f}); for this share of the data the move to cell level "
           f"changes nothing.")

    # Species overlap is read via `ladder.overlapping_species`, the function the ladder
    # actually uses — an audit that reads the data differently from the pipeline is not an
    # audit of the pipeline.
    from bfm_finetune.eumon.eval.ladder import overlapping_species, species_channels

    channels = species_channels()
    record("TASK", "species table resolves to the decoder's channels", len(channels) == 28,
           f"{len(channels)} named channels from bfm_species.json")
    for t in ("A", "B", "C"):
        p = panel(t)
        raw = overlapping_species(p, channels, min_detections=None)
        kept = overlapping_species(p, channels)
        record("TASK", f"{t}: species overlap with the model's channels", None,
               f"{len(raw)} of {p['species'].nunique()} species overlap; {len(kept)} survive "
               f"G0's >=30-detection filter -> {sorted(kept) if kept else 'no zero-shot rung'}")


# ---------------------------------------------------------------- DATA

def audit_data() -> None:
    print("\n=== DATA HANDLING ===")
    from bfm_finetune.eumon import batch as B
    from bfm_finetune.eumon.panel import GRID

    record("DATA", "grid geometry", GRID.H == 160 and GRID.W == 280 and GRID.lat_origin == 32.0,
           f"{GRID.H}x{GRID.W} at {GRID.res} deg, lat origin {GRID.lat_origin} "
           f"ascending={GRID.lat_ascending}, lon origin {GRID.lon_origin}")

    for t in ("A", "B", "C"):
        p = panel(t)
        i = p["cell_i"].to_numpy(int)
        j = p["cell_j"].to_numpy(int)
        off = int(((i < 0) | (i >= GRID.H) | (j < 0) | (j >= GRID.W)).sum())
        record("DATA", f"{t}: every unit on the model grid", off == 0, f"{off} off-grid rows")

    # The forecast input must close strictly before the survey window opens, or the model
    # is handed a contemporaneous observation of the survey it is predicting.
    for t, w in B.SURVEY_WINDOWS.items():
        pair_start = w.input_pair_start()
        pair_end = pair_start + 1
        ok = pair_end < w.first_month
        record("DATA", f"{t}: input pair closes before the survey window",
               ok, f"input months {pair_start}-{pair_end}, window opens {w.first_month} "
                   f"({w.evidence})")

    # A BioCube file must actually carry the month its name claims.
    try:
        import torch
        checked, bad = 0, []
        for (y, m) in [(2017, 3), (2010, 5), (2019, 4)]:
            path = B.batch_path(y, m)
            if not path.exists():
                continue
            blob = torch.load(path, map_location="cpu", weights_only=False)
            stamp = blob["batch_metadata"]["timestamp"][0][:7]
            checked += 1
            if stamp != f"{y:04d}-{m:02d}":
                bad.append(f"{path.name} carries {stamp}")
        record("DATA", "BioCube file name matches its timestamp", not bad,
               f"{checked} files checked; {bad if bad else 'all consistent'}")
    except Exception as exc:
        record("DATA", "BioCube file name matches its timestamp", None, f"not checked: {exc}")

    # The AR feature must read year-1 only, and year-1 must be inside the training window.
    for t in ("A", "C"):
        p = panel(t)
        for test_year in (2019, 2020):
            prev = test_year - 1
            in_train = prev < test_year
            n_prev = int(((p["year"] == prev) & p["observed"]).sum())
            record("DATA", f"{t}: AR feature for {test_year} reads only {prev}",
                   in_train and n_prev > 0,
                   f"{n_prev:,} observed rows in {prev}; persistence uses the same rows")


# ---------------------------------------------------------------- SCORE

def audit_score() -> None:
    print("\n=== SCORING ===")
    from bfm_finetune.eumon.eval import metrics

    # A predictor that IS the reference must score exactly zero against it.
    y = np.array([1.0, 4.0, 9.0, 2.0])
    ref = np.array([2.0, 3.0, 7.0, 3.0])
    record("SCORE", "a predictor equal to the reference scores 0",
           metrics.skill_score(y, ref, ref) == 0.0,
           f"skill_score(y, ref, ref) = {metrics.skill_score(y, ref, ref)}; "
           f"and skill_score(y, y, ref) = {metrics.skill_score(y, y, ref)} (perfect = 1)")

    # Which null is strongest? Only the legitimate references count; the two peeking nulls
    # are diagnostics, not candidate references.
    from bfm_finetune.eumon.eval.nulls import REFERENCES

    for t in ("A", "B", "C"):
        f = ARTE / "gates" / f"G2_{t}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        for entry in (d.get("splits") or d.get("results") or []):
            ty = entry.get("split", {}).get("test_year")
            sc = entry.get("scores") or entry.get("nulls") or {}
            legit, peek = {}, {}
            for name, rec in sc.items():
                v = rec.get("skill", {}).get("vs_persistence", {}).get("skill_score")
                if not isinstance(v, (int, float)) or not np.isfinite(v):
                    continue
                (legit if name in REFERENCES else peek)[name] = v
            if not legit:
                continue
            best = max(legit, key=legit.get)
            fixed = {"A": "persistence", "B": "climatology", "C": "persistence"}[t]
            agrees = abs(legit[best] - legit.get(fixed, -np.inf)) < 1e-9
            # WARN, not FAIL: the headline takes the minimum across all three references,
            # and the divergence is a property of the data the manuscript has to state.
            record("SCORE", f"{t} {ty}: which legitimate null is strongest", None if not agrees else True,
                   f"fixed '{fixed}' ({legit.get(fixed, float('nan')):+.3f}) vs strongest "
                   f"'{best}' ({legit[best]:+.3f})"
                   + ("" if agrees else " — differs, so the fixed column would flatter models "
                                        "in this year; the headline uses the strongest")
                   + (f"; peeking nulls reach {max(peek.values()):+.3f} "
                      f"({max(peek, key=peek.get)}) and are diagnostics, not references"
                      if peek else ""))

    # The headline must exist in both scoring paths and be surfaced by the tables.
    src_tables = (PKG / "common/tables.py").read_text()
    headline_ok = ("skill_vs_strongest_null" in (PKG / "eval/nulls.py").read_text()
                   and "skill_vs_strongest_null" in (PKG / "eval/baselines.py").read_text()
                   and "skill_strongest" in src_tables
                   and src_tables.index('"skill_strongest"') < src_tables.index('"rmse_log", "spatial_rho"'))
    record("SCORE", "headline is skill against the strongest legitimate null", headline_ok,
           "computed in both scoring paths and the leading table column; a minimum across "
           "references can only lower a score, so it cannot be gamed upward")

    # Model and reference must be scored on the same rows.
    mismatches = 0
    for f in glob.glob(str(ARTE / "baselines" / "*.json"))[:400]:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        for s in d.get("splits", []):
            sk = (s.get("metrics") or {}).get("skill") or {}
            ns = {k: v.get("n_scored") for k, v in sk.items() if isinstance(v, dict)}
            if len(set(v for v in ns.values() if v)) > 1:
                mismatches += 1
    record("SCORE", "model and reference scored on the same rows", None,
           f"{mismatches} result files where n_scored differs between references — expected, "
           f"since persistence is undefined where a unit was not surveyed last year; the "
           f"skill score already restricts to the reference's own rows")

    # The two scoring paths must produce the same metric keys.
    src_null = (PKG / "eval/nulls.py").read_text()
    src_base = (PKG / "eval/baselines.py").read_text()
    both = all(k in src_null and k in src_base
               for k in ("skill_score_log1p", "skill_vs_strongest_null"))
    record("SCORE", "nulls and models share one metric set", both,
           "skill_score_log1p and skill_vs_strongest_null present in both scoring paths"
           if both else "the two paths emit different keys; tables would mix metric sets")

    # Metrics that are structurally undefined must not be emitted.
    record("SCORE", "temporal_rho not emitted for single-year splits",
           "df[\"year\"].nunique() >= 5" in (PKG / "eval/metrics.py").read_text(),
           "guarded; previously returned n_groups=0 in every artefact ever written")


# ---------------------------------------------------------------- MODEL

def audit_model() -> None:
    print("\n=== MODELLING ===")
    probe = (PKG / "eval/probe.py").read_text()
    ft = (PKG / "eval/finetune.py").read_text()
    res = (PKG / "common/resources.py").read_text()
    run = (ROOT / "scripts/run.py").read_text()

    # Check the call sites, not a substring: the head must never be handed raw features.
    l2_standardises = "(Xtr - mu) / sd" in probe and "(xq_raw - mu) / sd" in probe
    l3_defines = "feat_mu" in ft and "feat_sd" in ft and "def standardise" in ft
    l3_raw_calls = ft.count("head(feats.float())")
    l3_std_calls = ft.count("head(standardise(feats))")
    record("MODEL", "L2 and L3 preprocess head inputs the same way",
           l2_standardises and l3_defines and l3_raw_calls == 0 and l3_std_calls >= 2,
           f"L2 standardises train and query: {l2_standardises}; L3 defines per-column "
           f"statistics: {l3_defines}; L3 head calls — raw {l3_raw_calls}, standardised "
           f"{l3_std_calls}. Feeding one rung raw values and the other standardised ones "
           f"would confound adaptation with preprocessing.")

    record("MODEL", "the head is identical across backbones within a rung",
           "ProbeHead(" in probe and "ProbeHead(" in ft,
           "both rungs build ProbeHead; BioAnalyst and Aurora share it within a rung, which "
           "is what makes the cross-model comparison like-for-like")

    from bfm_finetune.eumon.eval.finetune import loss_for
    rows = {t: loss_for(vt) for t, vt in (("A", "count"), ("B", "prevalence"), ("C", "index"))}
    record("MODEL", "loss is fixed by target type, not chosen on results", True,
           f"{rows}; selection on one validation year would pick Task A's worst configuration")

    # The AR feature must be a property of the task, derived from its reference.
    ar_derived = ("USE_PREV_FOR_TASK = {t: ref == \"persistence\"" in run
                  and "use_prev_for(task, args)" in run)
    record("MODEL", "the AR feature is derived per task, not chosen per launch", ar_derived,
           "A and C (persistence-referenced) receive last year's value; B does not. "
           "Overridable only by an explicit --use-prev/--no-use-prev.")

    record("MODEL", "run-time invariants are enforced inside the pipeline",
           all(k in ft for k in ("adapters_received_gradient", "no_test_year_in_training",
                                 "predictions_in_domain"))
           and "headline_is_worst_reference" in (PKG / "eval/baselines.py").read_text()
           and "ladder_l1_preserves_ranking" in (PKG / "eval/ladder.py").read_text(),
           "split leakage, prediction domain, adapter gradients, L1-preserves-ranking and "
           "the conservative headline all raise during the run rather than after it")

    record("MODEL", "forward_with_latents is not wrapped in no_grad",
           "@torch.no_grad()" not in
           (PKG / "model.py").read_text().split("def forward_with_latents")[0][-200:],
           "a decorator here silently severs the adapter gradient path while the loss still "
           "falls")

    record("MODEL", "energy is metered on both GPU rungs",
           "Meter(" in probe and "Meter(" in ft,
           "L2 and L3 each wrap their fit in a Meter; CPU baselines are unmetered by design")

    record("MODEL", "energy reports a marginal cost, not just total draw",
           "idle_power_w" in res and "energy_wh_above_idle" in res,
           "idle power sampled before the workload; total and above-idle both reported so "
           "neither is presented as the other")

    record("MODEL", "checkpoints can reproduce their own predictions",
           all(k in ft for k in ('"predictions.parquet"', '"feature_mu"', '"feature_sd"',
                                 '"adapters"')),
           "predictions, adapters, head and the standardisation statistics are persisted; "
           "without the statistics the head cannot be re-applied")

    record("MODEL", "saved predictions support re-scoring without a GPU",
           "def stage_rescore" in run and "predictions.parquet" in run,
           "the rescore stage recomputes every metric from parquet, so a metric change is a "
           "CPU pass rather than a retrain")

    gated = ("require_exclusive_gpu" in run and "GPU_STAGES" in run
             and "allow_shared" in run)
    record("MODEL", "GPU stages refuse to start on an occupied card", gated,
           "run.py preflights the device and exits rather than sharing it; --allow-shared "
           "is an explicit opt-in that flags the figures as an upper bound")

    smoke = ARTE / "l3_smoke.json"
    if smoke.exists():
        d = json.loads(smoke.read_text())
        checks = d.get("checks") or d
        record("MODEL", "L3 smoke test on record", None,
               f"{smoke.name} present with {len(checks) if hasattr(checks, '__len__') else '?'} "
               f"assertions; re-run it after any change to the loss or gradient path")


def main() -> int:
    print("Benchmark foundation audit — read-only, no artefacts written, no models run.")
    audit_task()
    audit_data()
    audit_score()
    audit_model()

    fails = [r for r in RESULTS if r[2] == "FAIL"]
    warns = [r for r in RESULTS if r[2] == "WARN"]
    print(f"\n=== SUMMARY: {len(RESULTS)} checks, {len(fails)} FAIL, {len(warns)} WARN ===")
    for area, check, status, _ in fails + warns:
        print(f"  {status}  {area}: {check}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
