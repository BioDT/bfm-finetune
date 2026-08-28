"""G0 / G1 / G1b / G2 as executable assertions.

Each gate returns a record carrying its numbers and a verdict, and writes it to
``<out_dir>/<gate>_<task>.json``. A red verdict stops the build rather than being worked
around. Thresholds are named constants so a reviewer can see and contest them.
"""

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PASS, AMBER, RED = "PASS", "AMBER", "RED"

G0_MIN_UNIT_YEARS = 30          # minimum sample in the test window
G0_MIN_SPECIES_TOTAL = 3        # fewer than 3 across A and C drops the reuse framing
G1_MIN_TREND_RHO = 0.8          # our aggregate vs the scheme's published index
G1B_MIN_UNITS_BOTH = 100        # below this, spatial replication cannot carry a year-pair
G1B_AMBER_RATIO = 0.90          # test-year units / previous-year units
G1B_RED_RATIO = 0.50

SPECIES_TABLE = Path(__file__).with_name("bfm_species.json")


def bfm_species(path: Path = SPECIES_TABLE) -> pd.DataFrame:
    spec = json.loads(path.read_text())
    df = pd.DataFrame(spec["decoder_species_vars"])
    df.attrs["source"] = spec["source"]
    df.attrs["biocube_keys"] = spec["biocube_batch_species_keys"]
    return df


def _write(record: dict[str, Any], out_dir: str | Path) -> Path:
    from .common.runner import write_json

    path = Path(out_dir) / f"{record['gate']}_{record['task']}.json"
    write_json(path, record)
    return path


def g0_species_overlap(panels: dict[str, pd.DataFrame], *, out_dir: str | Path,
                       test_years: Iterable[int] = (2019, 2020),
                       min_unit_years: int = G0_MIN_UNIT_YEARS) -> dict[str, Any]:
    """Which of the 28 decoder species appear in Tasks A and C, with enough unit-years.

    Detections are counted separately from observed unit-years: a species present purely as
    reconstructed zeros carries no signal to rank.
    """
    species = bfm_species()
    test_years = tuple(int(y) for y in test_years)
    per_species: list[dict[str, Any]] = []

    for _, row in species.iterrows():
        name = row["scientific_name"]
        entry: dict[str, Any] = {"gbif_key": row["gbif_key"], "scientific_name": name,
                                 "channel_index": int(row["i"]), "tasks": {}}
        for task, panel in panels.items():
            sub = panel.loc[(panel["species"] == name) & panel["observed"]]
            if sub.empty:
                continue
            by_year = {int(y): int(n) for y, n in sub.groupby("year", observed=True).size().items()}
            nz = sub.loc[sub["value"] > 0]
            entry["tasks"][task] = {
                "unit_years_total": int(len(sub)),
                "unit_years_in_test_window": int(sum(by_year.get(y, 0) for y in test_years)),
                "unit_years_detected_total": int(len(nz)),
                "unit_years_detected_in_test_window":
                    int(len(nz.loc[nz["year"].isin(test_years)])),
                "by_test_year": {str(y): by_year.get(y, 0) for y in test_years},
                "detected_by_test_year":
                    {str(y): int((nz["year"] == y).sum()) for y in test_years},
                "prevalence": float((sub["value"] > 0).mean()),
            }
        if entry["tasks"]:
            per_species.append(entry)

    def survives(entry: dict[str, Any]) -> bool:
        return any(t["unit_years_detected_in_test_window"] >= min_unit_years
                   for t in entry["tasks"].values())

    surviving = [e for e in per_species if survives(e)]
    verdict = PASS if len(surviving) >= G0_MIN_SPECIES_TOTAL else RED
    record = {
        "gate": "G0", "task": "+".join(sorted(panels)), "verdict": verdict,
        "criterion": f">= {min_unit_years} unit-years with a detection in {list(test_years)}, "
                     f"at least {G0_MIN_SPECIES_TOTAL} species across the tasks",
        "n_decoder_species": int(len(species)),
        "n_present_in_any_task": len(per_species),
        "n_surviving": len(surviving),
        "surviving": [{"scientific_name": e["scientific_name"], "tasks": list(e["tasks"])}
                      for e in surviving],
        "per_species": per_species,
        "on_red": "drop the reuse-own-outputs framing; run all tasks with new heads",
        "species_source": species.attrs["source"],
        "biocube_input_species": species.attrs["biocube_keys"],
    }
    _write(record, out_dir)
    return record


def g1b_survey_coverage(panel: pd.DataFrame, *, out_dir: str | Path,
                        pairs: Iterable[tuple[int, int]] = ((2018, 2019), (2019, 2020))) -> dict[str, Any]:
    """Units surveyed in each year of a forecast pair, and in both.

    Two separate questions: did coverage collapse in the test year, and is the overlap wide
    enough to define a previous-year reference forecast. Thin overlap is red on an annual
    fixed-unit panel but amber on a rotating panel, where it is the design.
    """
    obs = panel.loc[panel["observed"]]
    units_by_year = {int(y): set(g["unit_id"].unique())
                     for y, g in obs.groupby("year", observed=True)}
    coverage = {str(y): len(u) for y, u in sorted(units_by_year.items())}

    rotating = _is_rotating_panel(panel, units_by_year, pairs)

    results = []
    worst = PASS
    for y0, y1 in pairs:
        a, b = units_by_year.get(y0, set()), units_by_year.get(y1, set())
        ratio = (len(b) / len(a)) if a else float("nan")
        both = len(a & b)

        if not a or not b or (np.isfinite(ratio) and ratio < G1B_RED_RATIO):
            collapse = RED
        elif np.isfinite(ratio) and ratio < G1B_AMBER_RATIO:
            collapse = AMBER
        else:
            collapse = PASS

        if both >= G1B_MIN_UNITS_BOTH:
            overlap = PASS
        elif rotating["is_rotating"]:
            overlap = AMBER
        else:
            overlap = RED

        v = RED if RED in (collapse, overlap) else (AMBER if AMBER in (collapse, overlap) else PASS)
        worst = RED if RED in (worst, v) else (AMBER if AMBER in (worst, v) else PASS)
        results.append({"pair": [y0, y1], f"units_{y0}": len(a), f"units_{y1}": len(b),
                        "units_in_both": both,
                        "ratio_test_over_previous": None if not np.isfinite(ratio) else round(ratio, 4),
                        "retained_fraction_of_previous": round(both / len(a), 4) if a else None,
                        "coverage_verdict": collapse, "overlap_verdict": overlap, "verdict": v})

    record = {
        "gate": "G1b", "task": str(panel["task"].iloc[0]), "verdict": worst,
        "criterion": "coverage: red below "
                     f"{G1B_RED_RATIO}x the previous year's units, amber below {G1B_AMBER_RATIO}x. "
                     f"overlap: at least {G1B_MIN_UNITS_BOTH} units in both years of a pair, "
                     "relaxed to amber on a rotating panel design",
        "units_surveyed_per_year": coverage, "rotating_panel": rotating, "pairs": results,
        "on_red": "change the forecast split before any GPU time is spent",
    }
    _write(record, out_dir)
    return record


def _is_rotating_panel(panel: pd.DataFrame, units_by_year: dict[int, set],
                       pairs: Iterable[tuple[int, int]]) -> dict[str, Any]:
    """Distinguish a rotating sample design from an effort collapse.

    A rotating panel surveys a stable *number* of units while deliberately changing *which*
    ones, so annual coverage stays flat while consecutive-year overlap is low.
    """
    obs = panel.loc[panel["observed"]]
    unit_years = obs.loc[:, ["unit_id", "year"]].drop_duplicates().sort_values(["unit_id", "year"])
    gaps = unit_years.groupby("unit_id", observed=True)["year"].diff().dropna()
    counts = np.array([len(u) for _, u in sorted(units_by_year.items())], dtype=float)

    stable = bool(counts.size > 2 and counts.std(ddof=1) / counts.mean() < 0.25)
    retention = [len(units_by_year.get(y0, set()) & units_by_year.get(y1, set()))
                 / max(len(units_by_year.get(y0, set())), 1) for y0, y1 in pairs]
    thin = bool(retention and max(retention) < 0.5)
    return {
        "is_rotating": stable and thin,
        "annual_unit_count_cv": round(float(counts.std(ddof=1) / counts.mean()), 4) if counts.size > 1 else None,
        "median_revisit_gap_years": float(gaps.median()) if len(gaps) else None,
        "mean_revisit_gap_years": round(float(gaps.mean()), 3) if len(gaps) else None,
        "evidence": "annual unit count is stable (low coefficient of variation) while "
                    "consecutive-year retention is below 0.5",
    }


def g1_national_trend(panel: pd.DataFrame, published: pd.DataFrame, *, out_dir: str | Path,
                      species_col: str = "species", year_col: str = "year",
                      index_col: str = "index") -> dict[str, Any]:
    """Does the panel, aggregated, reproduce the scheme's own published national index?

    Compared per species on rank correlation of the annual series and of year-on-year
    changes; a parser bug that mangles units shows up in the second even when the first
    survives.
    """
    from scipy.stats import spearmanr

    obs = panel.loc[panel["observed"]]
    ours = obs.groupby(["species", "year"], observed=True)["value"].mean().rename("ours").reset_index()
    pub = published.rename(columns={species_col: "species", year_col: "year", index_col: "published"})
    pub["year"] = pub["year"].astype("int32")
    merged = ours.merge(pub, on=["species", "year"], how="inner")

    per_species = {}
    for sp, g in merged.groupby("species", observed=True):
        g = g.sort_values("year")
        if len(g) < 5:
            continue
        rho = spearmanr(g["ours"], g["published"]).statistic
        d_ours, d_pub = np.diff(g["ours"].to_numpy(float)), np.diff(g["published"].to_numpy(float))
        rho_d = spearmanr(d_ours, d_pub).statistic if len(d_ours) >= 4 else np.nan
        per_species[str(sp)] = {"n_years": int(len(g)),
                                "years": [int(g["year"].min()), int(g["year"].max())],
                                "rho_level": None if not np.isfinite(rho) else round(float(rho), 4),
                                "rho_change": None if not np.isfinite(rho_d) else round(float(rho_d), 4)}

    levels = [v["rho_level"] for v in per_species.values() if v["rho_level"] is not None]
    median_rho = float(np.median(levels)) if levels else float("nan")
    if not levels:
        verdict = AMBER
    elif median_rho >= G1_MIN_TREND_RHO:
        verdict = PASS
    else:
        verdict = RED

    record = {
        "gate": "G1", "task": str(panel["task"].iloc[0]), "verdict": verdict,
        "criterion": f"median per-species Spearman of our annual aggregate against the "
                     f"published index >= {G1_MIN_TREND_RHO}",
        "n_species_compared": len(per_species), "median_rho_level": median_rho,
        "per_species": per_species,
        "on_red": "parser bug; fix before proceeding",
    }
    _write(record, out_dir)
    return record


def g2_nulls(panel: pd.DataFrame, splits: Iterable[Any], *, out_dir: str | Path,
             seed: int = 0) -> dict[str, Any]:
    """Compute and record all nulls on the test splits.

    G2 cannot fail; the deliverable is that these scores exist on disk before any model
    score does, so the comparison cannot be tuned after the fact.
    """
    from .eval.nulls import score_nulls

    per_split = [score_nulls(panel, split, seed=seed) for split in splits]

    record = {
        "gate": "G2", "task": str(panel["task"].iloc[0]), "verdict": PASS,
        "criterion": "all nulls scored and written before any model score exists",
        "splits": per_split,
        "ordering_guarantee": "no model score for this task may be written before this file",
    }
    _write(record, out_dir)
    return record
