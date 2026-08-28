"""Per-task result tables: nulls above, learned baselines below, model last.

Two rules are enforced here rather than left to whoever assembles the manuscript. There is
no pooled cross-task number — the three targets are different quantities and averaging
them means nothing, so this module cannot emit one. And the null block always renders above
the learned baselines: nulls answer "is this better than trivial?", learned baselines
answer "are you comparing against real methods?".
"""

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

NULL_BLOCK = ("persistence", "lagged_persistence", "climatology", "species_mean",
              "site_mean", "spatial_neighbour", "distribution_matched")
BASELINE_BLOCK = ("glm", "glm_strata", "nbgam", "randomforest", "convlstm")
# A row named here but absent from the results is not rendered; a result present but not
# named here is invisible — `stage_tables` fails when a result file is unaccounted for.
MODEL_BLOCK = ("bfm_l0", "bfm_l1", "bfm_l2", "aurora_l2",
               "bfm_l3_lora4", "bfm_l3_full", "aurora_l3_lora4", "aurora_l3_full")

PRETTY = {
    "persistence": "Persistence", "lagged_persistence": "Persistence (lagged)",
    "climatology": "Climatology", "species_mean": "Species mean", "site_mean": "Site mean",
    "spatial_neighbour": "Spatial neighbour", "distribution_matched": "Distribution-matched random",
    "glm": "TRIM basic (site + shared trend)",
    "glm_strata": "TRIM + regional covariate", "nbgam": "NB-GAM (spatial smooth + environment)",
    "randomforest": "RandomForest", "convlstm": "ConvLSTM",
    "bfm_l3_lora4": "BioAnalyst L3 (LoRA r=4)",
    "aurora_l3_lora4": "Aurora L3 (LoRA r=4)",
    "aurora_l3_full": "Aurora L3 (full fine-tune)",
    "latentmlp": "LatentMLP", "bfm_l0": "BioAnalyst L0 (zero-shot)",
    "bfm_l1": "BioAnalyst L1 (calibrated)", "bfm_l2": "BioAnalyst L2 (frozen probe)",
    "bfm_l3_vera": "BioAnalyst L3 (VeRA r=256)",
    "bfm_l3_lora16": "BioAnalyst L3 (LoRA r=16)",
    "bfm_l3_lora1": "BioAnalyst L3 (LoRA r=1, budget-matched)",
    "bfm_l3_vera_asshipped": "BioAnalyst L3 (VeRA, bfm-model init)",
    "bfm_l3_full": "BioAnalyst L3 (full fine-tune)",
    "aurora_l2": "Aurora L2 (frozen probe)", "aurora_l3_vera": "Aurora L3 (VeRA r=256)",
    "bfm-matched_l2": "BioAnalyst L2 (Aurora-matched variables)",
}
IMPOSSIBLE = "not applicable — no species outputs"

TASK_UNITS = {"A": ("route", "count"), "B": ("0.25° cell", "prevalence"),
              "C": ("0.25° cell", "index")}


class PooledNumberRefused(RuntimeError):
    pass


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _mean_sd(values: Sequence[float], digits: int = 3) -> str:
    vals = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return "—"
    if vals.size == 1:
        return _fmt(float(vals[0]), digits)
    return f"{vals.mean():.{digits}f} ± {vals.std(ddof=1):.{digits}f}"


def pull_str(records: Sequence[dict[str, Any]], path: Sequence[str]) -> list[str]:
    out = []
    for rec in records:
        node: Any = rec
        for key in path:
            node = (node or {}).get(key) if isinstance(node, dict) else None
        if isinstance(node, str):
            out.append(node)
    return out


def _mode(values: Sequence[str]) -> str:
    """Most common strongest-reference across seeds; "--" when nothing recorded."""
    if not values:
        return "--"
    return max(set(values), key=values.count).replace("vs_", "")


def row_from_scores(name: str, records: Sequence[dict[str, Any]], reference: str) -> dict[str, Any]:
    """One table row, aggregating over seeds as mean ± sd."""
    def pull(path: Sequence[str]) -> list[float]:
        out = []
        for rec in records:
            node: Any = rec
            for key in path:
                node = (node or {}).get(key) if isinstance(node, dict) else None
            out.append(node if isinstance(node, (int, float)) else None)
        return out

    return {
        "name": PRETTY.get(name, name), "key": name, "n_seeds": len(records),
        # Two skill columns on purpose: `skill` is against the task's fixed reference, kept
        # for continuity; `skill_strongest` is the headline, against whichever legitimate
        # reference is hardest in that year. The headline column is fixed in advance.
        "skill": _mean_sd(pull(["skill", f"vs_{reference}", "skill_score"])),
        "skill_strongest": _mean_sd(pull(["skill_vs_strongest_null", "skill_score"])),
        "skill_log": _mean_sd(pull(["skill", f"vs_{reference}", "skill_score_log1p"])),
        "strongest_ref": _mode(pull_str(records, ["skill_vs_strongest_null", "reference"])),
        "rmse_log": _mean_sd(pull(["rmse_log"])),
        "spatial_rho": _mean_sd(pull(["spatial_rho", "mean"])),
        "auc_pr": _mean_sd(pull(["auc_pr"])),
        "mcc": _mean_sd(pull(["mcc"])),
        "sign_coverage": _mean_sd(pull(["sign_coverage"])),
        "n": _fmt(np.nanmax([v for v in pull(["n"]) if v is not None] or [np.nan]), 0),
        # Carried so the table can flag rows scored on a different population — L0 and L1
        # only cover species the model's own decoder emits.
        "n_species": np.nanmax([v for v in pull(["n_species"]) if v is not None] or [np.nan]),
    }


def task_table(task: str, blocks: dict[str, dict[str, list[dict[str, Any]]]], *,
               reference: str, ceiling: dict[str, Any] | None = None,
               test_year: int | None = None,
               row_notes: dict[str, str] | None = None) -> dict[str, Any]:
    """Assemble one task's table. ``blocks`` maps block name -> {row key -> [per-seed records]}."""
    unit, value_type = TASK_UNITS.get(task, ("unit", "value"))
    ordered = {}
    for block, names in (("nulls", NULL_BLOCK), ("baselines", BASELINE_BLOCK),
                         ("model", MODEL_BLOCK)):
        present = blocks.get(block, {})
        ordered[block] = [row_from_scores(n, present[n], reference) for n in names if n in present]

    # Flag rungs scored on fewer species than the rest, from the data itself.
    counts = [r["n_species"] for rows in ordered.values() for r in rows
              if np.isfinite(r.get("n_species", np.nan))]
    auto_notes: dict[str, str] = {}
    if counts:
        full = max(counts)
        for rows in ordered.values():
            for r in rows:
                k = r.get("n_species", np.nan)
                if np.isfinite(k) and k < full:
                    auto_notes[r["key"]] = (
                        f"scored on {int(k)} of {int(full)} species — only those the model's "
                        f"decoder emits — so this row is not comparable with the rungs above it")

    return {
        "task": task, "unit": unit, "value_type": value_type, "test_year": test_year,
        "skill_reference": reference,
        "reference_note":
            f"**Skill vs strongest null** is the headline: for each predictor it is the skill "
            f"against whichever of persistence, lagged persistence or climatology is hardest "
            f"in this test year. Taking the minimum can only lower a score. "
            f"*Skill vs {PRETTY.get(reference, reference)}* is the fixed-reference column, "
            f"kept for continuity, and that row is 0 by construction. The two differ because "
            f"the strongest null is not the same one every year. *Skill (log)* is the same "
            f"fixed-reference score on log(1+y), reported because raw skill on these targets "
            f"is decided by a small number of large units.",
        "blocks": ordered,
        "ceiling": ceiling,
        "row_notes": {k: v for k, v in {**auto_notes, **(row_notes or {})}.items()
                      if any(r["key"] == k for rows in ordered.values() for r in rows)},
        "columns": ["skill_strongest", "skill", "skill_log", "rmse_log", "spatial_rho"] +
                   (["auc_pr", "mcc"] if value_type == "prevalence" else []) +
                   ["sign_coverage", "n"],
    }


def to_markdown(table: dict[str, Any]) -> str:
    cols = table["columns"]
    header = {"skill_strongest": "**Skill vs strongest null**",
              "skill_log": "Skill (log)",
              "skill": f"Skill vs {PRETTY.get(table['skill_reference'], table['skill_reference'])}",
              "rmse_log": "log-RMSE", "spatial_rho": "Spatial ρ", "auc_pr": "AUC-PR",
              "mcc": "MCC", "sign_coverage": "Sign coverage", "n": "n"}
    lines = [f"**Task {table['task']}** — unit: {table['unit']}, target: {table['value_type']}"
             + (f", test year {table['test_year']}" if table["test_year"] else ""),
             "", "| | " + " | ".join(header[c] for c in cols) + " |",
             "|---|" + "|".join(["---"] * len(cols)) + "|"]

    labels = {"nulls": "*Null models*", "baselines": "*Learned baselines*",
              "model": "*Foundation models*"}
    for block in ("nulls", "baselines", "model"):
        rows = table["blocks"].get(block) or []
        if not rows:
            continue
        lines.append(f"| {labels[block]} |" + " |" * len(cols))
        for row in rows:
            lines.append("| " + row["name"] + " | " + " | ".join(str(row[c]) for c in cols) + " |")

    notes = table.get("row_notes") or {}
    if notes:
        lines.append("")
        for key, note in notes.items():
            lines.append(f"*{PRETTY.get(key, key)}: {note}*")

    if table.get("ceiling"):
        c = table["ceiling"]
        skill = c.get("skill", {}).get(f"vs_{table['skill_reference']}")
        lines += ["", f"*Cell-resolution ceiling (oracle): skill {_fmt(skill)}, "
                      f"spatial ρ {_fmt(c.get('spatial_rho'))}, "
                      f"{100 * c.get('within_cell_variance_fraction', float('nan')):.1f}% of variance "
                      f"within-cell. A prediction at 0.25° cannot exceed this.*"]
    lines += ["", f"*{table['reference_note']}.*"]
    return "\n".join(lines)


def refuse_pooled(*_args: Any, **_kwargs: Any) -> None:
    """Explicitly unavailable. A named refusal, so the absence reads as a decision."""
    raise PooledNumberRefused(
        "no pooled cross-task number: Tasks A, B and C target route counts, cell prevalence "
        "and a GAM-derived index respectively, and averaging them is not a quantity")


def write_tables(tables: Iterable[dict[str, Any]], out_dir: str | Path) -> dict[str, str]:
    from .runner import atomic_path, write_json

    out_dir = Path(out_dir)
    paths = {}
    tables = list(tables)
    for table in tables:
        stem = f"table_{table['task']}" + (f"_{table['test_year']}" if table.get("test_year") else "")
        write_json(out_dir / f"{stem}.json", table)
        with atomic_path(out_dir / f"{stem}.md", suffix=".md") as tmp:
            tmp.write_text(to_markdown(table))
        paths[table["task"]] = str(out_dir / f"{stem}.md")

    combined = "\n\n---\n\n".join(to_markdown(t) for t in tables)
    with atomic_path(out_dir / "tables_all_tasks.md", suffix=".md") as tmp:
        tmp.write_text("<!-- Per-task tables. There is deliberately no pooled row. -->\n\n"
                       + combined)
    paths["all"] = str(out_dir / "tables_all_tasks.md")
    return paths
