"""All figures. Every figure ships twice: vector PDF for the manuscript, PNG for inspection.

House style is set once in ``use_house_style`` so figures cannot drift apart. Colour
encodes one thing — which model — in a colour-blind-safe pair; nulls and classical
baselines use recessive ink, and every figure carries a non-colour cue as well.
"""

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .runner import artefacts_root, project_root

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"

TASK_LABELS = {
    "A": ("Task A — Swedish birds", "routes"),
    "B": ("Task B — Swedish NFI", "cells"),
    "C": ("Task C — UK butterflies", "sites"),
}
TEST_YEARS = (2019, 2020)

MODEL_COLOUR = {"bfm": BLUE, "aurora": ORANGE}
MODEL_MARKER = {"bfm": "o", "aurora": "^"}
MODEL_LABEL = {"bfm": "BioAnalyst", "aurora": "Aurora"}


def use_house_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.labelcolor": INK,
        "axes.edgecolor": INK_2, "axes.linewidth": 0.6, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, out_dir: str | Path, name: str, sources: list[dict[str, Any]] | None = None) -> dict[str, str]:
    """Write PDF and PNG atomically, with a provenance sidecar."""
    from .runner import artefact_provenance, atomic_path

    out_dir = Path(out_dir)
    paths = {}
    for ext in ("pdf", "png"):
        target = out_dir / f"{name}.{ext}"
        with atomic_path(target, suffix=f".{ext}") as tmp:
            fig.savefig(tmp, format=ext)
        paths[ext] = str(target)
        artefact_provenance(target, sources=sources or [], extra={"figure": name})
    return paths


def survey_coverage(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Units surveyed per year, and the fraction of the previous year's units resurveyed."""
    out = {}
    for task, panel in panels.items():
        obs = panel.loc[panel["observed"]]
        by_year = {int(y): set(g["unit_id"].unique())
                   for y, g in obs.groupby("year", observed=True)}
        years = sorted(by_year)
        rows = []
        for y in years:
            prev = by_year.get(y - 1)
            both = len(by_year[y] & prev) if prev else np.nan
            rows.append({"year": y, "units": len(by_year[y]),
                         "units_in_both": both,
                         "retention": (both / len(prev)) if prev else np.nan})
        out[task] = pd.DataFrame(rows)
    return out


def figure_g(panels: dict[str, pd.DataFrame], out_dir: str | Path,
             sources: list[dict[str, Any]] | None = None,
             year_min: int = 1996,
             name: str = "figure_G_survey_coverage") -> dict[str, Any]:
    """G — survey coverage per year per task; the direct evidence behind G1b.

    Two rows because the gate asks two questions that look alike and are not: whether
    coverage collapsed, and whether the same units recur.
    """
    import matplotlib.pyplot as plt

    use_house_style()
    stats = survey_coverage(panels)
    tasks = [t for t in ("A", "B", "C") if t in stats]

    fig, axes = plt.subplots(2, len(tasks), figsize=(max(3.2, 2.4 * len(tasks)), 4.3),
                             sharex="col",
                             gridspec_kw={"height_ratios": [1.0, 0.8], "hspace": 0.22,
                                          "wspace": 0.34})
    # Explicit reshape, not atleast_2d: with one task subplots returns shape (2,), which
    # atleast_2d would transpose to (1, 2) and break the [row, col] indexing.
    axes = np.asarray(axes).reshape(2, len(tasks))

    # The two test years are adjacent; their labels are pushed apart horizontally.
    side = {TEST_YEARS[0]: ("right", -5), TEST_YEARS[1]: ("left", 5)}

    for k, task in enumerate(tasks):
        df = stats[task].loc[stats[task]["year"] >= year_min]
        title, unit_word = TASK_LABELS[task]
        is_test = df["year"].isin(TEST_YEARS).to_numpy()

        ax = axes[0, k]
        ax.bar(df["year"], df["units"], width=0.72, linewidth=0,
               color=np.where(is_test, ORANGE, BLUE))
        ax.set_title(f"{title}\n({unit_word})", color=INK, pad=5)
        ax.set_ylim(0, float(df["units"].max()) * 1.22)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        for y in TEST_YEARS:
            row = df.loc[df["year"] == y]
            if row.empty:
                continue
            ha, dx = side[y]
            ax.annotate(f"{int(row['units'].iloc[0]):,}", (y, row["units"].iloc[0]),
                        textcoords="offset points", xytext=(dx, 3), ha=ha,
                        fontsize=6.4, color=ORANGE)

        ax = axes[1, k]
        ax.plot(df["year"], df["retention"], color=BLUE, linewidth=1.4, zorder=3)
        sel = df.loc[is_test]
        ax.plot(sel["year"], sel["retention"], "o", color=ORANGE, markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("year", color=INK_2)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        # Label the higher point above and the lower below, so neither is overwritten.
        vals = {y: df.loc[df["year"] == y, "retention"] for y in TEST_YEARS}
        vals = {y: v.iloc[0] for y, v in vals.items() if not v.empty and np.isfinite(v.iloc[0])}
        if vals:
            top = max(vals, key=vals.get)
            for y, value in vals.items():
                ha, dx = side[y]
                dy = 6 if (y == top and len(vals) > 1) else -11
                ax.annotate(f"{value:.2f}", (y, value), textcoords="offset points",
                            xytext=(dx, dy), ha=ha, fontsize=6.4, color=ORANGE)

    axes[0, 0].set_ylabel("units surveyed", color=INK_2)
    axes[1, 0].set_ylabel("fraction resurveyed\nfrom previous year", color=INK_2)

    handles = [plt.Line2D([], [], marker="s", linestyle="none", color=BLUE, markersize=5),
               plt.Line2D([], [], marker="s", linestyle="none", color=ORANGE, markersize=5)]
    fig.legend(handles, ["training years", "forecast test years (2019, 2020)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.045),
               handlelength=1.0, columnspacing=1.6)

    paths = save(fig, out_dir, name, sources)
    plt.close(fig)
    return {"paths": paths,
            "per_task": {t: stats[t].to_dict(orient="list") for t in tasks}}


def _seed_of(name: str) -> int | None:
    m = re.search(r"_s(\d+)_\d{4}$", name)
    return int(m.group(1)) if m else None


def seeds_available(task: str, pattern: str, year: int) -> set[int]:
    root = artefacts_root() / "checkpoints" / task
    if not root.exists():
        return set()
    return {sd for d in root.glob(f"{pattern}_s*_{year}")
            if (d / "predictions.parquet").exists() and (sd := _seed_of(d.name)) is not None}


def common_seeds(tasks: Sequence[str], patterns: Sequence[str], year: int) -> list[int]:
    """Seeds present for every (task, pattern) being plotted, so panels of one figure never
    pool different seed counts without saying so."""
    sets = [seeds_available(t, p, year) for t in tasks for p in patterns]
    sets = [s for s in sets if s]
    return sorted(set.intersection(*sets)) if sets else []


def _checkpoint_predictions(task: str, pattern: str, year: int,
                            seeds: Sequence[int] | None = None) -> pd.DataFrame | None:
    """Concatenate saved prediction frames for one test year, optionally a fixed seed set."""
    root = artefacts_root() / "checkpoints" / task
    if not root.exists():
        return None
    frames, seen = [], set()
    for d in sorted(root.glob(f"{pattern}_s*_{year}")):
        if d.name in seen:
            continue
        seen.add(d.name)
        sd = _seed_of(d.name)
        if seeds is not None and sd not in set(seeds):
            continue
        f = d / "predictions.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f).assign(run=d.name, seed=sd))
    return pd.concat(frames, ignore_index=True) if frames else None


def _reference_column(panel: pd.DataFrame, frame: pd.DataFrame, task: str, year: int,
                      first_train_year: int = 2000) -> np.ndarray:
    """The task's reference null, aligned to ``frame``'s rows, computed on the pipeline's
    own training window (2000 to year-1) so the figures match the tables."""
    obs = panel.loc[panel["observed"]]
    if task == "B":
        train = obs.loc[obs["year"].between(first_train_year, year - 1)]
        ref = train.groupby(["unit_id", "species"], observed=True)["value"].mean()
    else:
        ref = obs.loc[obs["year"] == year - 1].set_index(["unit_id", "species"])["value"]
    key = list(zip(frame["unit_id"], frame["species"]))
    return np.array([ref.get(k, np.nan) for k in key], dtype=float)


def _skill(y: np.ndarray, p: np.ndarray, r: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(r)
    if ok.sum() < 2:
        return float("nan")
    den = float(np.mean((y[ok] - r[ok]) ** 2))
    return float("nan") if den == 0 else 1 - float(np.mean((y[ok] - p[ok]) ** 2)) / den


def figure_d(tables: list[dict[str, Any]], out_dir: str | Path,
             lo: float = -1.15, hi: float = 0.4,
             tasks: Sequence[str] = ("A", "B", "C"),
             name: str = "figure_D_null_vs_model",
             sources: list[dict[str, Any]] | None = None) -> dict[str, str]:
    """D — every predictor's skill against the strongest null, one row each.

    The x-axis is clipped so one outlier cannot compress everything else into a clump at
    zero; clipped marks are drawn as an open chevron at the boundary with their true value
    printed, so nothing is hidden, only moved.
    """
    import matplotlib.pyplot as plt

    use_house_style()
    width = 12.4 * len(tasks) / 3 if len(tasks) > 1 else 5.5
    fig, axes = plt.subplots(2, len(tasks), figsize=(width, 8.6), sharex=True)
    axes = np.asarray(axes).reshape(2, len(tasks))
    for row, year in enumerate(TEST_YEARS):
        for col, task in enumerate(tasks):
            ax = axes[row, col]
            tab = next((t for t in tables if t["task"] == task and t["test_year"] == year), None)
            if tab is None:
                ax.set_visible(False)
                continue
            rows, unscored = [], []
            for block in ("nulls", "baselines", "model"):
                for r in tab["blocks"].get(block, []):
                    try:
                        v = float(str(r["skill_strongest"]).split("±")[0])
                    except (ValueError, TypeError):
                        # L0 is rank-only: a skill score is undefined rather than zero, so
                        # the rung is omitted rather than drawn at 0.0.
                        unscored.append(r["name"])
                        continue
                    key = r["key"]
                    who = ("aurora" if key.startswith("aurora")
                           else "bfm" if key.startswith("bfm") else None)
                    rows.append((r["name"], v, who, block))
            if not rows:
                ax.set_visible(False)
                continue
            ypos = np.arange(len(rows))[::-1]
            ax.axvline(0.0, color=INK, lw=0.9, zorder=4)
            for yp, (_, v, who, block) in zip(ypos, rows):
                c = MODEL_COLOUR.get(who, INK_2)
                m = MODEL_MARKER.get(who, "s")
                vc = min(max(v, lo), hi)
                ax.plot([0, vc], [yp, yp], color=c, lw=1.1, alpha=0.4, zorder=2,
                        solid_capstyle="butt")
                if v < lo:
                    ax.plot(lo, yp, "<", color=c, ms=5, mfc="white", mew=1.0, zorder=5)
                    ax.annotate(f"{v:.2f}", (lo, yp), textcoords="offset points",
                                xytext=(7, 0), va="center", fontsize=5.2, color=INK_2)
                else:
                    ax.plot(vc, yp, m, color=c, ms=4.8, mec="white", mew=0.7, zorder=5)
            ax.set_yticks(ypos)
            ax.set_yticklabels([n for n, _, _, _ in rows] if col == 0 else [], fontsize=5.9)
            ax.set_title(f"{year}" if len(tasks) == 1 else f"Task {task} — {year}",
                         pad=4, fontsize=7.6)
            ax.set_xlim(lo - 0.06, hi)
            ax.grid(axis="x", zorder=0)
            if row == 1:
                ax.set_xlabel("skill vs strongest null")
    handles = [plt.Line2D([], [], color=MODEL_COLOUR[k], marker=MODEL_MARKER[k], ls="none",
                          ms=5, label=MODEL_LABEL[k]) for k in ("bfm", "aurora")]
    handles += [plt.Line2D([], [], color=INK_2, marker="s", ls="none", ms=5,
                           label="null / classical baseline"),
                plt.Line2D([], [], color=INK_2, marker="<", ls="none", ms=5, mfc="white",
                           label=f"below {lo:g} (true value printed)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("Skill against the strongest legitimate null — right of the line beats it",
                 x=0.006, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    return save(fig, out_dir, name,
                sources=sources or [{"from": "tables/table_*.json"}])


def figure_e(panels: dict[str, pd.DataFrame], out_dir: str | Path,
             year: int = 2019, tasks: Sequence[str] = ("A", "B", "C"),
             name: str = "figure_E_calibration") -> dict[str, str]:
    """E — calibration: observed against predicted, with the 1:1 line.

    Binned medians are drawn over the cloud because a scatter of 130,000 points hides the
    systematic part; the 1:1 line is the claim being tested.
    """
    import matplotlib.pyplot as plt

    use_house_style()
    seeds = common_seeds(tuple(tasks), ("bfm_l3_lora4", "aurora_l3_lora"), year)
    n_bins: dict[str, tuple[int, float]] = {}
    fig, axes = plt.subplots(1, len(tasks), squeeze=False,
                             figsize=(10.5 * len(tasks) / 3 if len(tasks) > 1 else 4.2, 3.6))
    for ax, task in zip(axes[0], tasks):
        drawn = False
        for who in ("bfm", "aurora"):
            fr = _checkpoint_predictions(task, f"{who}_l3_lora4", year, seeds)
            if fr is None or fr.empty:
                continue
            y = fr["y_true"].to_numpy(float)
            p = fr["y_pred"].to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(p)
            y, p = y[ok], p[ok]
            ax.scatter(p, y, s=1.5, alpha=0.06, color=MODEL_COLOUR[who], linewidths=0,
                       rasterized=True, zorder=2)
            edges = np.unique(np.quantile(p, np.linspace(0, 1, 13)))
            floor_share = float(np.mean(p <= p.min() + 1e-12))
            n_bins[task] = (len(edges) - 1, floor_share)
            if len(edges) > 2:
                mid = 0.5 * (edges[:-1] + edges[1:])
                med = [np.median(y[(p >= a) & (p < b)]) if ((p >= a) & (p < b)).any() else np.nan
                       for a, b in zip(edges[:-1], edges[1:])]
                ax.plot(mid, med, MODEL_MARKER[who] + "-", color=MODEL_COLOUR[who], ms=4,
                        lw=1.4, mec="white", mew=0.6, zorder=4, label=MODEL_LABEL[who])
            drawn = True
        if not drawn:
            ax.set_visible(False)
            continue
        lim = ax.get_xlim() + ax.get_ylim()
        hi = max(lim)
        lo = 0
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.9, ls=(0, (4, 3)), zorder=5, label="1:1")
        if task != "B":
            ax.set_xscale("symlog", linthresh=1)
            ax.set_yscale("symlog", linthresh=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        nb, fl = n_bins.get(task, (0, 0.0))
        ax.set_title(f"Task {task} — {TASK_LABELS[task][0].split('— ')[1]}\n"
                     f"{nb} of 12 bins resolve; {fl:.0%} of predictions on the floor",
                     pad=4, fontsize=7.0)
        ax.set_xlabel("predicted")
        ax.set_ylabel("observed")
        ax.grid(True, zorder=0)
        ax.legend(loc="upper left")
    fig.suptitle(f"Calibration, L3 LoRA r=4, test {year}, seed(s) {seeds or 'none'} — "
                 f"points below 1:1 are under-prediction",
                 x=0.006, ha="left", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, out_dir, name,
                sources=[{"from": "artefacts/checkpoints/*/{bfm,aurora}_l3_lora*_s*"}])


def figure_b(panels: dict[str, pd.DataFrame], out_dir: str | Path,
             year: int = 2019, tasks: Sequence[str] = ("A", "B", "C"),
             name: str = "figure_B_skill_map") -> dict[str, str]:
    """B — where the model beats its reference, in space.

    Diverging palette because the quantity has a meaningful zero; the scale is symmetric so
    the midpoint sits at zero rather than the data's mean.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    use_house_style()
    cmap = LinearSegmentedColormap.from_list("skill", [ORANGE, "#f2f1ee", BLUE])
    seeds = common_seeds(tuple(tasks), ("bfm_l3_lora4",), year)
    fig, axes = plt.subplots(1, len(tasks), squeeze=False,
                             figsize=(11.5 * len(tasks) / 3 if len(tasks) > 1 else 4.6, 4.0))
    for ax, task in zip(axes[0], tasks):
        fr = _checkpoint_predictions(task, "bfm_l3_lora4", year, seeds)
        if fr is None or fr.empty:
            ax.set_visible(False)
            continue
        panel = panels[task]
        fr = fr.assign(ref=_reference_column(panel, fr, task, year))
        per_unit = []
        for unit, g in fr.groupby("unit_id"):
            s = _skill(g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float),
                       g["ref"].to_numpy(float))
            if np.isfinite(s):
                per_unit.append((g["lon"].iloc[0], g["lat"].iloc[0], s))
        if not per_unit:
            ax.set_visible(False)
            continue
        lon, lat, sk = map(np.array, zip(*per_unit))
        sk_c = np.clip(sk, -1, 1)
        sc = ax.scatter(lon, lat, c=sk_c, cmap=cmap, norm=TwoSlopeNorm(0, -1, 1),
                        s=13, linewidths=0.2, edgecolors="white", zorder=3)
        clipped = int((sk < -1).sum())
        ax.set_title(f"Task {task} — {int((sk > 0).sum())}/{len(sk)} units beat the null\n"
                     f"{clipped} clipped at −1 ({clipped / len(sk):.0%}), worst {sk.min():+.1f}",
                     pad=4, fontsize=7.0)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, zorder=0)
        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, ticks=[-1, 0, 1])
        cb.ax.set_yticklabels(["≤ −1", "0", "1"], fontsize=6)
        cb.outline.set_visible(False)
    fig.suptitle(f"Per-unit skill vs the task reference, BioAnalyst L3 LoRA r=4, test {year}, "
                 f"seed(s) {seeds or 'none'}"
                 f" — blue beats the null, orange loses to it",
                 x=0.006, ha="left", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, out_dir, name,
                sources=[{"from": "artefacts/checkpoints/*/bfm_l3_lora*_s*"}])


def figure_f(panels: dict[str, pd.DataFrame], out_dir: str | Path,
             year: int = 2019, top: int = 22, min_detections: int = 20,
             tasks: Sequence[str] = ("A", "B", "C"),
             name: str = "figure_F_species_skill") -> dict[str, str]:
    """F — per-species skill, with the spread across seeds as the interval.

    Species are ordered by BioAnalyst's skill, so the shape of the distribution is the
    message. Species with too few detections are excluded: a species absent from the test
    year scores skill 1.0 for predicting zero everywhere, which is arithmetic, not skill.
    """
    import matplotlib.pyplot as plt

    use_house_style()
    XLO = -2.0
    seeds = common_seeds(tuple(tasks), ("bfm_l3_lora4", "aurora_l3_lora"), year)
    fig, axes = plt.subplots(1, len(tasks), squeeze=False,
                             figsize=(12.0 * len(tasks) / 3 if len(tasks) > 1 else 5.0, 5.2))
    for ax, task in zip(axes[0], tasks):
        panel = panels[task]
        per_model: dict[str, dict[str, list[float]]] = {}
        for who in ("bfm", "aurora"):
            fr = _checkpoint_predictions(task, f"{who}_l3_lora4", year, seeds)
            if fr is None or fr.empty:
                continue
            fr = fr.assign(ref=_reference_column(panel, fr, task, year))
            acc: dict[str, list[float]] = {}
            for (sp, run), g in fr.groupby(["species", "run"]):
                yt = g["y_true"].to_numpy(float)
                if int((yt > 0).sum()) < min_detections:
                    continue
                v = _skill(yt, g["y_pred"].to_numpy(float), g["ref"].to_numpy(float))
                if np.isfinite(v):
                    acc.setdefault(str(sp), []).append(v)
            per_model[who] = acc
        if "bfm" not in per_model or not per_model["bfm"]:
            ax.set_visible(False)
            continue
        keep = [s for s in per_model["bfm"]
                if per_model["bfm"].get(s) or per_model.get("aurora", {}).get(s)]
        ranked = sorted(keep, key=lambda s: np.median(per_model["bfm"][s]), reverse=True)
        n_clear = len(ranked)
        n_total = panel.loc[panel["observed"] & (panel["year"] == year), "species"].nunique()
        order = ranked
        if len(order) > top:                       # keep both tails, drop the middle
            order = order[: top // 2] + order[-top // 2:]
        ypos = np.arange(len(order))[::-1]
        ax.axvline(0.0, color=INK, lw=0.8, zorder=1)
        for who in ("aurora", "bfm"):
            acc = per_model.get(who, {})
            for yp, sp in zip(ypos, order):
                v = acc.get(sp)
                if not v:
                    continue
                off = 0.18 if who == "bfm" else -0.18
                if len(v) > 1:
                    ax.plot([min(v), max(v)], [yp + off] * 2, color=MODEL_COLOUR[who],
                            lw=1.1, alpha=0.6, zorder=2, solid_capstyle="round")
                med = float(np.median(v))
                if med < XLO:
                    ax.plot(XLO, yp + off, "<", color=MODEL_COLOUR[who], ms=4.5,
                            mfc="white", mew=0.9, zorder=3)
                    ax.annotate(f"{med:.1f}", (XLO, yp + off), textcoords="offset points",
                                xytext=(7, 0), va="center", fontsize=5.0, color=INK_2)
                else:
                    ax.plot(med, yp + off, MODEL_MARKER[who], color=MODEL_COLOUR[who],
                            ms=4, mec="white", mew=0.5, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels([s[:26] for s in order], fontsize=5.6, style="italic")
        n_seed = max((len(v) for a in per_model.values() for v in a.values()), default=1)
        shown = "best/worst" if n_clear > len(order) else "all"
        ax.set_title(f"Task {task} — {n_clear}/{n_total} species qualify\n"
                     f"{shown} {len(order)} shown · {n_seed} seed(s)", pad=4, fontsize=7.0)
        ax.set_xlabel("per-species skill vs reference")
        ax.set_xlim(XLO - 0.08, 1.05)
        ax.grid(axis="x", zorder=0)
    handles = [plt.Line2D([], [], color=MODEL_COLOUR[k], marker=MODEL_MARKER[k], ls="none",
                          ms=5, label=MODEL_LABEL[k]) for k in ("bfm", "aurora")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Per-species skill, L3 LoRA r=4, test {year}, seed(s) {seeds or 'none'} — "
                 f"species with <{min_detections} "
                 f"detections excluded; interval is the range across seeds",
                 x=0.006, ha="left", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return save(fig, out_dir, name,
                sources=[{"from": "artefacts/checkpoints/*/{bfm,aurora}_l3_lora*_s*"}])


def figure_a(panels: dict[str, pd.DataFrame], out_dir: str | Path,
             task: str = "A", n_species: int = 8) -> dict[str, str]:
    """A — observed trajectory per species, with the test years' predictions on top.

    The models predict two years, not a trajectory, so the line is the observed national
    mean and the markers are what each model said for the two shaded test years.
    """
    import matplotlib.pyplot as plt

    use_house_style()
    panel = panels[task]
    obs = panel.loc[panel["observed"]]
    common = (obs.loc[obs["year"].isin(TEST_YEARS)].groupby("species", observed=True)["value"]
              .mean().sort_values(ascending=False).head(n_species).index.tolist())
    seeds = {y: common_seeds((task,), ("bfm_l3_lora4", "aurora_l3_lora"), y) for y in TEST_YEARS}
    by_year = {(who, y): _checkpoint_predictions(task, f"{who}_l3_lora4", y, seeds[y])
               for who in ("bfm", "aurora") for y in TEST_YEARS}

    ncol = 4
    nrow = int(np.ceil(len(common) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.5, 2.5 * nrow), sharex=True)
    for ax, sp in zip(np.atleast_1d(axes).ravel(), common):
        g = obs.loc[obs["species"] == sp]
        series = g.groupby("year")["value"].mean().sort_index()
        series = series.loc[series.index >= 2000]
        ax.axvspan(TEST_YEARS[0] - 0.5, TEST_YEARS[-1] + 0.5, color=GRID, alpha=0.55, zorder=0)
        ax.plot(series.index, series.to_numpy(), color=INK_2, lw=1.4, zorder=2,
                label="observed mean")
        for who in ("bfm", "aurora"):
            xs, ys = [], []
            for y in TEST_YEARS:
                fr = by_year.get((who, y))
                if fr is None:
                    continue
                sub = fr.loc[fr["species"] == sp, "y_pred"]
                if len(sub):
                    xs.append(y)
                    ys.append(float(sub.mean()))
            if xs:
                ax.plot(xs, ys, MODEL_MARKER[who], color=MODEL_COLOUR[who], ms=5.5,
                        mec="white", mew=0.7, ls="none", zorder=4, label=MODEL_LABEL[who])
        ax.set_title(str(sp)[:30], style="italic", pad=3)
        ax.grid(True, zorder=0)
    for ax in np.atleast_1d(axes).ravel()[len(common):]:
        ax.set_visible(False)
    for ax in np.atleast_1d(axes).reshape(nrow, ncol)[-1]:
        ax.set_xlabel("year")
    np.atleast_1d(axes).ravel()[0].legend(loc="upper left", fontsize=6)
    fig.suptitle(f"Task {task} — observed national mean, with L3 LoRA r=4 predictions for the "
                 f"shaded test years", x=0.006, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save(fig, out_dir, f"figure_A_trajectories_{task}",
                sources=[{"from": f"artefacts/panel_{task}.parquet"}])


def figure_c(panels: dict[str, pd.DataFrame], out_dir: str | Path,
             published: dict[str, pd.DataFrame] | None = None) -> dict[str, str]:
    """C — does the panel reproduce the scheme's own published national signal?

    The trajectory is built the way the schemes build theirs — each species indexed to a
    base period first, then averaged; averaging raw counts is dominated by whichever
    species are abundant. The lower row is what gate G1 tests: the distribution of
    per-species rank correlation.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    use_house_style()
    published = published or {}
    have = [t for t in ("A", "C") if published.get(t) is not None]
    if not have:
        return {}
    fig, axes = plt.subplots(2, len(have), figsize=(5.8 * len(have), 6.4), squeeze=False)

    for col, task in enumerate(have):
        panel = panels[task]
        obs = panel.loc[panel["observed"]]
        pub = published[task]
        shared = sorted(set(obs["species"].unique()) & set(pub["species"].unique()))
        if not shared:
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            continue
        ours = obs.loc[obs["species"].isin(shared)]
        theirs = pub.loc[pub["species"].isin(shared)]
        lo = max(int(ours["year"].min()), int(theirs["year"].min()))
        hi = min(int(ours["year"].max()), int(theirs["year"].max()))
        ours = ours.loc[ours["year"].between(lo, hi)]
        theirs = theirs.loc[theirs["year"].between(lo, hi)]

        def indexed(df: pd.DataFrame, value: str, base_years: int = 5) -> pd.Series:
            """Index each species to the mean of its first ``base_years`` years, then
            average the indices; a single-year base turns thin first years into spikes."""
            cols = []
            for _, g in df.groupby("species", observed=True):
                s_ = g.groupby("year")[value].mean().sort_index()
                if len(s_) < 3:
                    continue
                base = float(s_.iloc[:base_years].mean())
                if not np.isfinite(base) or base == 0:
                    continue
                cols.append(s_ / base)
            return pd.concat(cols, axis=1).mean(axis=1) if cols else pd.Series(dtype=float)

        a, b = indexed(ours, "value"), indexed(theirs, "index")
        ax = axes[0, col]
        if not a.empty and not b.empty:
            j = a.index.intersection(b.index)
            ax.plot(a.loc[j].index, a.loc[j], color=BLUE, lw=1.8, zorder=3,
                    label="this benchmark's panel")
            ax.plot(b.loc[j].index, b.loc[j], color=INK_2, lw=1.4, ls=(0, (5, 2)), zorder=2,
                    label="scheme's published index")
            r = float(np.corrcoef(a.loc[j], b.loc[j])[0, 1])
            ax.set_title(f"Task {task} — {len(shared)} shared species, r = {r:.3f}", pad=4)
            ax.set_ylabel(f"index (base = mean of {int(j.min())}–{int(j.min()) + 4})")
            ax.legend(loc="best")
        ax.set_xlabel("year")
        ax.grid(True, zorder=0)

        merged = (ours.groupby(["species", "year"], observed=True)["value"].mean()
                  .rename("ours").reset_index()
                  .merge(theirs.rename(columns={"index": "pub"}), on=["species", "year"]))
        rhos = []
        for _, g in merged.groupby("species", observed=True):
            if len(g) < 5:
                continue
            v = spearmanr(g.sort_values("year")["ours"], g.sort_values("year")["pub"]).statistic
            if np.isfinite(v):
                rhos.append(float(v))
        ax2 = axes[1, col]
        if rhos:
            ax2.hist(rhos, bins=np.linspace(-1, 1, 25), color=BLUE, alpha=0.85,
                     edgecolor="white", linewidth=0.5, zorder=3)
            med = float(np.median(rhos))
            ax2.axvline(med, color=ORANGE, lw=1.6, zorder=4)
            ax2.annotate(f"median {med:.3f}", (med, ax2.get_ylim()[1] * 0.92),
                         textcoords="offset points", xytext=(6, 0), color=ORANGE, fontsize=7)
            ax2.set_title(f"per-species rank correlation — {len(rhos)} species (gate G1)", pad=4)
        ax2.set_xlabel("Spearman ρ, panel vs published, per species")
        ax2.set_ylabel("species")
        ax2.grid(axis="y", zorder=0)

    fig.suptitle("Panel against the monitoring scheme's own published index — aggregate above, "
                 "per-species below", x=0.006, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return save(fig, out_dir, "figure_C_published_trend",
                sources=[{"from": "data/raw/*/published trends"}])


def load_published() -> dict[str, pd.DataFrame]:
    """The two schemes' own published indices, normalised to (species, year, index)."""
    out: dict[str, pd.DataFrame] = {}
    a = project_root() / "data/raw/task_a/published_trends/published_index_A.csv"
    if a.exists():
        out["A"] = pd.read_csv(a)
    c = project_root() / "data/raw/ukbms/collated_2021_extract/data/ukbms_collatedindices2021.csv"
    if c.exists():
        d = pd.read_csv(c, dtype=str)
        d = d.loc[d["COUNTRY"] == "UK"]
        out["C"] = pd.DataFrame({"species": d["SPECIES"],
                                 "year": pd.to_numeric(d["YEAR"], errors="coerce"),
                                 "index": pd.to_numeric(d["COLLATED_INDEX"], errors="coerce")}).dropna()
    return out
