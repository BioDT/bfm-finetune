"""Tasks A, B and C -> panel.

Each parser returns a frame obeying ``panel.PANEL_COLUMNS``. The two ways a masked cell
arises differ by design: Task A ships no absences, so negatives are reconstructed — a
species is zero at a route-year only where a completed visit exists; Task C ships its own
mask (``SITE_INDEX = -2`` means the index was not estimable) and those rows are masked,
never clipped to zero.
"""

import csv
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .panel import GRID, PANEL_COLUMNS, GridSpec, build_panel, coerce_panel

TASK_A_EFFORT = 1.0
UKBMS_NOT_ESTIMATED = -2.0


def _read_tsv(path: Path, usecols: list[str], dtype: dict[str, Any] | None = None,
              chunksize: int | None = None):
    return pd.read_csv(path, sep="\t", usecols=usecols, dtype=dtype or "string",
                       quoting=csv.QUOTE_NONE, on_bad_lines="error", chunksize=chunksize,
                       low_memory=False)


def parse_task_a(dwca_dir: str | Path, grid: GridSpec = GRID, *,
                 klass: str = "Aves", rank: str = "species") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Swedish Bird Survey fixed routes (Standardrutterna) -> panel A.

    The archive publishes exactly one event per route-year and no absence records, so every
    species in the scheme list is a true zero at a surveyed route where it was not recorded.
    """
    dwca_dir = Path(dwca_dir)
    ev = _read_tsv(dwca_dir / "event.txt",
                   ["eventID", "eventDate", "locality", "locationID", "county",
                    "decimalLatitude", "decimalLongitude", "geodeticDatum", "startDayOfYear"])
    datum = set(ev["geodeticDatum"].dropna().unique())
    if datum != {"EPSG:4326"}:
        raise ValueError(f"task A expects WGS84 coordinates, found {datum}")

    ev = ev.assign(
        unit_id=ev["locality"].str.rsplit(":", n=1).str[-1].astype("string"),
        year=ev["eventDate"].str[:4].astype("int32"),
        lat=ev["decimalLatitude"].astype("float64"),
        lon=ev["decimalLongitude"].astype("float64"),
        doy=pd.to_numeric(ev["startDayOfYear"], errors="coerce"))

    dup = ev.duplicated(subset=["unit_id", "year"]).sum()
    if dup:
        raise ValueError(f"task A assumes one event per route-year; found {dup} duplicates")

    visits = ev.loc[:, ["unit_id", "year", "lat", "lon"]].assign(effort=TASK_A_EFFORT, completed=True)

    occ = _read_tsv(dwca_dir / "occurrence.txt",
                    ["eventID", "scientificName", "class", "taxonRank", "individualCount"])
    n_occ = len(occ)
    occ = occ.loc[(occ["class"] == klass) & (occ["taxonRank"] == rank)]
    occ = occ.merge(ev.loc[:, ["eventID", "unit_id", "year"]], on="eventID", how="inner")
    records = occ.rename(columns={"scientificName": "species"}).assign(
        value=pd.to_numeric(occ["individualCount"], errors="coerce").astype("float64"))
    records = records.loc[:, ["unit_id", "year", "species", "value"]]
    if not np.isfinite(records["value"].to_numpy()).all():
        raise ValueError("task A has non-numeric individualCount values")

    species = sorted(records["species"].dropna().unique().tolist())
    panel = build_panel(visits, records, task="A", value_type="count", species=species, grid=grid)

    info = {
        "events": int(len(ev)), "occurrence_rows": int(n_occ),
        "occurrence_rows_kept": int(len(occ)),
        "occurrence_rows_dropped_not_target_rank_or_class": int(n_occ - len(occ)),
        "routes": int(ev["unit_id"].nunique()),
        "species_in_panel": len(species),
        "year_min": int(ev["year"].min()), "year_max": int(ev["year"].max()),
        "day_of_year_range": [int(ev["doy"].min()), int(ev["doy"].max())],
        "counties": int(ev["county"].nunique()),
        "coordinate_note": "event coordinates are the centroid of a 25x25 km survey square, "
                           "not the route itself (coordinateUncertaintyInMeters = 17700)",
    }
    return panel, info


def parse_task_b(dwca_dir: str | Path, grid: GridSpec = GRID, *,
                 chunksize: int = 2_000_000) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Swedish NFI vegetation -> panel B, as per-species cell prevalence.

    The archive is a fully crossed checklist, so absences are recorded rather than
    reconstructed. The unit is the 0.25 degree cell, not the plot: the archive publishes no
    plot identifier and its obfuscated coordinates collide across nearby plots.
    """
    dwca_dir = Path(dwca_dir)
    ev = _read_tsv(dwca_dir / "event.txt",
                   ["id", "year", "decimalLatitude", "decimalLongitude", "geodeticDatum"])
    datum = set(ev["geodeticDatum"].dropna().unique())
    if datum != {"EPSG:4326"}:
        raise ValueError(f"task B expects WGS84 coordinates, found {datum}")

    ci, cj = grid.to_cell(ev["decimalLatitude"].astype("float64"),
                          ev["decimalLongitude"].astype("float64"))
    ev = ev.assign(cell_i=ci, cell_j=cj, year=ev["year"].astype("int32"))
    n_events = len(ev)
    ev = ev.loc[(ev["cell_i"] >= 0) & (ev["cell_j"] >= 0)]

    ev["unit_id"] = (ev["cell_i"].astype(str) + "_" + ev["cell_j"].astype(str)).astype("string")
    plots = ev.groupby(["unit_id", "year"], observed=True).size().rename("effort").reset_index()
    lookup = ev.set_index("id")[["unit_id", "year"]]

    present = None
    n_rows = 0
    status_counts: dict[str, int] = {}
    ranks: dict[str, int] = {}
    for chunk in _read_tsv(dwca_dir / "occurrence.txt",
                           ["eventID", "scientificName", "occurrenceStatus", "taxonRank"],
                           chunksize=chunksize):
        n_rows += len(chunk)
        for k, v in chunk["occurrenceStatus"].value_counts().items():
            status_counts[str(k)] = status_counts.get(str(k), 0) + int(v)
        for k, v in chunk.drop_duplicates("scientificName").set_index("scientificName")["taxonRank"].items():
            ranks[str(v)] = ranks.get(str(v), 0)
        hit = chunk.loc[chunk["occurrenceStatus"].str.lower() == "present"]
        if hit.empty:
            continue
        hit = hit.join(lookup, on="eventID", how="inner")
        agg = hit.groupby(["unit_id", "year", "scientificName"], observed=True).size().rename("n_present")
        present = agg if present is None else present.add(agg, fill_value=0)

    if present is None:
        raise ValueError("task B: no present records found")
    present = present.reset_index().rename(columns={"scientificName": "species"})

    taxa = sorted(present["species"].dropna().unique().tolist())
    frame = plots.merge(pd.DataFrame({"species": pd.Series(taxa, dtype="string")}), how="cross")
    frame = frame.merge(present, on=["unit_id", "year", "species"], how="left")
    frame["n_present"] = frame["n_present"].fillna(0.0)

    cells = frame["unit_id"].str.split("_", n=1, expand=True)
    lat, lon = grid.cell_centre(cells[0].astype("int32"), cells[1].astype("int32"))
    panel = coerce_panel(frame.assign(
        task="B", cell_i=cells[0].astype("int32"), cell_j=cells[1].astype("int32"),
        lat=lat, lon=lon, value=frame["n_present"] / frame["effort"],
        value_type="prevalence", observed=True).loc[:, list(PANEL_COLUMNS)])

    info = {
        "events": n_events, "events_on_grid": int(len(ev)),
        "events_off_grid": int(n_events - len(ev)),
        "occurrence_rows": n_rows, "occurrence_status_counts": status_counts,
        "taxa": len(taxa), "taxon_ranks_seen": sorted(ranks),
        "cell_years": int(len(plots)), "cells": int(plots["unit_id"].nunique()),
        "plots_per_cell_year": {
            "min": int(plots["effort"].min()), "median": float(plots["effort"].median()),
            "max": int(plots["effort"].max())},
        "unit_note": "unit is the 0.25 degree cell; the archive publishes no plot identifier "
                     "(locationID empty in all events) and obfuscated coordinates collide",
    }
    return panel, info


def _ukbms_columns(fieldnames: Iterable[str]) -> dict[str, str]:
    """Editions alternate between ``SITE_CODE`` and ``SITE CODE`` spellings."""
    return {name: name.strip().upper().replace(" ", "_") for name in fieldnames}


OSGB36 = "EPSG:27700"
IRISH_GRID = "EPSG:29903"
_GRID_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def _parse_gridref(ref: str) -> tuple[float, float] | None:
    """Grid reference -> (easting, northing): two leading letters is the GB National Grid,
    one is the Irish Grid."""
    ref = str(ref).strip().upper().replace(" ", "")
    m = re.fullmatch(r"([A-Z]{1,2})(\d+)", ref)
    if not m or len(m.group(2)) % 2:
        return None
    letters, digits = m.group(1), m.group(2)
    if len(letters) == 2:
        i1, i2 = (_GRID_LETTERS.index(c) for c in letters)
        e0 = ((i1 % 5) * 5 - 10 + (i2 % 5)) * 100000
        n0 = ((4 - i1 // 5) * 5 - 5 + (4 - i2 // 5)) * 100000
    else:
        i = _GRID_LETTERS.index(letters)
        e0, n0 = (i % 5) * 100000, (4 - i // 5) * 100000
    half = len(digits) // 2
    scale = 10 ** (5 - half)
    return e0 + int(digits[:half]) * scale, n0 + int(digits[half:]) * scale


def load_ukbms_locations(csv_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """UKBMS site locations -> ``unit_id, lat, lon`` in WGS84.

    ``Easting``/``Northing`` reproduce the OSGB grid reference exactly for GB and the Isle
    of Man, but not for Northern Ireland (Irish Grid) or the Channel Islands. NI sites are
    re-derived from their grid reference and reprojected; Channel Islands sites are dropped
    rather than reprojected from an unidentified frame.
    """
    from pyproj import Transformer

    raw = pd.read_csv(csv_path, dtype="string", encoding="cp1252")
    raw["easting"] = pd.to_numeric(raw["Easting"], errors="coerce")
    raw["northing"] = pd.to_numeric(raw["Northing"], errors="coerce")

    parsed = raw["Gridreference"].map(lambda r: _parse_gridref(r) if pd.notna(r) else None)
    raw["ref_e"] = [p[0] if p else np.nan for p in parsed]
    raw["ref_n"] = [p[1] if p else np.nan for p in parsed]
    agrees = (np.isclose(raw["ref_e"], raw["easting"]) & np.isclose(raw["ref_n"], raw["northing"]))

    frames, dropped = [], {}
    for crs, sel, e_col, n_col in (
            (OSGB36, agrees, "easting", "northing"),
            (IRISH_GRID, (~agrees) & (raw["Country"] == "Northern Ireland") & raw["ref_e"].notna(),
             "ref_e", "ref_n")):
        sub = raw.loc[sel]
        if sub.empty:
            continue
        tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = tf.transform(sub[e_col].to_numpy(float), sub[n_col].to_numpy(float))
        frames.append(pd.DataFrame({"unit_id": sub["Site_Number"].astype("string"),
                                    "lat": lat, "lon": lon, "crs": crs,
                                    "country": sub["Country"], "survey_type": sub["Survey_type"]}))

    used = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    excluded = raw.loc[~raw["Site_Number"].astype("string").isin(used["unit_id"])]
    for country, g in excluded.groupby("Country", dropna=False):
        dropped[str(country)] = int(len(g))

    info = {
        "sites_in_file": int(len(raw)),
        "sites_georeferenced": int(len(used)),
        "by_crs": used["crs"].value_counts().to_dict() if len(used) else {},
        "dropped_by_country": dropped,
        "drop_reason": "Easting/Northing disagree with the site's own grid reference and the "
                       "frame is not identified; not guessed",
        "survey_type_counts": used["survey_type"].value_counts().to_dict() if len(used) else {},
    }
    return used.loc[:, ["unit_id", "lat", "lon"]], info


def _dedupe_site_years(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve repeated (site, species, year) keys in the UKBMS indices.

    Identical repeats collapse; a sentinel paired with a real estimate keeps the estimate;
    two differing real estimates are dropped rather than averaged.
    """
    key = ["unit_id", "species", "year"]
    dup_mask = df.duplicated(key, keep=False)
    if not dup_mask.any():
        return df, {"duplicate_keys": 0}

    dups = df.loc[dup_mask]
    stats = dups.groupby(key, observed=True)["value"].agg(["nunique", "size"])
    conflicting = stats.loc[stats["nunique"] > 1].index

    keep = df.set_index(key)
    dropped = keep.index.isin(conflicting)
    out = keep.loc[~dropped].reset_index()
    out = out.sort_values("observed", ascending=False).drop_duplicates(key, keep="first")

    return out.reset_index(drop=True), {
        "duplicate_keys": int(stats.shape[0]),
        "keys_collapsed": int((stats["nunique"] <= 1).sum()),
        "conflicting_keys_dropped": int(len(conflicting)),
        "rows_dropped_as_conflicting": int(dropped.sum()),
        "rows_removed_total": int(len(df) - len(out)),
        "policy": "identical repeats collapse; sentinel+estimate keeps the estimate; two "
                  "differing estimates are dropped, never averaged",
    }


def parse_task_c(csv_path: str | Path, locations: pd.DataFrame, grid: GridSpec = GRID,
                 *, years: tuple[int, int] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """UKBMS site indices -> panel C.

    ``locations`` must supply ``unit_id, lat, lon`` in WGS84; the site-indices product ships
    no coordinates of its own.
    """
    raw = pd.read_csv(csv_path, dtype="string")
    raw = raw.rename(columns=_ukbms_columns(raw.columns))
    needed = {"SITE_CODE", "SPECIES", "YEAR", "SITE_INDEX"}
    if not needed <= set(raw.columns):
        raise ValueError(f"UKBMS csv missing {needed - set(raw.columns)}")

    idx = pd.to_numeric(raw["SITE_INDEX"], errors="coerce").astype("float64")
    unexpected = sorted(np.unique(idx[(idx < 0) & (idx != UKBMS_NOT_ESTIMATED)]).tolist())
    if unexpected:
        raise ValueError(f"unknown negative SITE_INDEX sentinels {unexpected}; only "
                         f"{UKBMS_NOT_ESTIMATED} (index not estimable) is documented")

    observed = (idx != UKBMS_NOT_ESTIMATED).to_numpy()
    df = pd.DataFrame({
        "task": "C",
        "unit_id": raw["SITE_CODE"].astype("string"),
        "year": pd.to_numeric(raw["YEAR"]).astype("int32"),
        "species": raw["SPECIES"].astype("string"),
        "value": np.where(observed, idx.to_numpy(), np.nan),
        "value_type": "index",
        "effort": 1.0,
        "observed": observed,
    })
    if years:
        df = df.loc[df["year"].between(*years)]
    df, dedupe = _dedupe_site_years(df)

    loc = locations.loc[:, ["unit_id", "lat", "lon"]].copy()
    loc["unit_id"] = loc["unit_id"].astype("string")
    before = df["unit_id"].nunique()
    df = df.merge(loc.drop_duplicates("unit_id"), on="unit_id", how="inner")
    matched = df["unit_id"].nunique()

    ci, cj = grid.to_cell(df["lat"], df["lon"])
    df = df.assign(cell_i=ci, cell_j=cj)
    off_grid = int((df["cell_i"] < 0).sum())
    df = df.loc[df["cell_i"] >= 0]

    panel = coerce_panel(df.loc[:, list(PANEL_COLUMNS)])
    info = {
        "rows_in_source": int(len(raw)),
        "sites_in_source": int(before), "sites_matched_to_location": int(matched),
        "sites_unmatched": int(before - matched),
        "rows_dropped_off_grid": off_grid,
        "rows_masked_sentinel": int((~observed).sum()),
        "sentinel_value": UKBMS_NOT_ESTIMATED,
        "deduplication": dedupe,
        "effort_note": "the site-indices product publishes no per-site visit count; effort is "
                       "1.0 and the GAM's visit-coverage screen is what the mask encodes",
    }
    return panel, info
