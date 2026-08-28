"""CPU-only unit tests for the eumon benchmark package. No data, weights or GPU needed."""

import numpy as np
import pandas as pd
import pytest

from bfm_finetune.eumon import splits as S
from bfm_finetune.eumon.common import tables as T
from bfm_finetune.eumon.eval import metrics
from bfm_finetune.eumon.eval import nulls as N
from bfm_finetune.eumon.panel import (GRID, PanelContractError, build_panel, coerce_panel,
                                      validate_panel)


def toy_panel(years=range(2000, 2005), n_units=6, species=("sp1", "sp2")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    visits = pd.DataFrame([
        {"unit_id": f"u{k}", "year": y, "lat": 40.0 + k, "lon": 5.0 + k,
         "effort": 1.0, "completed": True}
        for y in years for k in range(n_units)])
    records = pd.DataFrame([
        {"unit_id": f"u{k}", "year": y, "species": sp,
         "value": float(rng.integers(0, 20))}
        for y in years for k in range(n_units) for sp in species])
    return build_panel(visits, records, task="A", value_type="count", species=species)


def test_grid_round_trip():
    ci, cj = GRID.to_cell([40.13, 32.0, 71.8], [5.4, -25.0, 44.8])
    lat, lon = GRID.cell_centre(ci, cj)
    ri, rj = GRID.to_cell(lat, lon)
    assert np.array_equal(ci, ri) and np.array_equal(cj, rj)
    assert (ci >= 0).all() and (cj >= 0).all()


def test_grid_outside_domain_is_minus_one():
    ci, cj = GRID.to_cell([10.0, 80.0, np.nan], [5.0, 5.0, 5.0])
    assert (ci == -1).all()


def test_build_panel_reconstructs_zeros_and_masks():
    visits = pd.DataFrame([
        {"unit_id": "a", "year": 2001, "lat": 40.0, "lon": 5.0, "effort": 1.0, "completed": True},
        {"unit_id": "b", "year": 2001, "lat": 41.0, "lon": 6.0, "effort": 1.0, "completed": False},
    ])
    records = pd.DataFrame([{"unit_id": "a", "year": 2001, "species": "sp1", "value": 3.0}])
    panel = build_panel(visits, records, task="A", value_type="count", species=("sp1", "sp2"))

    a = panel.set_index(["unit_id", "species"])
    assert a.loc[("a", "sp1"), "value"] == 3.0
    assert a.loc[("a", "sp2"), "value"] == 0.0          # completed visit, no record: true zero
    assert not a.loc[("b", "sp1"), "observed"]           # incomplete visit: masked
    assert np.isnan(a.loc[("b", "sp1"), "value"])
    validate_panel(panel)


def test_validate_panel_rejects_masked_value():
    panel = toy_panel()
    bad = panel.copy()
    bad.loc[0, "observed"] = False                       # value stays finite: contract broken
    with pytest.raises(PanelContractError):
        validate_panel(coerce_panel(bad))


def test_persistence_null_reads_previous_year():
    panel = toy_panel()
    split = N.Split(tuple(range(2000, 2004)), 2004)
    test = N.test_frame(panel, split)
    pred = N.persistence(panel, split, test)
    prev = panel.loc[panel["year"] == 2003].set_index(["unit_id", "species"])["value"]
    expect = np.array([prev.get((u, s), np.nan)
                       for u, s in zip(test["unit_id"], test["species"])], dtype=float)
    assert np.allclose(pred, expect, equal_nan=True)


def test_compute_nulls_covers_all_names():
    panel = toy_panel()
    preds = N.compute_nulls(panel, N.Split(tuple(range(2000, 2004)), 2004))
    assert all(name in preds.columns for name in N.NULL_NAMES)


def test_skill_score_identities():
    y = np.array([1.0, 4.0, 9.0, 2.0])
    ref = np.array([2.0, 3.0, 7.0, 3.0])
    assert metrics.skill_score(y, ref, ref) == 0.0
    assert metrics.skill_score(y, y, ref) == 1.0
    assert np.isnan(metrics.skill_score(y, ref, y))      # zero-error reference is undefined


def test_headline_is_minimum_across_references():
    skill = {"vs_a": {"skill_score": 0.4}, "vs_b": {"skill_score": -0.2},
             "vs_c": {"skill_score": float("nan")}}
    head = N.headline_skill(skill)
    assert head == {"reference": "vs_b", "skill_score": -0.2}


def test_evaluate_carries_required_keys():
    panel = toy_panel()
    obs = panel.loc[panel["observed"] & (panel["year"] == 2004)]
    frame = obs.rename(columns={"value": "y_true"}).assign(y_pred=obs["value"] + 1.0)
    rec = metrics.evaluate(frame, value_type="count")
    for key in ("n", "rmse", "rmse_log", "spatial_rho", "calibration"):
        assert key in rec
    assert "temporal_rho" not in rec                     # single-year frame


def test_spatial_folds_disjoint_and_buffered():
    panel = toy_panel(n_units=30)
    spec = S.BlockSpec(block_deg=1.0, n_folds=3, buffer_km=50.0)
    assignment = S.spatial_block_folds(panel, spec)
    for fold in range(spec.n_folds):
        rec = S.verify(assignment, fold)
        assert rec["buffer_respected"]
    train, test = S.apply_fold(panel, assignment, 0)
    assert not set(train["unit_id"]) & set(test["unit_id"])


def test_chronological_split_rejects_overlap():
    panel = toy_panel()
    with pytest.raises(ValueError):
        S.chronological_split(panel, train_end=2003, val_years=(2004,), test_years=(2004,))


def test_tables_refuse_pooled_number():
    with pytest.raises(T.PooledNumberRefused):
        T.refuse_pooled()


def test_task_table_renders_nulls_before_models():
    rec = {"skill": {"vs_persistence": {"skill_score": 0.1, "skill_score_log1p": 0.2}},
           "skill_vs_strongest_null": {"reference": "vs_persistence", "skill_score": 0.1},
           "rmse_log": 1.0, "spatial_rho": {"mean": 0.5}, "n": 10, "n_species": 2}
    table = T.task_table("A", {"nulls": {"persistence": [rec]},
                               "model": {"bfm_l2": [rec]}},
                         reference="persistence", test_year=2019)
    md = T.to_markdown(table)
    assert md.index("*Null models*") < md.index("*Foundation models*")
    assert "no pooled" not in md                          # refusal lives in code, not prose
