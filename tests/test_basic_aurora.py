from datetime import datetime

import pytest
import torch
from aurora import Aurora, AuroraSmall, Batch, Metadata, rollout

LAT_RES = 16  # pick a number that is multiple of patch_size (default 4)
LATITUDES = torch.linspace(90, -90, LAT_RES)
LON_RES = 32  # pick a number that is multiple of patch_size (default 4)
LONGITUDES = torch.linspace(0, 360, LON_RES + 1)[:-1]

batch = Batch(
    surf_vars={
        k: torch.randn(1, 2, LAT_RES, LON_RES) for k in ("2t", "10u", "10v", "msl")
    },
    static_vars={k: torch.randn(LAT_RES, LON_RES) for k in ("lsm", "z", "slt")},
    atmos_vars={
        k: torch.randn(1, 2, 4, LAT_RES, LON_RES) for k in ("z", "u", "v", "t", "q")
    },
    metadata=Metadata(
        lat=LATITUDES,
        lon=LONGITUDES,
        time=(datetime(2020, 6, 1, 12, 0),),
        atmos_levels=(100, 250, 500, 850),
    ),
)


def get_model(small=True):
    if small:
        model = AuroraSmall()
        model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
    else:
        model = Aurora(use_lora=False)
        model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
    model.to("cuda")
    return model


def do_test_forward(aurora_model, batch):
    with torch.inference_mode():
        prediction = aurora_model.forward(batch)

    surf_vars_pred_names = set(prediction.surf_vars.keys())
    surf_vars_target_names = set(batch.surf_vars.keys())
    assert (
        surf_vars_pred_names == surf_vars_target_names
    ), f"surf_vars.keys() mismatch: {surf_vars_pred_names}, {surf_vars_target_names}"
    shape_2t_pred = prediction.surf_vars["2t"].shape
    shape_2t_target = batch.surf_vars["2t"].shape
    # surf_vars: (b, t, h, w)
    # - b: must be the same
    # - t: will be 1
    # - h, w: must be the same (except patch_size rounding)
    for i in [0, 2, 3]:
        assert (
            shape_2t_pred[i] == shape_2t_target[i]
        ), f"shape 2t mismatch (i={i}): {shape_2t_pred}, {shape_2t_target}"


def do_test_rollout(aurora_model, batch):
    with torch.inference_mode():
        preds = [pred.to("cpu") for pred in rollout(aurora_model, batch, steps=2)]
        assert len(preds) == 2
        print(preds)


# some fixtures for pytest


@pytest.fixture
def aurora_model_small():
    return get_model(small=True)


@pytest.fixture
def aurora_model_big():
    return get_model(small=False)


# now the "test_*" functions that are called automatically by pytest


def test_load_and_random_prediction_small(aurora_model_small):
    do_test_forward(aurora_model_small, batch)


def test_load_and_random_prediction_big(aurora_model_big):
    do_test_forward(aurora_model_big, batch)


def test_rollout_small(aurora_model_small):
    do_test_rollout(aurora_model_small, batch)


def test_rollout_big(aurora_model_big):
    do_test_rollout(aurora_model_big, batch)


# test small via cli
if __name__ == "__main__":
    aurora_model = get_model(small=True)
    do_test_forward(aurora_model, batch)
    do_test_rollout(aurora_model, batch)
