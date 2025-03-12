from datetime import datetime

import torch

from aurora import AuroraSmall, Batch, Metadata, rollout

def test_load_and_random_prediction(model):
    prediction = model.forward(batch)

    print(prediction.surf_vars["2t"])

def prediction_aurora(model):
    model.eval()

    with torch.inference_mode():
        pred = model.forward(batch)
        print(pred)

def rollout_aurora(model):
    with torch.inference_mode():
        preds = [pred.to("cpu") for pred in rollout(model, batch, steps=2)]
        print(preds)

if __name__ == "__main__":

    batch = Batch(
    surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
    static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
    atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
    metadata=Metadata(
        lat=torch.linspace(90, -90, 17),
        lon=torch.linspace(0, 360, 32 + 1)[:-1],
        time=(datetime(2020, 6, 1, 12, 0),),
        atmos_levels=(100, 250, 500, 850),
    ),
    )

    model = AuroraSmall()

    model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
    model.to("cuda")

    test_load_and_random_prediction(model)
    prediction_aurora(model)
    rollout_aurora(model)