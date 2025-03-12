from datetime import datetime

import torch
from aurora import Aurora, AuroraSmall, Batch, Metadata

# BIG model
# model = Aurora(
#     use_lora=False,  # Model was not fine-tuned.
#     autocast=True,  # Use AMP.
# )
# model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

# SMALL model
model = AuroraSmall(
    use_lora=False,  # Model was not fine-tuned.
    autocast=True,  # Use AMP.
)
model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")


# shape 17 become 16 because patches will predict rounded up value (loss check)
batch = Batch(
    surf_vars={k: torch.randn(1, 2, 16, 32) for k in ("2t", "10u", "10v", "msl")},
    static_vars={k: torch.randn(16, 32) for k in ("lsm", "z", "slt")},
    atmos_vars={k: torch.randn(1, 2, 4, 16, 32) for k in ("z", "u", "v", "t", "q")},
    metadata=Metadata(
        lat=torch.linspace(90, -90, 16),
        lon=torch.linspace(0, 360, 32 + 1)[:-1],
        time=(datetime(2020, 6, 1, 12, 0),),
        atmos_levels=(100, 250, 500, 850),
    ),
)

model = model.cuda()
model.train()
model.configure_activation_checkpointing()

# TODO: customize weights or take from reference paper
loss_weights = {
    "surface": {"2t": 1.0, "10u": 1.0, "10v": 1.0, "msl": 1.0},
    "static": {"lsm": 1.0, "z": 1.0, "slt": 1.0},
    "atmos": {
        "z": 1.0,
        "u": 1.0,
        "v": 1.0,
        "t": 1.0,
        "q": 1.0,
    },
}


def get_loss(pred_batch: Batch, ref_batch: Batch):
    loss_total = torch.Tensor([0.0])
    loss_cnt = 0
    # TODO: normalization?
    for var_name, ref_var_value in ref_batch.surf_vars.items():
        pred_var_value = pred_batch.surf_vars[var_name].cpu()
        print("surface", var_name, ref_var_value.shape, pred_var_value.shape)
        loss_var = torch.mean(torch.abs(ref_var_value - pred_var_value))
        loss_total += loss_weights["surface"][var_name] * loss_var
        loss_cnt += 1
    for var_name, ref_var_value in ref_batch.static_vars.items():
        pred_var_value = pred_batch.static_vars[var_name].cpu()
        print("static", var_name, ref_var_value.shape, pred_var_value.shape)
        loss_var = torch.mean(torch.abs(ref_var_value - pred_var_value))
        loss_total += loss_weights["static"][var_name] * loss_var
        loss_cnt += 1
    for var_name, ref_var_value in ref_batch.atmos_vars.items():
        pred_var_value = pred_batch.atmos_vars[var_name].cpu()
        print("atmos", var_name, ref_var_value.shape, pred_var_value.shape)
        loss_var = torch.mean(torch.abs(ref_var_value - pred_var_value))
        loss_total += loss_weights["atmos"][var_name] * loss_var
        loss_cnt += 1
    if loss_cnt:
        loss_total /= loss_cnt
    print(loss_total.shape)
    return loss_total


pred = model.forward(batch)
loss = get_loss(pred, batch)
loss.backward()
