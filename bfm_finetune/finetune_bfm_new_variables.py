import importlib
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from bfm_model.bfm.model import BFM, BFMRollout
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from bfm_finetune import bfm_mod
from bfm_finetune.dataloaders.dataloader_utils import custom_collate_fn
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoLifeCLEFSpeciesDataset,
)
from bfm_finetune.dataloaders.toy_dataset.dataloader import ToyClimateDataset
from bfm_finetune.finetune_new_variables import (
    save_checkpoint,
    train_epoch,
    validate_epoch,
)
from bfm_finetune.paths import REPO_FOLDER, STORAGE_DIR
from bfm_finetune.plots_v2 import plot_eval
from bfm_finetune.utils import (
    get_lat_lon_ranges,
    load_checkpoint,
)

# checkpoint_file = STORAGE_DIR / "weights" / "epoch=268-val_loss=0.00493.ckpt"
# has _latent_parameter_list
checkpoint_file = STORAGE_DIR / "weights" / "epoch=00-val_loss=0.32124.ckpt"

if not os.path.exists(checkpoint_file):
    raise ValueError(f"checkpoint not found: {checkpoint_file}")

# checkpoint = torch.load(checkpoint_file)
# checkpoint.keys()
# # dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers'])
# checkpoint["state_dict"].keys()  # here all the weights


# raise ValueError(REPO_FOLDER)
bfm_config_path = REPO_FOLDER / "bfm-model/bfm_model/bfm/configs"
cwd = Path(os.getcwd())
bfm_config_path = str(bfm_config_path.relative_to(cwd))
bfm_config_path = f"../bfm-model/bfm_model/bfm/configs"
print(bfm_config_path)
with initialize(version_base=None, config_path=bfm_config_path, job_name="test_app"):
    cfg = compose(config_name="train_config.yaml")

swin_params = {}
if cfg.model.backbone == "swin":
    selected_swin_config = cfg.model_swin_backbone[cfg.model.swin_backbone_size]
    swin_params = {
        "swin_encoder_depths": tuple(selected_swin_config.encoder_depths),
        "swin_encoder_num_heads": tuple(selected_swin_config.encoder_num_heads),
        "swin_decoder_depths": tuple(selected_swin_config.decoder_depths),
        "swin_decoder_num_heads": tuple(selected_swin_config.decoder_num_heads),
        "swin_window_size": tuple(selected_swin_config.window_size),
        "swin_mlp_ratio": selected_swin_config.mlp_ratio,
        "swin_qkv_bias": selected_swin_config.qkv_bias,
        "swin_drop_rate": selected_swin_config.drop_rate,
        "swin_attn_drop_rate": selected_swin_config.attn_drop_rate,
        "swin_drop_path_rate": selected_swin_config.drop_path_rate,
    }

# BFM args
bfm_args = dict(
    surface_vars=(cfg.model.surface_vars),
    edaphic_vars=(cfg.model.edaphic_vars),
    atmos_vars=(cfg.model.atmos_vars),
    climate_vars=(cfg.model.climate_vars),
    species_vars=(cfg.model.species_vars),
    vegetation_vars=(cfg.model.vegetation_vars),
    land_vars=(cfg.model.land_vars),
    agriculture_vars=(cfg.model.agriculture_vars),
    forest_vars=(cfg.model.forest_vars),
    redlist_vars=(cfg.model.redlist_vars),
    misc_vars=(cfg.model.misc_vars),
    atmos_levels=cfg.data.atmos_levels,
    species_num=cfg.data.species_number,
    H=cfg.model.H,
    W=cfg.model.W,
    num_latent_tokens=cfg.model.num_latent_tokens,
    backbone_type=cfg.model.backbone,
    patch_size=cfg.model.patch_size,
    embed_dim=cfg.model.embed_dim,
    num_heads=cfg.model.num_heads,
    head_dim=cfg.model.head_dim,
    depth=cfg.model.depth,
    learning_rate=cfg.finetune.lr,
    weight_decay=cfg.finetune.wd,
    batch_size=cfg.finetune.batch_size,
    td_learning=cfg.finetune.td_learning,
    # ground_truth_dataset=None,
    strict=False,  # False if loading from a pre-trained with PEFT checkpoint
    peft_r=cfg.finetune.rank,
    lora_alpha=cfg.finetune.lora_alpha,
    d_initial=cfg.finetune.d_initial,
    peft_dropout=cfg.finetune.peft_dropout,
    peft_steps=cfg.finetune.rollout_steps,
    peft_mode=cfg.finetune.peft_mode,
    use_lora=cfg.finetune.use_lora,
    use_vera=cfg.finetune.use_vera,
    rollout_steps=cfg.finetune.rollout_steps,
    # lora_steps=cfg.finetune.rollout_steps, # 1 month
    # lora_mode=cfg.finetune.lora_mode, # every step + layers #single
    **swin_params,
)

base_model = BFMRollout.load_from_checkpoint(
    checkpoint_path=checkpoint_file, **bfm_args
)
# model = BFM.load_from_checkpoint(
#     checkpoint_path=checkpoint_file,
#     **bfm_args,
# )
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])


# FINETUNE
finetune_config_path = "."  # f"bfm_finetune"
with initialize(
    version_base=None, config_path=finetune_config_path, job_name="test_app"
):
    finetune_cfg = compose(config_name="finetune_config.yaml")

num_species = finetune_cfg.dataset.num_species
model = bfm_mod.BFMRaw(base_model=base_model, n_species=num_species, mode="train")

device = base_model.device
model.to(device)

params_to_optimize = model.parameters()

# model.to(device)
num_epochs = finetune_cfg.training.epochs

# TODO
# output_dir = HydraConfig.get().runtime.output_dir
output_dir = f"outputs_bfm_finetune_48800"

# THE FOLLOWING IS COPIED FROM finetune_new_variables.py

optimizer = torch.optim.AdamW(
    params_to_optimize,
    lr=finetune_cfg.training.lr,
    weight_decay=0.0001,
    betas=(0.9, 0.95),
    eps=1e-8,
)
criterion = nn.L1Loss()

total_steps = num_epochs
warmup_steps = int(0.05 * total_steps)  # 5 % warm-up
min_lr_ratio = 0.05  # final LR = 5 % of base


def lr_lambda(step):
    if step < warmup_steps:  # linear warm-up
        return step / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine


# scheduler = LambdaLR(optimizer, lr_lambda)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=2000, eta_min=finetune_cfg.training.lr / 10
)


checkpoint_save_path = Path(output_dir) / "checkpoints"


if finetune_cfg.dataset.toy:
    # customizable lat_lon
    lat_lon = get_lat_lon_ranges()
    train_dataset = ToyClimateDataset(
        num_samples=3,
        num_species=num_species,
        lat_lon=lat_lon,
    )
    val_dataset = ToyClimateDataset(
        num_samples=1,
        num_species=num_species,
        lat_lon=lat_lon,
    )
else:
    train_dataset = GeoLifeCLEFSpeciesDataset(
        num_species=num_species,
        mode="train",
        negative_lon_mode=finetune_cfg.dataset.negative_lon_mode,
    )
    val_dataset = GeoLifeCLEFSpeciesDataset(
        num_species=num_species,
        mode="val",
        negative_lon_mode=finetune_cfg.dataset.negative_lon_mode,
    )
    # get lat_lon from dataset
    lat_lon = train_dataset.get_lat_lon()
print("lat-lon", lat_lon[0].shape, lat_lon[1].shape)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=finetune_cfg.training.batch_size,
    shuffle=True,
    collate_fn=custom_collate_fn,
    num_workers=finetune_cfg.dataset.num_workers,
)
# TODO Make it distinct
val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=finetune_cfg.dataset.num_workers,
)

# Load checkpoint if available
_, best_loss = load_checkpoint(model, optimizer, finetune_cfg.training.checkpoint_path)
val_loss = 1_000_000
# mlflow.set_experiment("BFM_Finetune")

plots_dir = Path(output_dir) / "plots"
predictions_dir = Path(output_dir) / "predictions"
if not os.path.exists(plots_dir) or not os.path.exists(predictions_dir):
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

# with mlflow.start_run():
#     mlflow.log_param("num_epochs", cfg.training.epochs)
#     mlflow.log_param("learning_rate", cfg.training.lr)
#     mlflow.log_param("batch_size", cfg.training.batch_size)
for epoch in range(1, num_epochs):
    train_loss = train_epoch(
        model, train_dataloader, optimizer, criterion, scheduler, device
    )

    if epoch % finetune_cfg.training.val_every == 0:
        val_loss = validate_epoch(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")
        # mlflow.log_metric("val_loss", val_loss, step=epoch + 1)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
    # mlflow.log_metric("train_loss", train_loss, step=epoch + 1)

    if val_loss < best_loss:
        best_loss = val_loss
        save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_save_path)
        # mlflow.log_metric("best_loss", best_loss, step=epoch + 1)

# final evaluate
for sample in val_dataloader:
    batch = sample["batch"]  # .to(device)
    batch["species_distribution"] = batch["species_distribution"].to(device)
    target = sample["target"]
    with torch.inference_mode():
        prediction = model.forward(batch)
    unnormalized_preds = val_dataset.scale_species_distribution(
        prediction.clone(), unnormalize=True
    )
    save_path = predictions_dir / "finetune_predictions.pt"
    torch.save(unnormalized_preds, save_path)
    plot_eval(
        batch=batch,
        # prediction_species=prediction,
        prediction_species=unnormalized_preds,
        out_dir=plots_dir,
        save=True,
    )
