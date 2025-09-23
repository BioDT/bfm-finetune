import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from bfm_model.bfm.model_helpers import (
    find_checkpoint_to_resume_from,
    get_mlflow_logger,
    get_trainer,
    post_training_get_last_checkpoint,
    setup_bfm_model,
    setup_checkpoint_callback,
    setup_fsdp,
)
from hydra import compose, initialize
from omegaconf import OmegaConf


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

from sklearn.model_selection import GroupKFold
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# V1
# checkpoint_file = STORAGE_DIR / "weights" / "epoch=268-val_loss=0.00493.ckpt"
# V2
checkpoint_file = STORAGE_DIR / "weights" / "iclr" / "21-50-39/checkpoints/epoch=301-val_loss=0.16791.ckpt"

if not os.path.exists(checkpoint_file):
    raise ValueError(f"checkpoint not found: {checkpoint_file}")


bfm_config_path = "configs"
print(bfm_config_path)
with initialize(version_base=None, config_path=bfm_config_path, job_name="test_app"):
    cfg = compose(config_name="bfm_train_config.yaml")

base_model = setup_bfm_model(cfg, mode="train")

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
output_dir = f"outputs_bfm_spatial"

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

# === SPATIAL K-FOLD CONFIGURATION ===
file_list = train_dataset.files

block_size = 0.25  # degrees
n_splits = min(5, len(file_list))    # number of folds
fold_id = 0     # can be dynamically set
plot_output = f"fold_{fold_id}_map.png"

print("Preparing spatial K-fold split...")

# 1. Collect lat/lon from all files (assume train_dataset == full dataset)
#file_list = train_dataset.files
lat_list, lon_list = [], []

for f in file_list:
    data = torch.load(f, map_location="cpu", weights_only=True)
    lat = np.mean(data["metadata"]["lat"])
    lon = np.mean(data["metadata"]["lon"])
    lat_list.append(lat)
    lon_list.append(lon)

lat_array = np.array(lat_list)
lon_array = np.array(lon_list)

# 2. Assign each file to a spatial block
lat_blocks = np.floor(lat_array / block_size).astype(int)
lon_blocks = np.floor(lon_array / block_size).astype(int)
block_ids = np.array([f"{lat}_{lon}" for lat, lon in zip(lat_blocks, lon_blocks)])

# Optional: Debug block distribution
block_counts = Counter(block_ids)
print("Block distribution:", dict(block_counts))

# 3. Handle low block count gracefully
unique_blocks = np.unique(block_ids)
if len(unique_blocks) < 2:
    print(f"❌ Not enough spatial blocks ({len(unique_blocks)}) to perform K-Fold. Using all data for both train and val.")
    train_files = file_list
    val_files = file_list
else:
    if len(unique_blocks) < n_splits:
        print(f"⚠️ Only {len(unique_blocks)} unique spatial blocks — reducing n_splits to match.")
        n_splits = len(unique_blocks)
    print(f"Applying GroupKFold with n_splits={n_splits}, fold_id={fold_id}...")
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(file_list, groups=block_ids))
    train_idx, val_idx = splits[fold_id]
    train_files = [file_list[i] for i in train_idx]
    val_files = [file_list[i] for i in val_idx]

# 4. Inject these into your datasets
train_dataset.files = train_files
val_dataset.files = val_files

print(f"Fold {fold_id}: {len(train_files)} train files, {len(val_files)} val files")

# 5. Plot spatial split
plt.figure(figsize=(10, 6))
plt.scatter(lon_array[:len(train_files)], lat_array[:len(train_files)], label="Train", alpha=0.5)
plt.scatter(lon_array[-len(val_files):], lat_array[-len(val_files):], label="Val", alpha=0.5)
plt.legend()
plt.title(f"Spatial Fold {fold_id}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.savefig(plot_output, dpi=300)
print(f"Saved spatial fold map: {plot_output}")
#-------------------------------------------------------------------------------

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
    target = batch["species_distribution"] #target = sample["target"]
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
