# Updated script for spatial finetuning of BFM using patch-level lat/lon
# NOTE: Assumes each yearly_species_*.pt file contains many samples with metadata["lat"] and metadata["lon"] arrays

import importlib
import math
import os
from pathlib import Path
import torch
import torch.nn as nn
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.model_selection import GroupKFold

from bfm_model.bfm.rollout_finetuning import BFM_Forecastinglighting as BFM_forecast
from bfm_finetune import bfm_mod
from bfm_finetune.dataloaders.dataloader_utils import custom_collate_fn
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import GeoLifeCLEFSpeciesDataset
from bfm_finetune.dataloaders.toy_dataset.dataloader import ToyClimateDataset
from bfm_finetune.finetune_new_variables import save_checkpoint, train_epoch, validate_epoch
from bfm_finetune.paths import REPO_FOLDER, STORAGE_DIR
from bfm_finetune.plots_v2 import plot_eval
from bfm_finetune.utils import get_lat_lon_ranges, load_checkpoint

# Load BFM config and checkpoint
checkpoint_file = STORAGE_DIR / "weights" / "epoch=268-val_loss=0.00493.ckpt"
bfm_config_path = "../bfm-model/bfm_model/bfm/configs"
with initialize(version_base=None, config_path=bfm_config_path, job_name="test_app"):
    cfg = compose(config_name="train_config.yaml")

# Swin backbone parameters
swin_params = {}
if cfg.model.backbone == "swin":
    selected = cfg.model_swin_backbone[cfg.model.swin_backbone_size]
    swin_params = {
        "swin_encoder_depths": tuple(selected.encoder_depths),
        "swin_encoder_num_heads": tuple(selected.encoder_num_heads),
        "swin_decoder_depths": tuple(selected.decoder_depths),
        "swin_decoder_num_heads": tuple(selected.decoder_num_heads),
        "swin_window_size": tuple(selected.window_size),
        "swin_mlp_ratio": selected.mlp_ratio,
        "swin_qkv_bias": selected.qkv_bias,
        "swin_drop_rate": selected.drop_rate,
        "swin_attn_drop_rate": selected.attn_drop_rate,
        "swin_drop_path_rate": selected.drop_path_rate,
        "swin_use_lora": selected.use_lora,
    }

# Instantiate base BFM model
bfm_args = dict(
    surface_vars=cfg.model.surface_vars,
    edaphic_vars=cfg.model.edaphic_vars,
    atmos_vars=cfg.model.atmos_vars,
    climate_vars=cfg.model.climate_vars,
    species_vars=cfg.model.species_vars,
    vegetation_vars=cfg.model.vegetation_vars,
    land_vars=cfg.model.land_vars,
    agriculture_vars=cfg.model.agriculture_vars,
    forest_vars=cfg.model.forest_vars,
    redlist_vars=cfg.model.redlist_vars,
    misc_vars=cfg.model.misc_vars,
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
    ground_truth_dataset=None,
    strict=False,
    peft_r=cfg.finetune.rank,
    lora_alpha=cfg.finetune.lora_alpha,
    d_initial=cfg.finetune.d_initial,
    peft_dropout=cfg.finetune.peft_dropout,
    peft_steps=cfg.finetune.rollout_steps,
    peft_mode=cfg.finetune.peft_mode,
    use_lora=cfg.finetune.use_lora,
    use_vera=cfg.finetune.use_vera,
    rollout_steps=cfg.finetune.rollout_steps,
    **swin_params,
)

base_model = BFM_forecast.load_from_checkpoint(checkpoint_path=checkpoint_file, **bfm_args)

# Load finetune config
with initialize(version_base=None, config_path=".", job_name="test_app"):
    finetune_cfg = compose(config_name="finetune_config.yaml")

num_species = finetune_cfg.dataset.num_species
model = bfm_mod.BFMRaw(base_model=base_model, n_species=num_species, mode="train")
device = base_model.device
model.to(device)
model.device = device

# Dataset
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

# Spatial K-Fold Split
file_list = train_dataset.files
block_size = 0.25
lat_all, lon_all, file_ids = [], [], []

for i, f in enumerate(file_list):
    data = torch.load(f, map_location="cpu")
    lat = np.ravel(data["metadata"]["lat"])
    lon = np.ravel(data["metadata"]["lon"])
    lat_all.extend(lat)
    lon_all.extend(lon)
    file_ids.extend([i] * len(lat))

lat_array = np.array(lat_all)
lon_array = np.array(lon_all)
file_ids = np.array(file_ids)
lat_blocks = np.floor(lat_array / block_size).astype(int)
lon_blocks = np.floor(lon_array / block_size).astype(int)
block_ids = np.array([f"{la}_{lo}" for la, lo in zip(lat_blocks, lon_blocks)])

# Assign dominant block to each file
file_to_blocks = defaultdict(list)
for file_id, block in zip(file_ids, block_ids):
    file_to_blocks[file_id].append(block)
file_block_ids = [Counter(blks).most_common(1)[0][0] for blks in file_to_blocks.values()]

# KFold assignment
fold_id = 0
n_splits = min(5, len(set(file_block_ids)))
if n_splits < 2:
    print("❌ Not enough unique spatial blocks to perform split.")
    train_files, val_files = file_list, file_list
else:
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(file_list, groups=file_block_ids))
    train_idx, val_idx = splits[fold_id]
    train_files = [file_list[i] for i in train_idx]
    val_files = [file_list[i] for i in val_idx]

train_dataset.files = train_files
val_dataset.files = val_files

# Dataloaders
train_dataloader = DataLoader(
    train_dataset, batch_size=finetune_cfg.training.batch_size,
    shuffle=True, collate_fn=custom_collate_fn,
    num_workers=finetune_cfg.dataset.num_workers,
)
val_dataloader = DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    collate_fn=custom_collate_fn, num_workers=finetune_cfg.dataset.num_workers,
)

# Optimizer & Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_cfg.training.lr, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=finetune_cfg.training.lr / 10)
criterion = nn.L1Loss()

# Train
output_dir = "outputs_bfm_spatial"
checkpoint_save_path = Path(output_dir) / "checkpoints"
predictions_dir = Path(output_dir) / "predictions"
plots_dir = Path(output_dir) / "plots"
os.makedirs(checkpoint_save_path, exist_ok=True)
_, best_loss = load_checkpoint(model, optimizer, finetune_cfg.training.checkpoint_path)

for epoch in range(1, finetune_cfg.training.epochs):
    train_loss = train_epoch(model, train_dataloader, optimizer, criterion, scheduler, device)
    if epoch % finetune_cfg.training.val_every == 0:
        val_loss = validate_epoch(model, val_dataloader, criterion, device)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_save_path)


for sample in val_dataloader:
    batch = sample["batch"]
    batch["species_distribution"] = batch["species_distribution"].to(device)
    target = sample["batch"]["species_distribution"].mean(dim=0)  # polygon-level aggregation

    with torch.inference_mode():
        prediction = model.forward(batch)

    unnormalized_preds = val_dataset.scale_species_distribution(prediction.clone(), unnormalize=True)

    # SAVE PATCH-LEVEL PREDICTIONS
    patch_preds = val_dataset.scale_species_distribution(prediction.clone(), unnormalize=True)
    torch.save(patch_preds, predictions_dir / "patch_level_predictions.pt")

    # SAVE POLYGON-LEVEL PREDICTIONS
    poly_preds = patch_preds.mean(dim=0, keepdim=True)
    torch.save(poly_preds, predictions_dir / "polygon_level_predictions.pt")

    # plot
    plot_eval(
        batch=batch,
        prediction_species=poly_preds,
        out_dir=plots_dir,
        save=True,
    )

# --- ADDITION 2: MAP VISUALIZATION OF SPATIAL COVERAGE ---

import geopandas as gpd
from shapely.geometry import Point

# Collect patch-level coordinates for spatial coverage plot
patch_coords = []
for f in train_dataset.files:
    data = torch.load(f, map_location="cpu")
    coords = zip(data["metadata"]["lon"], data["metadata"]["lat"])
    patch_coords.extend(list(coords))

# Save as GeoJSON
patch_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lon, lat in patch_coords])
patch_gdf.to_file(plots_dir / "train_coverage.geojson", driver="GeoJSON")

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(patch_gdf.geometry.x, patch_gdf.geometry.y, alpha=0.4, s=3)
plt.title("Spatial Coverage of Training Patches")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.savefig(plots_dir / "train_coverage_map.png")
plt.close()

print("✅ Saved spatial training coverage map and GeoJSON")
