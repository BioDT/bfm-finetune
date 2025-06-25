import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from bfm_model.bfm.model import BFM
from hydra import compose, initialize
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from bfm_finetune import bfm_mod
from bfm_finetune.dataloaders.dataloader_utils import custom_collate_fn
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoLifeCLEFSpeciesDataset,
)
from bfm_finetune.finetune_new_variables import (
    save_checkpoint,
    train_epoch,
    validate_epoch,
)
from bfm_finetune.metrics import compute_geolifeclef_f1, compute_rmse
from bfm_finetune.paths import REPO_FOLDER, STORAGE_DIR
from bfm_finetune.utils import (
    get_lat_lon_ranges,
    load_checkpoint,
)

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

# TODO: Add your path here
checkpoint_file = (
    "/home/atrantas/bfm-finetune/bfm_finetune/outputs_bfm_finetune_48800/checkpoints/"
)

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
    # strict=False,  # False if loading from a pre-trained with PEFT checkpoint
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

base_model = BFM(**bfm_args)


finetune_config_path = "."  # f"bfm_finetune"
with initialize(
    version_base=None, config_path=finetune_config_path, job_name="test_app"
):
    finetune_cfg = compose(config_name="finetune_config.yaml")

num_species = finetune_cfg.dataset.num_species
model = bfm_mod.BFMRaw(base_model=base_model, n_species=num_species, mode="eval")

device = base_model.device
model.to(device)

val_dataset = GeoLifeCLEFSpeciesDataset(
    num_species=num_species,
    mode="val",
    negative_lon_mode=finetune_cfg.dataset.negative_lon_mode,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=finetune_cfg.dataset.num_workers,
)

params_to_optimize = model.parameters()

optimizer = torch.optim.AdamW(
    params_to_optimize,
    lr=finetune_cfg.training.lr,
    weight_decay=0.0001,
    betas=(0.9, 0.95),
    eps=1e-8,
)

criterion = nn.L1Loss()

_, best_loss = load_checkpoint(
    model, optimizer, finetune_cfg.training.checkpoint_path, strict=False
)

# final evaluate
all_backbones = []
sample_ids = []  # To keep track of which sample each backbone comes from
all_rmse = []
all_f1 = []

for i, sample in enumerate(val_dataloader):
    batch = sample["batch"]  # .to(device)
    batch["species_distribution"] = batch["species_distribution"].to(device)
    target = sample["target"]
    with torch.inference_mode():
        prediction, encoder, backbone = model.forward(batch)
    unnormalized_preds = val_dataset.scale_species_distribution(
        prediction.clone(), unnormalize=True
    )

    # Compute metrics
    # Move target to device for computation
    target_tensor = target.to(device)

    # Calculate RMSE
    rmse = compute_rmse(prediction, target_tensor).item()
    all_rmse.append(rmse)

    # Calculate custom F1 score
    f1 = compute_geolifeclef_f1(prediction.cpu(), target.cpu())
    all_f1.append(f1)

    print(f"Sample {i} - RMSE: {rmse:.4f}, Custom F1: {f1:.4f}")

    # Store the backbone features
    all_backbones.append(backbone.cpu().numpy())
    sample_ids.append(i)

    # Print backbone shape
    print(f"Backbone size: {backbone.shape}")

# Calculate and print average metrics over validation set
avg_rmse = np.mean(all_rmse)
avg_f1 = np.mean(all_f1)
print(f"\nValidation Set Metrics:")
print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Average GeoLifeCLEF F1: {avg_f1:.4f}")
print("DONE")

# Run PCA on all collected backbones
print(f"Collected backbones from {len(all_backbones)} samples")

# Reshape and concatenate all backbones
all_backbones_array = np.concatenate(
    [b.reshape(b.shape[0] * b.shape[1], b.shape[2]) for b in all_backbones], axis=0
)
print(f"Combined backbone shape: {all_backbones_array.shape}")

# Apply PCA
n_components = 6  # Number of principal components to compute
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(all_backbones_array)

# Print explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
print("Explained variance ratio by component:", explained_variance)
print("Cumulative explained variance:", cumulative_variance)

# Create a correlation matrix between the first 6 principal components
corr_matrix = np.corrcoef(principal_components, rowvar=False)

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
cmap = plt.cm.viridis
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = (
    True  # Optional: mask the upper triangle for cleaner viz
)

im = plt.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1)
plt.colorbar(im, label="Correlation")
# plt.title('Correlation Matrix of First 6 Principal Components')

# Add text annotations
for i in range(n_components):
    for j in range(n_components):
        text = plt.text(
            j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black"
        )

# Set tick labels
tick_labels = [f"PC{i+1}\n({explained_variance[i]:.2%})" for i in range(n_components)]
plt.xticks(range(n_components), tick_labels, rotation=45)
plt.yticks(range(n_components), tick_labels)

plt.tight_layout()
plt.savefig("pca_correlation_matrix.png", dpi=300)
plt.close()

# Save PCA results
np.savez(
    "backbone_pca_results.npz",
    principal_components=principal_components,
    explained_variance=explained_variance,
    pca_components=pca.components_,
    correlation_matrix=corr_matrix,
)

print("PCA analysis completed and saved")
