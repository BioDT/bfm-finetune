import os
from datetime import timedelta
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from aurora import Aurora, AuroraSmall
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from bfm_finetune.aurora_mod import AuroraFlex, AuroraRaw
from bfm_finetune.dataloaders.dataloader_utils import custom_collate_fn
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoLifeCLEFSpeciesDataset,
)
from bfm_finetune.dataloaders.toy_dataset.dataloader import ToyClimateDataset
from bfm_finetune.metrics import compute_geolifeclef_f1, compute_rmse
from bfm_finetune.utils import (
    get_lat_lon_ranges,
    get_supersampling_target_lat_lon,
    load_checkpoint,
)


@hydra.main(version_base=None, config_path="", config_name="finetune_config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = cfg.training.gpu

    # Set the internal precision for TensorCores (H100s)
    torch.set_float32_matmul_precision(cfg.training.precision_in)

    # Load base Aurora model
    if cfg.model.base_small:
        base_model = AuroraSmall()
        base_model.load_checkpoint(
            "microsoft/aurora", "aurora-0.25-small-pretrained.ckpt"
        )
        atmos_levels = (100, 250, 500, 850)
    elif cfg.model.big:
        base_model = Aurora(
            use_lora=True,
            lora_steps=1,
            autocast=True,
        )
        base_model.load_checkpoint(
            "microsoft/aurora", "aurora-0.25-pretrained.ckpt", strict=False
        )
        atmos_levels = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    elif cfg.model.big_ft:
        base_model = Aurora(use_lora=False)
        base_model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")

    base_model.to(device)

    num_species = cfg.dataset.num_species

    # Get lat-lon ranges for supersampling
    supersampling_target_lat_lon = get_supersampling_target_lat_lon(
        cfg.model.supersampling
    )
    if supersampling_target_lat_lon:
        if len(supersampling_target_lat_lon) != 2:
            raise ValueError(
                "Invalid supersampling_target_lat_lon: Expected a tuple of (lat, lon)."
            )
        if (
            supersampling_target_lat_lon[0].shape[0] <= 0
            or supersampling_target_lat_lon[1].shape[0] <= 0
        ):
            raise ValueError(
                "Invalid supersampling_target_lat_lon: Latitude or longitude arrays are empty."
            )
        print(
            "supersampling lat-lon",
            supersampling_target_lat_lon[0].shape,
            supersampling_target_lat_lon[1].shape,
        )

    # Create datasets
    if cfg.dataset.toy:
        lat_lon = get_lat_lon_ranges()
        val_dataset = ToyClimateDataset(
            num_samples=1,
            num_species=num_species,
            lat_lon=lat_lon,
        )
    else:
        val_dataset = GeoLifeCLEFSpeciesDataset(
            num_species=num_species,
            mode="val",
            negative_lon_mode=cfg.dataset.negative_lon_mode,
        )
        lat_lon = val_dataset.get_lat_lon()

    print("lat-lon", lat_lon[0].shape, lat_lon[1].shape)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=cfg.dataset.num_workers,
    )

    # Create model using the same architecture as in finetune_new_variables.py
    model = AuroraRaw(base_model=base_model, n_species=num_species)
    model.to(device)

    # Setup optimizer (needed for checkpoint loading)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Load checkpoint
    _, best_loss = load_checkpoint(
        model, optimizer, cfg.training.checkpoint_path, strict=False
    )

    print(f"Loaded checkpoint with best loss: {best_loss}")
    if best_loss == float('inf'):
        print("WARNING: Best loss is infinity - this may indicate an issue with checkpoint saving/loading")
    
    # Set model to evaluation mode
    model.eval()

    # Evaluation loop
    all_backbones = []
    sample_ids = []
    all_rmse = []
    all_f1 = []

    print("Starting evaluation...")
    for i, sample in enumerate(val_dataloader):
        batch = sample["batch"]
        batch["species_distribution"] = batch["species_distribution"].to(device)
        target = sample["target"]

        with torch.inference_mode():
            # Use the model's forward method to get prediction
            prediction = model.forward(batch)

            # To get backbone features, we need to do a separate forward pass
            # through the model components manually
            species_data = batch["species_distribution"]
            if len(species_data.shape) == 4:  # (T, C, H, W)
                species_data = species_data.unsqueeze(0)  # (1, T, C, H, W)

            # Get encoder output (tokens)
            tokens = model.encoder(species_data)  # (B, num_tokens, embed_dim)

            # Use the tokens as backbone features since we can't easily access
            # the Aurora backbone intermediate features
            backbone_tensor = tokens

        # Unnormalize predictions if needed
        if hasattr(val_dataset, "scale_species_distribution"):
            # prediction should have shape (N, H, W)
            # Add batch and time dimensions for unnormalization: (1, 1, N, H, W)
            pred_for_unnorm = prediction.unsqueeze(0).unsqueeze(0).clone()
            unnormalized_preds = val_dataset.scale_species_distribution(
                pred_for_unnorm, unnormalize=True
            )
            unnormalized_preds = unnormalized_preds.squeeze(0).squeeze(0)
        else:
            unnormalized_preds = prediction

        # Compute metrics
        # Target shape: (1, N, H, W) -> (N, H, W)
        target_tensor = target.squeeze(0).to(device)

        # Ensure prediction and target have the same shape for metrics
        if prediction.dim() == 3:  # (N, H, W)
            prediction_tensor = prediction.unsqueeze(
                0
            )  # Add batch dimension: (1, N, H, W)
        else:
            prediction_tensor = prediction

        if target_tensor.dim() == 3:  # (N, H, W)
            target_for_metrics = target_tensor.unsqueeze(
                0
            )  # Add batch dimension: (1, N, H, W)
        else:
            target_for_metrics = target_tensor

        # Calculate RMSE
        rmse = compute_rmse(prediction_tensor, target_for_metrics).item()
        all_rmse.append(rmse)

        # Calculate custom F1 score
        f1 = compute_geolifeclef_f1(prediction_tensor.cpu(), target_for_metrics.cpu())
        all_f1.append(f1)

        print(f"Sample {i} - RMSE: {rmse:.4f}, Custom F1: {f1:.4f}")

        # Store the backbone features
        all_backbones.append(backbone_tensor.cpu().numpy())
        sample_ids.append(i)

        # Print backbone shape
        print(f"Backbone size: {backbone_tensor.shape}")

    # Calculate and print average metrics over validation set
    avg_rmse = np.mean(all_rmse)
    avg_f1 = np.mean(all_f1)
    print(f"\nValidation Set Metrics:")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average GeoLifeCLEF F1: {avg_f1:.4f}")

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

    # Add text annotations
    for i in range(n_components):
        for j in range(n_components):
            text = plt.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    # Set tick labels
    tick_labels = [
        f"PC{i+1}\n({explained_variance[i]:.2%})" for i in range(n_components)
    ]
    plt.xticks(range(n_components), tick_labels, rotation=45)
    plt.yticks(range(n_components), tick_labels)

    plt.tight_layout()
    plt.savefig("aurora_pca_correlation_matrix.png", dpi=300)
    plt.close()

    # Save PCA results
    np.savez(
        "aurora_backbone_pca_results.npz",
        principal_components=principal_components,
        explained_variance=explained_variance,
        pca_components=pca.components_,
        correlation_matrix=corr_matrix,
    )

    # Save results to text file
    with open("aurora_evaluation_results.txt", "w") as f:
        f.write("Aurora Model Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Average RMSE: {avg_rmse:.4f}\n")
        f.write(f"Average GeoLifeCLEF F1: {avg_f1:.4f}\n")
        f.write(f"Number of samples evaluated: {len(all_rmse)}\n")
        f.write(f"PCA explained variance: {explained_variance}\n")
        f.write(f"PCA cumulative variance: {cumulative_variance}\n")

    print("Aurora evaluation completed and results saved")
    print("DONE")


if __name__ == "__main__":
    main()
