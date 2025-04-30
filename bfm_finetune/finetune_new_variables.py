import os
import math
import numpy as np

import mlflow
from pathlib import Path

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from aurora import Aurora, AuroraSmall
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from bfm_finetune.aurora_mod import AuroraFlex, AuroraRaw
from bfm_finetune.dataloaders.dataloader_utils import custom_collate_fn
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoLifeCLEFSpeciesDataset,
)
from bfm_finetune.dataloaders.toy_dataset.dataloader import ToyClimateDataset
from bfm_finetune.utils import (
    save_checkpoint,
    load_checkpoint,
    seed_everything,
    get_supersampling_target_lat_lon,
    get_lat_lon_ranges,
)
from bfm_finetune.metrics import compute_ssim_metric, compute_spc, compute_rmse
from bfm_finetune.plots_v2 import plot_eval


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_statio_temporal_loss(outputs, targets):
    criterion = nn.MSELoss()
    weight_1 = 10.0
    weight_2 = 5.0
    weight_3 = 0.5
    # print(f"prediction shape : {outputs.shape} | target shape : {targets.shape} ")
    ssim = compute_ssim_metric(outputs, targets)
    spc = compute_spc(outputs, targets)
    rmse_t = criterion(outputs, targets)
    rmse = compute_rmse(outputs, targets)
    # print(f"SSIM: {ssim} | SPC: {spc} | RMSE: {rmse}")
    loss = (
        weight_3 * rmse + weight_1 * (1.0 - ssim) + weight_2 * (1.0 - (spc + 1.0) / 2.0)
    )
    return loss


def train_epoch(model, dataloader, optimizer, criterion, scheduler, device, clip_value=1.0):
    model.train()
    epoch_loss = 0.0
    for sample in dataloader:
        batch = sample["batch"]
        batch["species_distribution"] = batch["species_distribution"].to(device)
        targets = sample["target"].to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()
        # Clip gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        scheduler.step() # Optional
        epoch_loss += loss.item()
    epoch_loss /= len(dataloader)
    return epoch_loss


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    with torch.inference_mode():
        for sample in dataloader:
            batch = sample["batch"]  # .to(device)
            batch["species_distribution"] = batch["species_distribution"].to(device)
            targets = sample["target"].to(device)
            outputs = model(batch)
            loss = criterion(outputs, targets)
            # loss = compute_statio_temporal_loss(outputs, targets)
            epoch_loss += loss.item()
    epoch_loss /= len(dataloader)

    # Extract and optionally visualize latent features
    if hasattr(model, "encode_latent"):
        all_latents = []
        all_species_ids = []

        for sample in dataloader:
            batch = sample["batch"]
            batch["species_distribution"] = batch["species_distribution"].to(device)
            targets = sample["target"].to(device)

            # Extract latent representation
            latent = model.encode_latent(batch)  # assumes method exists
            if hasattr(latent, "surf_vars"):
                # flatten surf_vars into a single feature vector
                tensors = [v.reshape(v.size(0), -1) for v in latent.surf_vars.values()]
                latent_tensor = torch.cat(tensors, dim=1)
            else:
                latent_tensor = latent.reshape(latent.size(0), -1)
            latent_flat = latent_tensor.cpu().detach().numpy()
            all_latents.append(latent_flat)

            if "species_id" in batch:
                all_species_ids.extend(batch["species_id"].cpu().tolist())

        all_latents = np.vstack(all_latents)
        # dynamic PCA components based on data
        n_samples, n_feats = all_latents.shape
        n_comp = min(2, n_samples, n_feats)
        if n_comp < 1:
            print("Skipping PCA: no data for projection, got", all_latents.shape)
        else:
            pca = PCA(n_components=n_comp)
            proj = pca.fit_transform(all_latents)
            plt.figure(figsize=(10, 7))
            if n_comp == 2:
                sns.scatterplot(
                    x=proj[:, 0],
                    y=proj[:, 1],
                    hue=all_species_ids if all_species_ids else None,
                    palette="tab20",
                    legend=False,
                )
                plt.xlabel("PC1")
                plt.ylabel("PC2")
            else:
                sns.histplot(
                    proj[:, 0],
                    hue=all_species_ids if all_species_ids else None,
                    palette="tab20",
                    legend=False,
                )
                plt.xlabel("PC1")
            plt.title("Latent Feature Space by Species (PCA-reduced)")
            plt.tight_layout()
            plt.savefig("latent_feature_space.png")
            plt.close()

    return epoch_loss

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(version_base=None, config_path="", config_name="finetune_config")
def main(cfg):

    print(OmegaConf.to_yaml(cfg))
    # Set the internal precision for TensorCores (H100s)
    torch.set_float32_matmul_precision(cfg.training.precision_in)

    # Seed the experiment for numpy, torch and python.random.
    # TODO: sometimes linked to loss nan???
    # seed_everything(0)

    output_dir = HydraConfig.get().runtime.output_dir

    print(output_dir)

    if cfg.model.base_small:
        base_model = AuroraSmall()
        base_model.load_checkpoint(
            "microsoft/aurora", "aurora-0.25-small-pretrained.ckpt"
        ) 
        atmos_levels = (100, 250, 500, 850)
    elif cfg.model.big:
        base_model = Aurora(use_lora=True)  # stabilise_level_agg=True, TODO: set strict=False 
        base_model.load_checkpoint(
            "microsoft/aurora", "aurora-0.25-pretrained.ckpt", strict=False
        ) 
        atmos_levels = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    elif cfg.model.big_ft:
        base_model = Aurora(use_lora=False)
        base_model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")

    base_model.to(device)

    num_species = (
        cfg.dataset.num_species
    )  # Our new finetuning dataset has 500 channels.
    # geo_size = (152, 320)  # WORKS
    # geo_size = (152, 200) # TODO: make this dynamic (200 comes from only positive lon)
    # # geo_size = (17, 32)  # WORKS
    # if cfg.model.supersampling:
    #     geo_size = (721, 1440) #WORKS
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

    latent_dim = 12160
    num_epochs = cfg.training.epochs

    if cfg.dataset.toy:
        # customizable lat_lon
        lat_lon = get_lat_lon_ranges()
        train_dataset = ToyClimateDataset(
            num_samples=100,
            new_input_channels=num_species,
            num_species=num_species,
            lat_lon=lat_lon,
        )
        val_dataset = ToyClimateDataset(
            num_samples=20,
            new_input_channels=num_species,
            num_species=num_species,
            lat_lon=lat_lon,
        )
    else:
        train_dataset = GeoLifeCLEFSpeciesDataset(
            num_species=num_species,
            mode="train",
            negative_lon_mode=cfg.dataset.negative_lon_mode,
        )
        val_dataset = GeoLifeCLEFSpeciesDataset(
            num_species=num_species,
            mode="val",
            negative_lon_mode=cfg.dataset.negative_lon_mode,
        )
        # get lat_lon from dataset
        lat_lon = train_dataset.get_lat_lon()
    print("lat-lon", lat_lon[0].shape, lat_lon[1].shape)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=cfg.dataset.num_workers,
    )
    # TODO Make it distinct
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=cfg.dataset.num_workers,
    )
    ###### V3
    # model = AuroraFlex(
    #     base_model=base_model,
    #     in_channels=num_species,
    #     hidden_channels=cfg.model.hidden_dim,
    #     out_channels=num_species,
    #     lat_lon=lat_lon,
    #     supersampling_cfg=cfg.model.supersampling,
    #     atmos_levels=atmos_levels,
    # )
    ### V4 
    model = AuroraRaw(base_model=base_model)
    
    params_to_optimize = model.parameters()

    model.to(device)

    optimizer = torch.optim.AdamW(params_to_optimize, lr=cfg.training.lr, 
                                  weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8,)
    criterion = nn.MSELoss()

    total_steps = num_epochs
    warmup_steps = int(0.05 * total_steps)   # 5 % warm-up
    min_lr_ratio = 0.05                      # final LR = 5 % of base

    def lr_lambda(step):
        if step < warmup_steps:                       # linear warm-up
            return step / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda)


    checkpoint_save_path = Path(output_dir) / "checkpoints"

    # Load checkpoint if available
    _, best_loss = load_checkpoint(model, optimizer, cfg.training.checkpoint_path)
    val_loss = 1_000_000
    mlflow.set_experiment("BFM_Finetune")

    plots_dir = Path(output_dir) / "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    with mlflow.start_run():
        mlflow.log_param("num_epochs", cfg.training.epochs)
        mlflow.log_param("learning_rate", cfg.training.lr)
        mlflow.log_param("batch_size", cfg.training.batch_size)
        for epoch in range(1, num_epochs):
            train_loss = train_epoch(
                model, train_dataloader, optimizer, criterion, scheduler, device
            )

            if epoch % cfg.training.val_every == 0:
                val_loss = validate_epoch(model, val_dataloader, criterion, device)
                print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")
                mlflow.log_metric("val_loss", val_loss, step=epoch + 1)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
            mlflow.log_metric("train_loss", train_loss, step=epoch + 1)

            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, best_loss, checkpoint_save_path
                )
                mlflow.log_metric("best_loss", best_loss, step=epoch + 1)

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
        plot_eval(
            batch=batch,
            # prediction_species=prediction,
            prediction_species=unnormalized_preds,
            out_dir=plots_dir,
            save=True,
        )


if __name__ == "__main__":
    main()