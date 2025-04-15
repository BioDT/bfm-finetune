import os
import mlflow
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from aurora import Aurora, AuroraSmall
from torch.utils.data import DataLoader

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from bfm_finetune.aurora_mod import AuroraModified, AuroraExtend, AuroraFlex
from bfm_finetune.dataloaders.dataloader_utils import custom_collate_fn
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoLifeCLEFSpeciesDataset,
)
from bfm_finetune.dataloaders.toy_dataset.dataloader import ToyClimateDataset
from bfm_finetune.utils import save_checkpoint, load_checkpoint, seed_everything, get_supersampling_target_lat_lon, get_lat_lon_ranges
from bfm_finetune.metrics import compute_ssim_metric, compute_spc, compute_rmse
from bfm_finetune.plots import plot_eval

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
    loss = weight_3 * rmse + weight_1 * (1.0 - ssim) + weight_2 * (1.0 - (spc + 1.0) / 2.0)
    return loss


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    for sample in dataloader:
        batch = sample["batch"]#.to(device)
        batch["species_distribution"] = batch["species_distribution"].to(device)
        targets = sample["target"].to(device)
        # print(f"Target_shape: {targets.shape}")
        optimizer.zero_grad()
        outputs = model(batch)  # e.g., outputs shape: [B, 10000, H, W]
        loss = criterion(outputs, targets)
        # loss = compute_statio_temporal_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(dataloader)
    return epoch_loss

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    with torch.inference_mode():
        for sample in dataloader:
            batch = sample["batch"]#.to(device)
            batch["species_distribution"] = batch["species_distribution"].to(device)
            targets = sample["target"].to(device)
            outputs = model(batch)
            loss = criterion(outputs, targets)
            # loss = compute_statio_temporal_loss(outputs, targets)
            epoch_loss += loss.item()
    epoch_loss /= len(dataloader)
    return epoch_loss


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
        base_model = Aurora(use_lora=False) # stabilise_level_agg=True
        base_model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt") # strict=False
        atmos_levels = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    elif cfg.model.big_ft:
        base_model = Aurora(use_lora=False)
        base_model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
    
    base_model.to(device)

    num_species = cfg.dataset.num_species  # Our new finetuning dataset has 500 channels.
    # geo_size = (152, 320)  # WORKS
    # geo_size = (152, 200) # TODO: make this dynamic (200 comes from only positive lon)
    # # geo_size = (17, 32)  # WORKS
    # if cfg.model.supersampling:
    #     geo_size = (721, 1440) #WORKS
    supersampling_target_lat_lon = get_supersampling_target_lat_lon(cfg.model.supersampling)
    if supersampling_target_lat_lon:
        print("supersampling lat-lon", supersampling_target_lat_lon[0].shape, supersampling_target_lat_lon[1].shape)
    
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
        train_dataset = GeoLifeCLEFSpeciesDataset(num_species=num_species, mode="train", negative_lon_mode=cfg.dataset.negative_lon_mode)
        val_dataset = GeoLifeCLEFSpeciesDataset(num_species=num_species, mode="val", negative_lon_mode=cfg.dataset.negative_lon_mode)
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
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=cfg.dataset.num_workers,
    )
    #######   V1
    # model = AuroraModified(
    #     base_model=base_model,
    #     new_input_channels=num_species,
    #     use_new_head=True,
    #     target_size=geo_size,
    #     latent_dim=latent_dim,
    # )

    # # Freeze all pretrained parts. We already froze base_model inside AuroraModified.
    # # Ensure that in the input adapter, only the LoRA parameters are trainable.
    # for name, param in model.input_adapter.named_parameters():
    #     if "lora_A" not in name and "lora_B" not in name:
    #         param.requires_grad = False

    # # Optimizer on the LoRA adapter parameters and new head.
    # params_to_optimize = (
    #     list(model.input_adapter.lora_A.parameters())
    #     + list(model.input_adapter.lora_B.parameters())
    #     + list(model.new_head.parameters())
    # )

    ####### V2
    # model = AuroraExtend(base_model=base_model,
    #                      latent_dim=latent_dim,
    #                      in_channels=num_species,
    #                      hidden_channels=128,
    #                      out_channels=num_species,
    #                      target_size=geo_size)
    # params_to_optimize = model.parameters()

    ###### V3
    model = AuroraFlex(base_model=base_model, in_channels=num_species, hidden_channels=cfg.model.hidden_dim,
                        out_channels=num_species, lat_lon=lat_lon, supersampling_cfg=cfg.model.supersampling, atmos_levels=atmos_levels,)
    params_to_optimize = model.parameters()
    
    model.to(device)
    
    optimizer = optim.AdamW(params_to_optimize, lr=cfg.training.lr)
    criterion = nn.MSELoss()

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
            train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)

            if epoch % cfg.training.val_every ==0:
                val_loss = validate_epoch(model, val_dataloader, criterion, device)
                print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")
                mlflow.log_metric("val_loss", val_loss, step=epoch+1)


            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
            mlflow.log_metric("train_loss", train_loss, step=epoch+1)
            
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_save_path)
                mlflow.log_metric("best_loss", best_loss, step=epoch+1)

    # final evaluate
    for sample in val_dataloader:
        batch = sample["batch"]# .to(device)
        batch["species_distribution"] = batch["species_distribution"].to(device)
        target = sample["target"]
        with torch.inference_mode():
            prediction = model.forward(batch)
        unnormalized_preds = val_dataset.scale_species_distribution(prediction.clone(), unnormalize=True)
        plot_eval(
            batch=batch,
            # prediction_species=prediction,
            prediction_species=unnormalized_preds,
            out_dir=plots_dir,
            save=True
        )


if __name__ == "__main__":
    main()