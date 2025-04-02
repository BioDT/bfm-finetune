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
from bfm_finetune.utils import save_checkpoint, load_checkpoint, seed_everything

from bfm_finetune.plots import plot_eval

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    for sample in dataloader:
        batch = sample["batch"].to(device)
        targets = sample["target"].to(device)
        optimizer.zero_grad()
        outputs = model(batch)  # e.g., outputs shape: [B, 10000, H, W]
        loss = criterion(outputs, targets)
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
            batch = sample["batch"].to(device)
            targets = sample["target"].to(device)
            outputs = model(batch)
            loss = criterion(outputs, targets)
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

    num_species = cfg.dataset.num_species  # Our new finetuning dataset has 1000 channels.
    geo_size = (152, 320)  # WORKS
    # geo_size = (17, 32)  # WORKS
    batch_size = 2 if cfg.model.base_small else 1
    latent_dim = 12160
    num_epochs = cfg.training.epochs

    if cfg.dataset.toy:
        dataset = ToyClimateDataset(
            num_samples=100,
            new_input_channels=num_species,
            num_species=num_species,
            geo_size=geo_size,
        )
    else:
        train_dataset = GeoLifeCLEFSpeciesDataset(num_species=num_species, mode="train")
        val_dataset = GeoLifeCLEFSpeciesDataset(num_species=num_species, mode="val")
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
                        out_channels=num_species, atmos_levels=atmos_levels)
    params_to_optimize = model.parameters()
    
    model.to(device)
    
    optimizer = optim.AdamW(params_to_optimize, lr=cfg.training.lr)
    criterion = nn.MSELoss()

    checkpoint_save_path = Path(output_dir) / "checkpoints"

    # Load checkpoint if available
    start_epoch, best_loss = load_checkpoint(model, optimizer, cfg.training.checkpoint_path)
    val_loss = 1_000_000
    mlflow.set_experiment("BFM_Finetune")

    plots_dir = Path(output_dir) / "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    with mlflow.start_run():
        mlflow.log_param("num_epochs", cfg.training.epochs)
        mlflow.log_param("learning_rate", cfg.training.lr)
        mlflow.log_param("batch_size", cfg.training.batch_size)
        for epoch in range(start_epoch, num_epochs):
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
        batch = sample["batch"].to(device)
        target = sample["target"]
        with torch.inference_mode():
            prediction = model.forward(batch)
        plot_eval(
            batch=batch,
            prediction_species=prediction,
            out_dir=plots_dir,
        )


if __name__ == "__main__":
    main()