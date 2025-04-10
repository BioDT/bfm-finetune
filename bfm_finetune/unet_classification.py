import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from aurora import AuroraSmall
from aurora.batch import Batch, Metadata

# Import the model wrapper from our codebase that uses Aurora as a backbone.
from bfm_finetune.aurora_mod import AuroraExtend

# Import dataloader from GeoLifeCLEFSpecies
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoLifeCLEFSpeciesDataset,
)

# Import plotting utility
from bfm_finetune.plots import plot_eval
from bfm_finetune.utils import save_checkpoint  # added import
from bfm_finetune.dataloaders.dataloader_utils import custom_collate_fn  # added import

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_species = 500
target_size = (152, 320)
latent_dim = 12160  # example latent dimension
epochs = 5
lr = 1e-4
checkpoint_save_path = "./checkpoints"  # added checkpoint path
os.makedirs(checkpoint_save_path, exist_ok=True)


def main():
    # Load Pretrained Aurora Backbone
    backbone = AuroraSmall(use_lora=False, autocast=True)
    backbone.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
    backbone.to(device)

    # Build the model using Aurora as backbone via AuroraExtend
    model = AuroraExtend(
        base_model=backbone,
        latent_dim=latent_dim,
        in_channels=num_species,  # new modality: species_distribution with 500 channels
        hidden_channels=160,
        out_channels=num_species,  # predict same 500-channel output (next time step)
        target_size=target_size,
    )
    model.to(device)

    # Setup dataloaders from GeoLifeCLEFSpeciesDataset
    train_dataset = GeoLifeCLEFSpeciesDataset(num_species=num_species, mode="train")
    val_dataset = GeoLifeCLEFSpeciesDataset(num_species=num_species, mode="val")

    # updated DataLoader instantiation to use the custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn,
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def train_epoch():
        model.train()
        total_loss = 0.0
        for sample in train_loader:
            batch = sample["batch"].to(device)
            target = sample["target"].to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate_epoch():
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for sample in val_loader:
                batch = sample["batch"].to(device)
                target = sample["target"].to(device)
                output = model(batch)
                loss = criterion(output, target)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    best_loss = float("inf")  # added best loss initialization
    # Training Loop
    for epoch in range(epochs):
        train_loss = train_epoch()
        val_loss = validate_epoch()
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, best_loss, checkpoint_save_path
            )  # save checkpoint

    # Prediction: run model on one validation sample and plot prediction versus target.
    sample = next(iter(val_loader))
    batch = sample["batch"].to(device)
    with torch.no_grad():
        prediction = model(batch)
    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_eval(batch=batch, prediction_species=prediction, out_dir=plots_dir, save=True)
    print("Training and prediction completed.")


if __name__ == "__main__":
    main()
