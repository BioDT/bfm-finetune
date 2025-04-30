accinfoimport os
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


# Add a helper function to move dictionaries to device
def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    else:
        return data


# Add a function to convert dictionary to Batch object
def dict_to_batch(batch_dict):
    """Convert a dictionary to a Batch object with the expected structure."""
    if isinstance(batch_dict, Batch):
        return batch_dict

    # Create surf_vars with species_distribution
    surf_vars = {}
    species_distribution = None

    # Try to extract species_distribution from different possible locations
    if "species_distribution" in batch_dict:
        species_distribution = batch_dict["species_distribution"]
    elif (
        "surf_vars" in batch_dict and "species_distribution" in batch_dict["surf_vars"]
    ):
        species_distribution = batch_dict["surf_vars"]["species_distribution"]

    # Ensure species_distribution has the correct shape: [B, T, C, H, W]
    if species_distribution is not None:
        # Print shape for debugging
        print(f"Original species_distribution shape: {species_distribution.shape}")

        # Reshape if needed - ensure it has 5 dimensions: [B, T, C, H, W]
        if len(species_distribution.shape) == 3:  # [C, H, W]
            species_distribution = species_distribution.unsqueeze(0).unsqueeze(
                0
            )  # [1, 1, C, H, W]
        elif len(species_distribution.shape) == 4:
            if species_distribution.shape[1] == num_species:  # [B, C, H, W]
                species_distribution = species_distribution.unsqueeze(
                    1
                )  # [B, 1, C, H, W]
            else:  # [B, T, H, W]
                species_distribution = species_distribution.unsqueeze(
                    2
                )  # [B, T, 1, H, W]

        # Ensure C dimension is num_species
        B, T, C, H, W = species_distribution.shape
        if C != num_species:
            print(f"Warning: Expected {num_species} species channels, got {C}")
            # If C is 3, it might be RGB channels instead of species
            if C == 3:
                # Create a zero tensor with correct shape and put the original data in first 3 channels
                new_tensor = torch.zeros(
                    (B, T, num_species, H, W), device=species_distribution.device
                )
                new_tensor[:, :, :3, :, :] = species_distribution
                species_distribution = new_tensor

        print(f"Reshaped species_distribution shape: {species_distribution.shape}")
        surf_vars["species_distribution"] = species_distribution

    # Get or create metadata
    metadata = batch_dict.get("metadata", None)

    # Create Batch object
    return Batch(
        surf_vars=surf_vars,
        metadata=metadata,
        static_vars=batch_dict.get("static_vars", {}),
        atmos_vars=batch_dict.get("atmos_vars", {}),
    )


__all__ = ["to_device", "dict_to_batch"]


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

    # Debug model input/output shapes
    def debug_model_shapes():
        # Get a sample batch
        sample = next(iter(val_loader))
        batch_dict = to_device(sample["batch"], device)

        # Print shapes before conversion to Batch
        print("Batch dict keys:", batch_dict.keys())
        if "species_distribution" in batch_dict:
            print(
                "Species distribution shape:", batch_dict["species_distribution"].shape
            )
        elif (
            "surf_vars" in batch_dict
            and "species_distribution" in batch_dict["surf_vars"]
        ):
            print(
                "Species distribution shape:",
                batch_dict["surf_vars"]["species_distribution"].shape,
            )

        # Convert to Batch object
        batch = dict_to_batch(batch_dict)

        # Print shapes after conversion
        print("Batch surf_vars keys:", batch.surf_vars.keys())
        print(
            "Batch species_distribution shape:",
            batch.surf_vars["species_distribution"].shape,
        )

        # Check model's expected input shape
        print(f"Model expected input: {num_species} channels")

        # Try one forward pass with small subset
        with torch.no_grad():
            try:
                output = model(batch)
                print("Model output shape:", output.shape)
                return True
            except Exception as e:
                print("Forward pass failed:", str(e))
                return False

    # Run debug function before training
    print("=== Running shape debugging ===")
    if not debug_model_shapes():
        print("Shape mismatch detected. Please check the model and data dimensions.")
        return

    def train_epoch():
        model.train()
        total_loss = 0.0
        for sample in train_loader:
            batch_dict = to_device(sample["batch"], device)
            batch = dict_to_batch(batch_dict)
            target = to_device(sample["target"], device)
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
                batch_dict = to_device(sample["batch"], device)
                batch = dict_to_batch(batch_dict)
                target = to_device(sample["target"], device)
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
    batch_dict = to_device(sample["batch"], device)
    batch = dict_to_batch(batch_dict)
    # Fix static_vars shapes
    B, T = next(iter(batch.surf_vars.values())).shape[:2]
    for k, v in batch.static_vars.items():
        batch.static_vars[k] = (
            batch.static_vars[k]
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(B, T, 1, *batch.static_vars[k].shape)
        )
    with torch.no_grad():
        prediction = model(batch)
    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_eval(batch=batch, prediction_species=prediction, out_dir=plots_dir, save=True)
    print("Training and prediction completed.")


if __name__ == "__main__":
    main()
