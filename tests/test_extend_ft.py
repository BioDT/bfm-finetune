from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from aurora import Aurora, AuroraSmall
from aurora.batch import Batch, Metadata
from torch.utils.data import DataLoader, Dataset, default_collate

from bfm_finetune.aurora_mod import AuroraRaw
from bfm_finetune.dataloaders.dataloader_utils import custom_collate_fn
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoLifeCLEFSpeciesDataset,
)
from bfm_finetune.dataloaders.toy_dataset.dataloader import ToyClimateDataset
from bfm_finetune.utils import get_lat_lon_ranges
import pytest

@pytest.mark.skip(reason="CUDA OOM")
def test_aurora_raw():
    # Dummy input for the new modality:
    # Batch size B=2, T=2 timesteps, species_channels=500, spatial dims 152x320.
    B, T = 2, 2
    species_channels = 50
    H_in, W_in = 152, 320
    # new_modality_input = torch.randn(B, T, species_channels, H_in, W_in)
    lat_lon = get_lat_lon_ranges()
    toy_dataset = ToyClimateDataset(
        lat_lon=lat_lon,
        num_species=species_channels,
        num_samples=3,
    )
    dataloader = DataLoader(
        toy_dataset,
        batch_size=B,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=15,
    )
    data = [el for el in dataloader]
    new_modality_input = data[0]["batch"]

    # Instantiate a dummy frozen Aurora backbone.
    # base_model = AuroraSmall()
    # base_model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
    base_model = Aurora(
        use_lora=False
    )  # stabilise_level_agg=True, TODO: set strict=False
    base_model.load_checkpoint(
        "microsoft/aurora", "aurora-0.25-pretrained.ckpt", strict=False
    )

    # latent dim := H x W // patch_size => 48640 // 4 = 12160
    # Instantiate our full module that extends Aurora.
    latent_dim = 12160  # TODO make it configurable
    model = AuroraRaw(
        base_model=base_model,
        n_species=species_channels,
        # hidden_channels=256,
        # out_channels=species_channels,
        # lat_lon=lat_lon,
    )
    # Move model to device (CPU or GPU).
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # new_modality_input = new_modality_input.to(device)
    new_modality_input["species_distribution"] = new_modality_input["species_distribution"].to(device)

    # Forward pass.
    output: torch.Tensor = model(new_modality_input)
    print("Output shape:", output.shape)
    # Expected output shape: [B, 500, 152, 320]
    assert output.dim() == 5
    assert output.shape[0] == B
    assert output.shape[1] == 1
    assert output.shape[2] == species_channels
    assert output.shape[3] == H_in
    assert output.shape[4] == W_in


if __name__ == "__main__":
    test_aurora_raw()
