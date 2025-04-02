from datetime import datetime
from typing import Tuple

import torch
from aurora.batch import Batch, Metadata
from torch.utils.data import Dataset


class ToyClimateDataset(Dataset):
    def __init__(
        self,
        geo_size: Tuple[int, int],
        num_samples=10,
        new_input_channels=10,
        num_species=10000,
    ):
        self.num_samples = num_samples
        self.new_input_channels = new_input_channels
        self.num_species = num_species

        # Define latitude and longitude grids.
        self.lat = torch.linspace(
            start=90, end=-90, steps=geo_size[0]
        )
        self.lon = torch.linspace(0, 360, geo_size[1] + 1)[:-1]
        self.metadata = Metadata(
            lat=self.lat,
            lon=self.lon,
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(100, 250, 500, 850),
        )
        self.H, self.W = len(self.lat), len(self.lon)
        print(f"Lat {self.H} Long {self.W}")
        # Set history length T to 2 (as required by the Aurora encoder).
        self.T = 2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        new_input = torch.randn(self.T, self.new_input_channels, self.H, self.W)
        surf_vars = {"species_distribution": new_input}
        static_vars = {k: torch.randn(self.H, self.W) for k in ("lsm", "z", "slt")}
        atmos_vars = {
            k: torch.randn(2, 4, self.H, self.W) for k in ("z", "u", "v", "t", "q")
        }
        batch = Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=self.metadata,
        )
        # TODO Validate: Need to add at first dim T=1 for V2 and V3 model implementations
        target = torch.randn(self.num_species, self.H, self.W)
        return {"batch": batch, "target": target}
