from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from aurora.batch import Batch, Metadata
from torch.utils.data import Dataset


class ToyClimateDataset(Dataset):
    def __init__(
        self,
        lat_lon: Tuple[np.ndarray, np.ndarray],
        num_samples=10,
        num_species=500,
    ):
        self.num_samples = num_samples
        self.num_species = num_species

        # Define latitude and longitude grids.
        # self.lat = torch.linspace(
        #     start=90, end=-90, steps=geo_size[0]
        # )
        # self.lon = torch.linspace(0, 360, geo_size[1] + 1)[:-1]
        self.lat = torch.Tensor(lat_lon[0])
        self.lon = torch.Tensor(lat_lon[1])
        self.H, self.W = len(self.lat), len(self.lon)
        print(f"Lat {self.H} Long {self.W}")
        # Set history length T to 2 (as required by the Aurora encoder).
        self.T = 2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        species_distribution = torch.randn(self.T, self.num_species, self.H, self.W)
        year = 2017 + idx
        batch = {
            "species_distribution": species_distribution,
            "metadata": {
                "lat": torch.Tensor(self.lat),
                "lon": torch.Tensor(self.lon),
                "time": tuple(datetime(el, 1, 1, 12, 0) for el in [year, year + 1]),
            },
        }
        target = torch.randn(1, self.num_species, self.H, self.W)
        return {"batch": batch, "target": target}

    def scale_species_distribution(
        self, species_distribution: torch.Tensor, unnormalize: bool = False
    ) -> torch.Tensor:
        # does not do anything, only for interchangeability with the real dataloaders
        return species_distribution
