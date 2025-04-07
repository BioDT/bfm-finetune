from datetime import datetime
from glob import glob
from pathlib import Path

import torch
from aurora.batch import Batch, Metadata
from torch.utils.data import Dataset

from bfm_finetune.dataloaders.geolifeclef_species import utils


class GeoLifeCLEFSpeciesDataset(Dataset):
    def __init__(
        self,
        data_dir: Path = utils.aurorashape_species_location,
        num_species: int = 500,
        mode: str = "train",
    ):
        self.data_dir = data_dir
        self.num_species = num_species
        if mode == "train":
            self.files = glob(str(data_dir / "train" / "*.pt"))
            self.files = sorted(self.files)
            print(f"files {len(self.files)}")
        else:
            self.files = glob(str(data_dir / "val" / "*.pt"))
            self.files = sorted(self.files)
            print(f"files {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        data = torch.load(fpath, map_location="cpu", weights_only=True)
        lat = data["metadata"]["lat"]
        lon = data["metadata"]["lon"]
        # print(lon)
        metadata = Metadata(
            lat=torch.Tensor(lat),
            lon=torch.Tensor(lon),
            time=tuple(datetime(el, 1, 1, 12, 0) for el in data["metadata"]["time"]),
            atmos_levels=(100, 250, 500, 850),
        )
        species_distribution = data["species_distribution"]
        # print(species_distribution.shape)
        # [T, S, H, W]
        H = species_distribution.shape[2]
        W = species_distribution.shape[3]
        assert (
            species_distribution.shape[1] == self.num_species
        ), f"species_distribution.shape[1]={species_distribution.shape[1]}, self.num_species={self.num_species}"
        surf_vars = {"species_distribution": species_distribution}
        static_vars = {k: torch.randn(H, W) for k in ("lsm", "z", "slt")}
        atmos_vars = {k: torch.randn(2, 4, H, W) for k in ("z", "u", "v", "t", "q")}
        batch = Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=metadata,
        )
        # target = torch.randn(self.num_species, H, W)
        target = species_distribution[1, :, :, :].unsqueeze(0) # Add the time dimension
        # print(target.shape)
        return {"batch": batch, "target": target}
