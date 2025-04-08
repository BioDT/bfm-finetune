from datetime import datetime
from glob import glob
from pathlib import Path
import json

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
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"
        if mode == "train":
            self.files = glob(str(train_dir / "*.pt"))
            self.files = sorted(self.files)
            print(f"files {len(self.files)}")
        else:
            self.files = glob(str(val_dir / "*.pt"))
            self.files = sorted(self.files)
            print(f"files {len(self.files)}")
        stats_file = data_dir / "stats.json"
        with open(stats_file) as f:
            self.stats = json.load(f)

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
        species_distribution = self.scale_species_distribution(species_distribution)
        assert (
            species_distribution.shape[1] == self.num_species
        ), f"species_distribution.shape[1]={species_distribution.shape[1]}, self.num_species={self.num_species}"
        surf_vars = {"species_distribution": species_distribution}
        static_vars = {k: torch.randn(H, W) for k in ("lsm", "z", "slt")} # NOT USED
        atmos_vars = {k: torch.randn(2, 4, H, W) for k in ("z", "u", "v", "t", "q")} # NOT USED
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
    
    def scale_batch(self, batch: Batch, unnormalize: bool = False):
        # TODO: check this function
        species_distribution = batch.surf_vars["species_distribution"]
        species_distribution = self.scale_species_distribution(species_distribution, unnormalize=unnormalize)
        batch.surf_vars["species_distribution"] = species_distribution

    def scale_species_distribution(self, species_distribution: torch.Tensor, unnormalize: bool = False) -> torch.Tensor:
        # TODO: check this function
        print(species_distribution.shape) # [B, T, S, H, W]
        for species_i in range(species_distribution.shape[1]):
            stats_species = self.stats[species_i]
            row = species_distribution[:, species_i, :, :, :]
            if unnormalize:
                row = row * stats_species["std"] + stats_species["mean"]
            else:
                row = (row - stats_species["mean"]) / stats_species["std"]
            species_distribution[:, species_i, :, :, :] = row
        return species_distribution