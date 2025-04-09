from datetime import datetime
from glob import glob
from pathlib import Path
import json
from typing import Tuple
import numpy as np

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
        unnormalize: bool = False,
    ):
        self.data_dir = data_dir
        self.num_species = num_species
        self.unnormalize = unnormalize
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
        species_distribution = self.scale_species_distribution(species_distribution, unnormalize=self.unnormalize)
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
        target = species_distribution[1, :, :, :].unsqueeze(0) # Add time dimension
        # print("Target shape dataset", target.shape)
        return {"batch": batch, "target": target}
    
    def scale_batch(self, batch: Batch, unnormalize: bool = False):
        # TODO: check this function
        species_distribution = batch.surf_vars["species_distribution"]
        species_distribution = self.scale_species_distribution(species_distribution, unnormalize=unnormalize)
        batch.surf_vars["species_distribution"] = species_distribution

    def scale_species_distribution(self, species_distribution: torch.Tensor, unnormalize: bool = False) -> torch.Tensor:
        # TODO: check this function
        # print("Species distribution shape in scale", species_distribution.shape) # [T, S, H, W]
        for species_i in range(species_distribution.shape[1]):
            stats_species = self.stats[species_i]
            row = species_distribution[:, species_i, :, :]
            if unnormalize:
                row = row * stats_species["std"] + stats_species["mean"]
            else:
                row = (row - stats_species["mean"]) / stats_species["std"]
            species_distribution[:, species_i, :, :] = row
        return species_distribution
    
    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        # returns the lat_lon for the first file
        item = self[0]
        metadata = item["batch"].metadata
        return metadata.lat.numpy(), metadata.lon.numpy()



def test_normalization_and_denormalization():
    """
    Creates a dummy species distribution tensor and a dummy stats structure.
    Then tests normalization and denormalization.
    """
    B, T, S, H, W = 2, 2, 3, 10, 10  # small spatial size for testing.
    raw_data = torch.rand(B, T, S, H, W) * 100.0
    
    dummy_stats = {
        0: {"species_i": 0, "min": 0.0, "max": 100.0, "mean": 50.0, "std": 10.0, "count": 1000},
        1: {"species_i": 1, "min": 0.0, "max": 100.0, "mean": 40.0, "std": 8.0, "count": 800},
        2: {"species_i": 2, "min": 0.0, "max": 100.0, "mean": 60.0, "std": 12.0, "count": 1200},
    }
    
    # Create a dummy dataset object and override self.stats
    class DummyDataset:
        def __init__(self):
            self.stats = dummy_stats
        def scale_species_distribution(self, species_distribution, unnormalize=False):
            B, T, S, H, W = species_distribution.shape
            for species_i in range(S):
                stats_species = self.stats[species_i]
                row = species_distribution[:, :, species_i, :, :]
                if unnormalize:
                    row = row * stats_species["std"] + stats_species["mean"]
                else:
                    row = (row - stats_species["mean"]) / (stats_species["std"] + 1e-8)
                species_distribution[:, :, species_i, :, :] = row
            return species_distribution
        
    dataset = DummyDataset()
    
    normalized = dataset.scale_species_distribution(raw_data.clone(), unnormalize=False)
    # For species 0: mean=50, std=10, so if raw[0,0,0,0,0]=e.g. 80, normalized=3.0
    pixel0 = raw_data[0,0,0,0,0].item()
    normalized0 = normalized[0,0,0,0,0].item()
    expected0 = (pixel0 - dummy_stats[0]["mean"]) / dummy_stats[0]["std"]
    print(f"Species 0, pixel[0]: raw={pixel0:.3f}, normalized={normalized0:.3f}, expected={expected0:.3f}")
    
    denormalized = dataset.scale_species_distribution(normalized.clone(), unnormalize=True)
    pixel0_denorm = denormalized[0,0,0,0,0].item()
    print(f"Species 0, pixel[0] after denorm: {pixel0_denorm:.3f} (should match raw={pixel0:.3f})")
    
    assert np.allclose(pixel0, pixel0_denorm, atol=1e-4), "Denormalization failed!"
    print("Normalization and denormalization test passed.")

# Uncomment to check
# if __name__=="__main__":
#     # test_normalization_and_denormalization()
#     train_dataset = GeoLifeCLEFSpeciesDataset(num_species=500, mode="train")
#     sample = train_dataset[1]
#     print(sample["batch"])