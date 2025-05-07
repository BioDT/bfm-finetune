import json
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
from aurora.batch import Batch, Metadata
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from bfm_finetune.dataloaders.dataloader_utils import (
    manage_negative_lon,
    manage_negative_lon_dict,
)
from bfm_finetune.dataloaders.geolifeclef_species import utils
from bfm_finetune.prithvi.utils import prithvi_species_patches_location


class GeoLifeCLEFSpeciesDataset(Dataset):
    def __init__(
        self,
        data_dir: Path = utils.aurorashape_species_location,
        num_species: int = 500,
        mode: str = "train",
        unnormalize: bool = False,
        negative_lon_mode: Literal["roll", "exclude", "translate", "ignore"] = "ignore",
    ):
        self.data_dir = data_dir
        self.num_species = num_species
        self.unnormalize = unnormalize
        self.negative_lon_mode = negative_lon_mode
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
        species_distribution = data["species_distribution"]
        H = species_distribution.shape[2]
        W = species_distribution.shape[3]
        species_distribution = self.scale_species_distribution(
            species_distribution, unnormalize=self.unnormalize
        )
        assert (
            species_distribution.shape[1] >= self.num_species
        ), f"species_distribution.shape[1]={species_distribution.shape[1]}, self.num_species={self.num_species}"
        batch = {
            "species_distribution": species_distribution[:, : self.num_species, :, :],
            "metadata": {
                "lat": torch.Tensor(lat),
                "lon": torch.Tensor(lon),
                "time": tuple(
                    datetime(el, 1, 1, 12, 0) for el in data["metadata"]["time"]
                ),
            },
        }
        batch = manage_negative_lon_dict(batch, mode=self.negative_lon_mode)
        target = batch["species_distribution"][1, :, :, :].unsqueeze(
            0
        )  # Add time dimension

        # Compute a single coordinate pair per sample (e.g., mean of grid)
        coords = torch.tensor(
            [float(np.mean(lat)), float(np.mean(lon))],
            dtype=torch.float32,
        )

        return {"batch": batch, "target": target, "coords": coords}

    def scale_batch(self, batch: Batch, unnormalize: bool = False):
        species_distribution = batch.surf_vars["species_distribution"]
        species_distribution = self.scale_species_distribution(
            species_distribution, unnormalize=unnormalize
        )
        batch.surf_vars["species_distribution"] = species_distribution

    def scale_species_distribution(
        self, species_distribution: torch.Tensor, unnormalize: bool = False
    ) -> torch.Tensor:
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
        item = self[0]
        metadata = item["batch"]["metadata"]
        return metadata["lat"].numpy(), metadata["lon"].numpy()


class GeoLifeCLEFSpeciesDatasetPrithvi(Dataset):
    def __init__(
        self,
        data_dir: str,
        num_species: int = 500,
        unnormalize: bool = False,
        geo_size=(64, 128),
    ):
        self.data_dir = data_dir
        self.num_species = num_species
        self.unnormalize = unnormalize
        self.geo_size = geo_size
        stats_file = str(utils.aurorashape_species_location / "stats.json")
        with open(stats_file) as f:
            self.stats = json.load(f)
        self.files = glob(str(Path(self.data_dir) / "*.pt"))
        self.files = sorted(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        data = torch.load(fpath, map_location="cpu", weights_only=False)

        lat = data["metadata"]["lat_patch"]
        lon = data["metadata"]["lon_patch"]
        nbatch = len(data)
        # print("Len of batch", nbatch)
        # self.lat = torch.linspace(start=90, end=-90, steps=self.geo_size[0])
        # self.lon = torch.linspace(0, 360, self.geo_size[1] + 1)[:-1]
        # print(lat, lon)
        self.lat = torch.tensor(lat)
        self.lon = torch.tensor(lon)
        # print(f"len lat {len(self.lat)}, lon {len(self.lon)}")

        # Calculate static surface variables (latitude and longitude in radians)
        latitudes = self.lat / 360 * 2.0 * torch.pi
        longitudes = self.lon / 360 * 2.0 * torch.pi

        # Create a meshgrid of latitudes and longitudes
        latitudes, longitudes = torch.meshgrid(
            torch.as_tensor(latitudes), torch.as_tensor(longitudes), indexing="ij"
        )
        # Stack sine and cosine of latitude and longitude to create static surface tensor
        self.sur_static = torch.stack(
            [torch.sin(latitudes), torch.cos(longitudes), torch.sin(longitudes)], axis=0
        )

        species_distribution = data["patch"]  # [T=2, C=500, H=64, W=128]
        # print(species_distribution.shape)
        # print(f"Timestap of patch {species_distribution.shape[0]}")
        # [T, S, H, W]
        # H = species_distribution.shape[2]
        # W = species_distribution.shape[3]
        species_distribution = self.scale_species_distribution(
            species_distribution, unnormalize=self.unnormalize
        )
        # only select these species
        species_distribution = species_distribution[:, : self.num_species, :, :]
        assert (
            species_distribution.shape[1] == self.num_species
        ), f"species_distribution.shape[1]={species_distribution.shape[1]}, self.num_species={self.num_species}"

        # target = torch.randn(self.num_species, H, W)
        x = species_distribution[0, :, :, :]  # [500, 152, 320] T=0 => [500, 64, 128]
        target = species_distribution[
            1, :, :, :
        ]  # [500, 152, 320] T=1  -> We crop to geo_size = (64,128) => Patches that Prithvi requires

        # lead_time = torch.empty((nbatch,), dtype=torch.float32)
        # input_time = torch.empty((nbatch,), dtype=torch.float32)
        lead_time = torch.tensor(6, dtype=torch.float32)
        input_time = torch.tensor(-6, dtype=torch.float32)
        # print("Target shape dataset", target.shape)

        return {
            "x": x.unsqueeze(0),
            "y": x,
            "target": target,
            "lead_time": lead_time,  # torch.zeros(1),
            "input_time": input_time,
            "static": self.sur_static,
            "lat_original": data["metadata"]["lat_array"],
            "lon_original": data["metadata"]["lon_array"],
        }

    def scale_batch(self, batch: dict, unnormalize: bool = False):
        # TODO: check this function
        species_distribution = batch.surf_vars["species_distribution"]
        species_distribution = self.scale_species_distribution(
            species_distribution, unnormalize=unnormalize
        )
        batch.surf_vars["species_distribution"] = species_distribution

    def scale_species_distribution(
        self, species_distribution: torch.Tensor, unnormalize: bool = False
    ) -> torch.Tensor:
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


class GeoCLEFDataModulePrithvi:
    """
    This module handles data loading, batching, and train/validation splits.

    Attributes:
        train_data_path: Path to training data.
        valid_data_path: Path to validation data.
        file_glob_pattern: Pattern to match NetCDF files.
        batch_size: Size of each mini-batch.
        num_workers: Number of subprocesses for data loading.
    """

    def __init__(
        self,
        train_data_path: Path = prithvi_species_patches_location / "train",
        valid_data_path: Path = prithvi_species_patches_location / "val",
        batch_size: int = 16,
        num_data_workers: int = 8,
    ):
        """Initializes the ERA5DataModule with the specified settings.

        Args:
            train_data_path: Directory containing training data.
            valid_data_path: Directory containing validation data.
            file_glob_pattern: Glob pattern to match NetCDF files.
            batch_size: Size of mini-batches. Defaults to 16.
            num_data_workers: Number of workers for data loading.
        """
        super().__init__()
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path

        self.batch_size: int = batch_size
        self.num_workers: int = num_data_workers

        stats_file = utils.aurorashape_species_location / "stats.json"
        with open(stats_file) as f:
            self.stats = json.load(f)

    def setup(self, stage: str | None = None):
        """Sets up the datasets for different stages
        (train, validation, predict).

        Args:
            stage: Stage for which the setup is performed ("fit", "predict").
        """
        if stage == "fit":
            self.dataset_train = GeoLifeCLEFSpeciesDatasetPrithvi(
                data_dir=str(self.train_data_path)
            )
            self.dataset_val = GeoLifeCLEFSpeciesDatasetPrithvi(
                data_dir=str(self.valid_data_path)
            )
        elif stage == "predict":
            self.dataset_predict = GeoLifeCLEFSpeciesDatasetPrithvi(
                data_dir=str(self.valid_data_path)
            )
        else:
            raise ValueError(f"stage {stage} not implemented")

    def train_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the training data."""
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            sampler=DistributedSampler(dataset=self.dataset_train, shuffle=True),
        )

    def val_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the validation data."""

        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            sampler=DistributedSampler(dataset=self.dataset_val, shuffle=False),
        )

    def predict_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the prediction data."""
        return DataLoader(
            dataset=self.dataset_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def test_normalization_and_denormalization():
    """
    Creates a dummy species distribution tensor and a dummy stats structure.
    Then tests normalization and denormalization.
    """
    B, T, S, H, W = 2, 2, 3, 10, 10  # small spatial size for testing.
    raw_data = torch.rand(B, T, S, H, W) * 100.0

    dummy_stats = {
        0: {
            "species_i": 0,
            "min": 0.0,
            "max": 100.0,
            "mean": 50.0,
            "std": 10.0,
            "count": 1000,
        },
        1: {
            "species_i": 1,
            "min": 0.0,
            "max": 100.0,
            "mean": 40.0,
            "std": 8.0,
            "count": 800,
        },
        2: {
            "species_i": 2,
            "min": 0.0,
            "max": 100.0,
            "mean": 60.0,
            "std": 12.0,
            "count": 1200,
        },
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
    pixel0 = raw_data[0, 0, 0, 0, 0].item()
    normalized0 = normalized[0, 0, 0, 0, 0].item()
    expected0 = (pixel0 - dummy_stats[0]["mean"]) / dummy_stats[0]["std"]
    print(
        f"Species 0, pixel[0]: raw={pixel0:.3f}, normalized={normalized0:.3f}, expected={expected0:.3f}"
    )

    denormalized = dataset.scale_species_distribution(
        normalized.clone(), unnormalize=True
    )
    pixel0_denorm = denormalized[0, 0, 0, 0, 0].item()
    print(
        f"Species 0, pixel[0] after denorm: {pixel0_denorm:.3f} (should match raw={pixel0:.3f})"
    )

    assert np.allclose(pixel0, pixel0_denorm, atol=1e-4), "Denormalization failed!"
    print("Normalization and denormalization test passed.")


# Uncomment to check
# if __name__=="__main__":
#     # test_normalization_and_denormalization()
#     train_dataset = GeoLifeCLEFSpeciesDataset(num_species=500, mode="train")
#     sample = train_dataset[1]
#     print(sample["batch"])
