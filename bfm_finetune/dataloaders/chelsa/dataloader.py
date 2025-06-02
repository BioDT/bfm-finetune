import torch
import xarray as xr
from torch.utils.data import Dataset

class LatentCHELSADataset(Dataset):
    def __init__(self, netcdf_path, latent_type="backbone_output"):
        """
        Args:
            netcdf_path: Path to the NetCDF file containing the latent representations
            latent_type: Which latent representation to use - 'encoder_output' or 'backbone_output'
        """
        self.ds = xr.open_dataset(netcdf_path)
        self.latent_type = latent_type
        
        if latent_type not in self.ds:
            raise ValueError(f"Latent type '{latent_type}' not found in dataset. Available: {list(self.ds.data_vars)}")
        
        self.latents = self.ds[latent_type]
        self.targets = self.ds["target"]

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        x = self.latents[idx].values
        y = self.targets[idx].values
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
