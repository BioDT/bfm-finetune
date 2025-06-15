import torch
import xarray as xr
from torch.utils.data import Dataset

class LatentCHELSADataset(Dataset):
    def __init__(self, netcdf_path, latent_type="backbone_output", input_dim="decoder_input"):
        """
        Args:
            netcdf_path: Path to the NetCDF file containing the latent representations
            latent_type: Which latent representation to use - 'encoder_output' or 'backbone_output'
        """
        self.ds = xr.open_dataset(netcdf_path)
        #self.latent_type = latent_type
        
        #if latent_type not in self.ds:
        #    raise ValueError(f"Latent type '{latent_type}' not found in dataset. Available: {list(self.ds.data_vars)}")
        
        #self.latents = self.ds[latent_type]
        self.targets = self.ds["target"]
        self.inputs = self.ds[input_dim]  # <-- store the tensor as self.inputs
        self.input_dim = self.ds[input_dim].shape[-1]  # <-- store the dimension as int

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        x = self.inputs[idx].values  # <-- use self.inputs
        y = self.targets[idx].values
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
