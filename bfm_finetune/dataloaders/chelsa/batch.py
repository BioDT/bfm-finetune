import os
import numpy as np
import xarray as xr
import rioxarray as rio
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from bfm_model.bfm.dataloader_monthly import batch_to_device

def coarse_grid(data, grid_lat, grid_lon):
    return (
        data
        .groupby_bins('y', grid_lat, labels=grid_lat[:-1])
        .mean()
        .groupby_bins('x', grid_lon, labels=grid_lon[:-1])
        .mean()
    )

def load_chelsa_targets(tas_path, pr_path, grid_lat, grid_lon):
    tas = rio.open_rasterio(tas_path).squeeze("band", drop=True)
    pr = rio.open_rasterio(pr_path).squeeze("band", drop=True)
    tas = (tas * 0.1) - 273.15
    if not tas.rio.equals(pr):
        raise ValueError(f"Mismatch in geotransform: {tas_path} vs {pr_path}")
    tas_coarse = coarse_grid(tas, grid_lat, grid_lon)
    pr_coarse = coarse_grid(pr, grid_lat, grid_lon)
    combined = xr.concat([tas_coarse, pr_coarse], dim="variable")
    combined = combined.assign_coords(variable=["tas", "pr"])
    return combined.transpose("y_bins", "x_bins", "variable").values

@hydra.main(config_path=".", config_name="batch_config.yaml")
def main(cfg: DictConfig):
    from batch_model_loader import get_bfm_model_and_dataloader
    model, dataloader, time_index = get_bfm_model_and_dataloader(cfg)
    model.eval().to(cfg.device)

    lat_bins = np.round(np.arange(cfg.grid.lat_start, cfg.grid.lat_end + 1e-6, cfg.grid.resolution), 3)
    lon_bins = np.round(np.arange(cfg.grid.lon_start, cfg.grid.lon_end + 1e-6, cfg.grid.resolution), 3)

    all_latents, all_backbone_outputs, all_targets, all_times = [], [], [], []

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        x, _ = batch
        x = batch_to_device(x, cfg.device)
        with torch.no_grad():
            # Extract encoded, backbone_output and model output as in bfm_get_latent.py
            encoded, backbone_output, _ = model(x)
        encoded = encoded.squeeze(0).cpu().numpy()
        backbone_output = backbone_output.squeeze(0).cpu().numpy()

        dt = pd.to_datetime(time_index[batch_idx])
        tas_path = os.path.join(cfg.data.tas_dir, f"CHELSA_tas_{dt.month:02d}_{dt.year}_V.2.1.tif")
        pr_path = os.path.join(cfg.data.pr_dir, f"CHELSA_pr_{dt.month:02d}_{dt.year}_V.2.1.tif")

        chelsa = load_chelsa_targets(tas_path, pr_path, lat_bins, lon_bins)
        chelsa_flat = chelsa.reshape(-1, 2)

        all_latents.append(encoded)
        all_backbone_outputs.append(backbone_output)
        all_targets.append(chelsa_flat)
        all_times.append(dt)

    latent_arr = np.stack(all_latents, axis=0)
    backbone_output_arr = np.stack(all_backbone_outputs, axis=0)
    target_arr = np.stack(all_targets, axis=0)
    time_coords = np.array(all_times)

    ds = xr.Dataset(
        {
            "encoder_output": (["time", "pixel", "embedding_dim"], latent_arr),
            "backbone_output": (["time", "pixel", "embedding_dim"], backbone_output_arr),
            "target": (["time", "pixel", "variable"], target_arr),
        },
        coords={
            "time": time_coords,
            "variable": ["tas", "pr"],
            "embedding_dim": np.arange(latent_arr.shape[-1]),
        },
    )
    Path(cfg.output.file).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(cfg.output.file)
    print(f"Saved to: {cfg.output.file}")

if __name__ == "__main__":
    main()#