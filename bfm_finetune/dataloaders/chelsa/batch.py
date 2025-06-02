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
    #tas = (tas * 0.1) - 273.15
    
    # Use a safer method to check if geospatial attributes match
    try:
        if hasattr(tas, 'rio') and hasattr(pr, 'rio'):
            if hasattr(tas.rio, 'equals') and callable(tas.rio.equals):
                if not tas.rio.equals(pr):
                    print(f"Warning: Mismatch in geotransform between {tas_path} and {pr_path}")
            else:
                # Alternative check: compare basic geospatial attributes
                tas_attrs = {k: getattr(tas, k) for k in ['transform', 'crs', 'res'] if hasattr(tas, k)}
                pr_attrs = {k: getattr(pr, k) for k in ['transform', 'crs', 'res'] if hasattr(pr, k)}
                if tas_attrs != pr_attrs:
                    print(f"Warning: Mismatch in geospatial attributes between {tas_path} and {pr_path}")
    except Exception as e:
        print(f"Warning: Could not compare geospatial properties: {e}")
    
    # Continue with processing regardless
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
    
    # Enable gradient checkpointing to save memory if available
    if hasattr(model.backbone, 'gradient_checkpointing_enable'):
        model.backbone.gradient_checkpointing_enable()
    elif hasattr(model.backbone, 'use_checkpoint') and hasattr(model.backbone.use_checkpoint, '__call__'):
        model.backbone.use_checkpoint = True
    else:
        print("Warning: Gradient checkpointing not available for this model")
    
    # Use torch.amp.autocast to use mixed precision (fixed deprecated version)
    from torch.amp import autocast
    
    lat_bins = np.round(np.arange(cfg.grid.lat_start, cfg.grid.lat_end + 1e-6, cfg.grid.resolution), 3)
    lon_bins = np.round(np.arange(cfg.grid.lon_start, cfg.grid.lon_end + 1e-6, cfg.grid.resolution), 3)

    all_latents, all_backbone_outputs, all_targets, all_times = [], [], [], []

    # Free up memory before starting processing
    torch.cuda.empty_cache()
    
    # Filter time_index to only include dates up to end of 2018 (you can adjust this date)
    end_date = pd.to_datetime("2018-12-31")  # Or whatever your latest data date is
    valid_time_index = []
    valid_indices = []
    
    print("Checking available dates in CHELSA dataset...")
    for i, date_str in enumerate(time_index):
        try:
            dt = pd.to_datetime(date_str)
            if dt <= end_date:
                # Quick check if files exist
                year, month = dt.year, dt.month
                tas_path = os.path.join(cfg.data.tas_dir, f"CHELSA_tas_{month:02d}_{year}_V.2.1.tif")
                pr_path = os.path.join(cfg.data.pr_dir, f"CHELSA_pr_{month:02d}_{year}_V.2.1.tif")
                
                if os.path.exists(tas_path) and os.path.exists(pr_path):
                    valid_time_index.append(date_str)
                    valid_indices.append(i)
                    print(f"Found valid data for {date_str}")
                else:
                    print(f"Files not found for {date_str}")
            else:
                print(f"Skipping {date_str} as it's after cutoff date {end_date}")
        except Exception as e:
            print(f"Error parsing date {date_str}: {e}")
    
    if not valid_time_index:
        print("No valid dates found in your date range.")
        print(f"Available CHELSA data directories:")
        if os.path.exists(cfg.data.tas_dir):
            print(f"Temperature directory contents: {os.listdir(cfg.data.tas_dir)[:10]}...")
        if os.path.exists(cfg.data.pr_dir):
            print(f"Precipitation directory contents: {os.listdir(cfg.data.pr_dir)[:10]}...")
        raise ValueError("No valid data was found. Check your file paths and available dates.")
    
    print(f"Processing {len(valid_time_index)} valid dates...")

    for batch_idx, idx in tqdm(enumerate(valid_indices), total=len(valid_indices), desc="Processing datasets"):
        batch = next(iter(dataloader))  # Just get the first batch since we're loading files manually
        
        # Process in smaller chunks if needed
        try:
            x, _ = batch
            x = batch_to_device(x, cfg.device)
            
            # Use mixed precision to reduce memory usage
            with torch.no_grad(), autocast('cuda'):  # Fixed deprecated syntax
                # Extract encoded, backbone_output and model output
                encoded, backbone_output, _ = model(x)
                
            # Move results to CPU immediately to free GPU memory
            encoded = encoded.squeeze(0).cpu().numpy()
            backbone_output = backbone_output.squeeze(0).cpu().numpy()
            
            # Clear GPU cache after each batch
            torch.cuda.empty_cache()

            date_str = valid_time_index[batch_idx]
            try:
                dt = pd.to_datetime(date_str)
                year, month = dt.year, dt.month
                tas_path = os.path.join(cfg.data.tas_dir, f"CHELSA_tas_{month:02d}_{year}_V.2.1.tif")
                pr_path = os.path.join(cfg.data.pr_dir, f"CHELSA_pr_{month:02d}_{year}_V.2.1.tif")
                
                chelsa = load_chelsa_targets(tas_path, pr_path, lat_bins, lon_bins)
                chelsa_flat = chelsa.reshape(-1, 2)

                all_latents.append(encoded)
                all_backbone_outputs.append(backbone_output)
                all_targets.append(chelsa_flat)
                all_times.append(dt)
                
                print(f"Successfully processed {date_str}")
            except Exception as e:
                print(f"Error processing date {date_str}: {e}")
                continue
                
        except torch.cuda.OutOfMemoryError:
            # If we still get OOM, try to recover
            torch.cuda.empty_cache()
            print(f"CUDA out of memory on batch {batch_idx}. Skipping and continuing...")
            continue

    if not all_latents:
        raise ValueError("No valid data was processed. Check your file paths and date formats.")
    
    print(f"Successfully processed {len(all_latents)} dates. Creating output dataset...")
    
    # Debug shape information
    latent_shape = all_latents[0].shape
    backbone_shape = all_backbone_outputs[0].shape
    target_shape = all_targets[0].shape
    print(f"Model latent shape: {latent_shape} (total pixels: {latent_shape[0]})")
    print(f"Model backbone shape: {backbone_shape} (total pixels: {backbone_shape[0]})")
    print(f"Target shape: {target_shape} (total pixels: {target_shape[0]})")

    # Stack the arrays
    latent_arr = np.stack(all_latents, axis=0)
    backbone_output_arr = np.stack(all_backbone_outputs, axis=0)
    target_arr = np.stack(all_targets, axis=0)
    time_coords = np.array(all_times)

    # Convert float16 to float32 BEFORE creating datasets
    if latent_arr.dtype == np.float16:
        print("Converting float8 to float32 for NetCDF compatibility")
        latent_arr = latent_arr.astype(np.float32)
    if backbone_output_arr.dtype == np.float16:
        backbone_output_arr = backbone_output_arr.astype(np.float32)

    # Now create the datasets with the float32 data
    ds_model = xr.Dataset(
        {
            "encoder_output": (["time", "pixel", "embedding_dim"], latent_arr),
            "backbone_output": (["time", "pixel", "embedding_dim"], backbone_output_arr),
        },
        coords={
            "time": time_coords,
            "embedding_dim": np.arange(latent_arr.shape[-1]),
        },
    )

    ds_target = xr.Dataset(
        {
            "target": (["time", "target_pixel", "variable"], target_arr),
        },
        coords={
            "time": time_coords,
            "variable": ["tas", "pr"],
        },
    )

    # Combine them into a single dataset but with different pixel dimensions
    ds = xr.merge([ds_model, ds_target])

    # Get the output path from the correct location in the config
    if hasattr(cfg.data, 'output') and hasattr(cfg.data.output, 'file'):
        output_path = Path(cfg.data.output.file)
    else:
        # Use a default path if not specified in config
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"./chelsa_bfm_latents_{current_time}.nc")
        print(f"Output path not specified in config. Using default: {output_path}")

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the file
    ds.to_netcdf(output_path)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()