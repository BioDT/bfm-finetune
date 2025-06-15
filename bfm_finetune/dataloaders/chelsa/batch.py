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
import re

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

def extract_date(filename, var):
    # e.g. CHELSA_tas_08_2012_V.2.1.tif -> (2012, 8)
    m = re.match(rf"CHELSA_{var}_(\d{{2}})_(\d{{4}})_V\.2\.1\.tif", filename)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        return year, month
    return None

@hydra.main(config_path=".", config_name="batch_config.yaml")
def main(cfg: DictConfig):
    from batch_model_loader import get_bfm_model_and_dataloader
    model, dataloader, time_index, predictions = get_bfm_model_and_dataloader(cfg)
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

    all_latents, all_backbone_outputs, all_decoded, all_targets, all_times = [], [], [], [], []

    # Free up memory before starting processing
    torch.cuda.empty_cache()
    
    # Filter time_index to only include dates up to end of 2018 (you can adjust this date)
    end_date = pd.to_datetime("2018-12-31")  # Or whatever your latest data date is
    valid_time_index = []
    valid_indices = []
    
    print("Checking available dates in CHELSA dataset...")
    tas_files = os.listdir(cfg.data.tas_dir)
    pr_files = os.listdir(cfg.data.pr_dir)

    tas_dates = set(extract_date(f, "tas") for f in tas_files if extract_date(f, "tas"))
    pr_dates = set(extract_date(f, "pr") for f in pr_files if extract_date(f, "pr"))

    valid_dates = sorted(tas_dates & pr_dates)
    print(f"Found {len(valid_dates)} valid date pairs.")

    if not valid_dates:
        raise ValueError("No valid date pairs found. Check file names and directories.")

    # TODO: Add these to the config and read them here
    start_year = getattr(cfg.timeperiod, "start_year", 2000)
    end_year = getattr(cfg.timeperiod, "end_year", 2018)
    start_month = getattr(cfg.timeperiod, "start_month", 1)
    end_month = getattr(cfg.timeperiod, "end_month", 12)

    # Filter valid_dates by the specified range
    valid_dates = [
        (year, month)
        for (year, month) in valid_dates
        if (start_year < year or (start_year == year and month >= start_month))
        and (year < end_year or (year == end_year and month <= end_month))
    ]

    print(f"Processing {len(valid_dates)} valid dates in range {start_year}-{start_month:02d} to {end_year}-{end_month:02d}...")

    for batch_idx, (year, month) in tqdm(enumerate(valid_dates), total=len(valid_dates), desc="Processing datasets"):
        batch = next(iter(dataloader))  # Just get the first batch since we're loading files manually
        
        # Process in smaller chunks if needed
        try:
            x, _ = batch
            x = batch_to_device(x, cfg.device)
            
            # Use mixed precision to reduce memory usage
            with torch.no_grad(), autocast('cuda'):
                # Extract encoded, backbone_output and model output
                encoded, backbone_output, _ = model(x)
                
            # Move results to CPU immediately to free GPU memory and convert BFloat16 to float32
            encoded = encoded.squeeze(0).cpu()
            backbone_output = backbone_output.squeeze(0).cpu()
            
            # Convert BFloat16 tensors to float32
            if encoded.dtype == torch.bfloat16:
                encoded = encoded.to(torch.float32)
            if backbone_output.dtype == torch.bfloat16:
                backbone_output = backbone_output.to(torch.float32)
                
            # Now convert to numpy
            encoded = encoded.numpy()
            backbone_output = backbone_output.numpy()
            
            # Clear GPU cache after each batch
            torch.cuda.empty_cache()

            tas_path = os.path.join(cfg.data.tas_dir, f"CHELSA_tas_{month:02d}_{year}_V.2.1.tif")
            pr_path = os.path.join(cfg.data.pr_dir, f"CHELSA_pr_{month:02d}_{year}_V.2.1.tif")
            
            chelsa = load_chelsa_targets(tas_path, pr_path, lat_bins, lon_bins)
            chelsa_flat = chelsa.reshape(-1, 2)
            
            # Check if batch_idx is in range of predictions list before accessing it
            pred_data = None
            if batch_idx < len(predictions) and predictions[batch_idx]:
                pred_data = predictions[batch_idx][0]
                print(f"Keys in predictions[batch_idx][0]: {list(pred_data.keys())}")
                if 'pred' in pred_data:
                    print(f"Keys in predictions[batch_idx][0]['pred']: {list(pred_data['pred'].keys())}")
            else:
                print(f"Warning: No prediction data available for batch_idx={batch_idx} ({year}-{month:02d})")
            
            try:
                # Only try to extract prediction data if pred_data exists
                if pred_data and 'pred' in pred_data and 'climate_variables' in pred_data['pred']:
                    decoded_t2m = pred_data['pred']['climate_variables']['t2m']
                    decoded_pr = pred_data['pred']['climate_variables']['tp']
                    
                    # Handle BFloat16 if present
                    if isinstance(decoded_t2m, torch.Tensor) and decoded_t2m.dtype == torch.bfloat16:
                        decoded_t2m = decoded_t2m.to(torch.float32).cpu().numpy()
                    if isinstance(decoded_pr, torch.Tensor) and decoded_pr.dtype == torch.bfloat16:
                        decoded_pr = decoded_pr.to(torch.float32).cpu().numpy()
                        
                    # Also check if they're already torch tensors of other types
                    if isinstance(decoded_t2m, torch.Tensor):
                        decoded_t2m = decoded_t2m.cpu().numpy()
                    if isinstance(decoded_pr, torch.Tensor):
                        decoded_pr = decoded_pr.cpu().numpy()
                        
                    decoded_flat = np.stack([decoded_t2m, decoded_pr], axis=-1).reshape(-1, 2)
                else:
                    # Fallback to zeros if no prediction data
                    print(f"No prediction data for {year}-{month:02d}, using zeros")
                    #decoded_flat = np.zeros((chelsa_flat.shape[0], 2))
            except Exception as inner_e:
                print(f"Error processing prediction data: {inner_e}")
                # Fallback to using zeros
                decoded_flat = np.zeros((chelsa_flat.shape[0], 2))

            # Handle encoded data similarly
            if pred_data and 'encoded' in pred_data:
                latent_data = pred_data['encoded']
                if isinstance(latent_data, torch.Tensor):
                    if latent_data.dtype == torch.bfloat16:
                        latent_data = latent_data.to(torch.float32) 
                    latent_data = latent_data.cpu().numpy()
                all_latents.append(latent_data)
            else:
                all_latents.append(encoded)
            
            all_backbone_outputs.append(backbone_output)
            all_targets.append(chelsa_flat)
            all_times.append(datetime(year, month, 1))
            all_decoded.append(decoded_flat)
            
            print(f"Successfully processed {year}-{month:02d}")
        except Exception as e:
            print(f"Error processing date {year}-{month:02d}: {e}")
            continue

    if not all_latents:
        raise ValueError("No valid data was processed. Check your file paths and date formats.")
    
    print(f"Successfully processed {len(all_latents)} dates. Creating output dataset...")
    
    # Debug shape information
    latent_shape = all_latents[0].shape
    print(latent_shape)
    backbone_shape = all_backbone_outputs[0].shape
    target_shape = all_targets[0].shape
    decoded_shape = all_decoded[0].shape
    print(f"Model latent shape: {latent_shape} (total pixels: {latent_shape[0]})")
    print(f"Model backbone shape: {backbone_shape} (total pixels: {backbone_shape[0]})")
    print(f"Target shape: {target_shape} (total pixels: {target_shape[0]})")
    print(f"Decoded shape: {decoded_shape} (total pixels: {decoded_shape[0]})")

    # Debug shape information before stacking
    print(f"Before stacking:")
    print(f"First latent shape: {all_latents[0].shape}")
    print(f"First backbone shape: {all_backbone_outputs[0].shape}")

    # Stack the arrays
    latent_arr = np.stack(all_latents, axis=0)
    backbone_output_arr = np.stack(all_backbone_outputs, axis=0)
    target_arr = np.stack(all_targets, axis=0)
    time_coords = np.array(all_times)
    decoded_arr = np.stack(all_decoded, axis=0)

    # Debug shape information after stacking
    print(f"After stacking:")
    print(f"Latent array shape: {latent_arr.shape}")
    print(f"Backbone array shape: {backbone_output_arr.shape}")
    
    # Set default value for use_flat_representation
    use_flat_representation = True  # Default to True for safety
    
    # Define lat/lon dimensions if available
    try:
        n_lats = len(lat_bins)
        n_lons = len(lon_bins)
        expected_pixels = n_lats * n_lons
        actual_pixels = latent_arr.shape[1] if len(latent_arr.shape) == 3 else latent_arr.shape[1] * latent_arr.shape[2]
        
        # Check if we can reshape safely based on the available grid dimensions
        if len(latent_arr.shape) == 3 and latent_arr.shape[1] == expected_pixels:
            use_flat_representation = False
            print(f"Using grid representation with {n_lats}x{n_lons} grid")
        else:
            print(f"Expected {expected_pixels} pixels for {n_lats}x{n_lons} grid, but got {actual_pixels}. Using flat representation.")
    except Exception as e:
        print(f"Error determining grid dimensions: {e}. Using flat representation.")
    
    # Check dimensions and reshape if needed
    if len(latent_arr.shape) == 4:
        print(f"Detected 4D latent array, reshaping...")
        # Reshape to flatten the first two dimensions if that's the issue
        latent_arr = latent_arr.reshape(latent_arr.shape[0], -1, latent_arr.shape[-1])
        print(f"Reshaped latent array: {latent_arr.shape}")
    
    # Convert float16 to float32 BEFORE creating datasets
    if latent_arr.dtype == np.float16:
        print("Converting float16 to float32 for NetCDF compatibility")
        latent_arr = latent_arr.astype(np.float32)
    if backbone_output_arr.dtype == np.float16:
        backbone_output_arr = backbone_output_arr.astype(np.float32)

    # Now create the datasets with the float32 data
    if use_flat_representation:
        # Use a flat pixel dimension if we can't safely reshape
        ds_model = xr.Dataset(
            {
                "encoder_output": (["time", "pixel", "embedding_dim"], latent_arr),
                "backbone_output": (["time", "pixel", "embedding_dim"], backbone_output_arr),
            },
            coords={
                "time": time_coords,
                "embedding_dim": np.arange(latent_arr.shape[-1]),
                "pixel": np.arange(latent_arr.shape[1]),
            },
        )
    else:
        # Before attempting to reshape, make sure latent_arr is 3D
        if len(latent_arr.shape) != 3:
            print(f"Warning: Expected 3D array for reshaping to (time, lat, lon, embedding_dim) but got shape {latent_arr.shape}")
            print(f"Falling back to flat representation")
            
            # Use flat representation instead
            ds_model = xr.Dataset(
                {
                    "encoder_output": (["time", "pixel", "embedding_dim"], latent_arr.reshape(latent_arr.shape[0], -1, latent_arr.shape[-1])),
                    "backbone_output": (["time", "pixel", "embedding_dim"], backbone_output_arr),
                },
                coords={
                    "time": time_coords,
                    "embedding_dim": np.arange(latent_arr.shape[-1]),
                    "pixel": np.arange(latent_arr.reshape(latent_arr.shape[0], -1, latent_arr.shape[-1]).shape[1]),
                },
            )
        else:
            # Use the lat/lon grid structure
            ds_model = xr.Dataset(
                {
                    "encoder_output": (["time", "lat", "lon", "embedding_dim"], 
                                      latent_arr.reshape(latent_arr.shape[0], n_lats, n_lons, latent_arr.shape[2])),
                    "backbone_output": (["time", "pixel", "embedding_dim"], backbone_output_arr),
                },
                coords={
                    "time": time_coords,
                    "lat": lat_bins,
                    "lon": lon_bins,
                    "embedding_dim": np.arange(latent_arr.shape[-1]),
                    "pixel": np.arange(backbone_output_arr.shape[1]),
                },
            )
    ds_target = xr.Dataset(
        {
            "target": (["time", "target_pixel", "variable"], target_arr),
            "decoder_input": (["time", "target_pixel", "variable"], decoded_arr),
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