"""
Copyright 2025 (C) TNO. Licensed under the MIT license.
"""

import os
from collections import namedtuple
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Union

import torch
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader, Dataset, default_collate
from dataloaders.dataloader_utils import manage_negative_lon_aurora_batch

from bfm_model.bfm.scaler import (
    _rescale_recursive,
    dimensions_to_keep_monthly,
    load_stats,
)

from aurora import Batch, Metadata


def crop_variables(variables, new_H, new_W, handle_nans=False, nan_mode="zero", fix_dim=False):
    """
    Crop and clean variables to specified dimensions, handling NaN and Inf values.

    Args:
        variables (dict): Dictionary of variable tensors to process
        new_H (int): Target height dimension
        new_W (int): Target width dimension
        handle_nans (bool): Whether to handle NaN values at all
        nan_mode (str): Strategy for NaN handling.
            - "mean_clip": old logic (replace NaNs with mean, clip to mean ± 2*std)
            - "zero": replace all NaNs with 0.0, no extra clipping

    Returns:
        dict: Processed variables with cleaned and cropped tensors
    """
    processed_vars = {}
    for k, v in variables.items():
        # crop dimensions
        if fix_dim:
            cropped = v[:, :new_H, :new_W, :]
        else:
            cropped = v[..., :new_H, :new_W]
        # Handle infinities
        inf_mask = torch.isinf(cropped)
        inf_count = inf_mask.sum().item()
        if inf_count > 0:
            valid_values = cropped[~inf_mask & ~torch.isnan(cropped)]
            if len(valid_values) > 0:
                max_val = valid_values.max().item()
                min_val = valid_values.min().item()
                cropped = torch.clip(cropped, min_val, max_val)
            else:
                cropped = torch.clip(cropped, -1e6, 1e6)

        # Handle NaNs if requested
        if handle_nans:
            nan_mask = torch.isnan(cropped)
            nan_count = nan_mask.sum().item()
            if nan_count > 0:
                if nan_mode == "mean_clip":
                    valid_values = cropped[~nan_mask & ~torch.isinf(cropped)]
                    if len(valid_values) > 0:
                        mean_val = valid_values.mean().item()
                        std_val = valid_values.std().item()
                        clip_min = mean_val - 2 * std_val
                        clip_max = mean_val + 2 * std_val

                        # Replace NaNs with mean
                        cropped = torch.nan_to_num(cropped, nan=mean_val)
                        # Convert to float32 if needed
                        cropped = cropped.to(torch.float32)
                        # Clip
                        cropped = torch.clip(cropped, clip_min, clip_max)
                    else:
                        # If no valid values, just fill with 0 and do a small clip
                        cropped = torch.nan_to_num(cropped, nan=0.0)
                        cropped = torch.clip(cropped, -1.0, 1.0)

                elif nan_mode == "zero":
                    # Simply replace all NaNs with 0.0
                    cropped = torch.nan_to_num(cropped)
                    # cropped = cropped.to(torch.float32)

                else:
                    raise ValueError(f"Unknown nan_mode: {nan_mode}")
        # else: do nothing special for NaNs

        processed_vars[k] = cropped

    return processed_vars


class LargeClimateDataset(Dataset):
    """
    A dataset where each file in `data_dir` is a single sample.
    Each file should have structure:
    {
        "batch_metadata" {...},
        "surface_variables" {...},
        "edaphic_variables" {...},
        "atmospheric_variables" {...},
        "climate_variables" {...},
        "species_variables" {...},
        "vegetation_variables" {...},
        "land_variables" {...},
        "agriculture_variables" {...},
        "forest_variables" {...},
        "redlist_variables" {...}
        "misc_variables" {...},
    }
    """

    def __init__(
        self,
        data_dir: str,
        scaling_settings: DictConfig,
        num_species: int = 2,
        atmos_levels: list = [50],
        mode: str = "pretrain",
        model_patch_size: int = 4,
        max_files: int | None = None,
    ):
        self.data_dir = data_dir
        self.num_species = num_species
        self.atmos_levels = atmos_levels
        self.mode = mode
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")]
        self.max_files = max_files
        self.files.sort()
        if self.max_files:
            self.files = self.files[: self.max_files]
        self.scaling_settings = scaling_settings
        self.scaling_statistics = load_stats(scaling_settings.stats_path)
        self.model_patch_size = model_patch_size
        print(f"We scale the dataset {scaling_settings.enabled} with {scaling_settings.mode}")

    def __len__(self):
        if self.mode == "pretrain":
            return max(0, len(self.files) - 1)
        else:
            return len(self.files)

    def load_and_process_files(self, fpath: str):
        data = torch.load(fpath, map_location="cpu", weights_only=False)

        latitudes = data["batch_metadata"]["latitudes"]
        longitudes = data["batch_metadata"]["longitudes"]
        timestamps = data["batch_metadata"]["timestamp"]
        pressure_levels = data["batch_metadata"]["pressure_levels"]
        # Determine original spatial dimensions from metadata lists
        H = len(data["batch_metadata"]["latitudes"])
        W = len(data["batch_metadata"]["longitudes"])

        # crop dimensions to be divisible by patch size
        new_H = (H // self.model_patch_size) * self.model_patch_size
        new_W = (W // self.model_patch_size) * self.model_patch_size
        # normalize or standardize variables
        data = self.scale_batch(data, direction="scaled")

        surface_vars = crop_variables(data["surface_variables"], new_H, new_W)
        atmospheric_vars = crop_variables(data["atmospheric_variables"], new_H, new_W)
        # Selection for Aurora
        static_vars = {k: surface_vars[k][0] for k in ("z", "lsm", "slt") if k in surface_vars}
        raw_surf = {k: surface_vars[k] for k in ("t2m", "u10", "v10", "msl") if k in surface_vars}
        # Rename of the surface exxpected by Aurora
        surf_vars = self.rename_surface_vars(raw_surf)

        # crop metadata dimensions
        latitude_var = torch.tensor(latitudes[:new_H])
        longitude_var = torch.tensor(longitudes[:new_W])

        atmospheric_vars = extract_atmospheric_levels(atmospheric_vars, pressure_levels, self.atmos_levels, level_dim=1)

        surf_vars, static_vars, atmos_vars, lat, lon = manage_negative_lon_aurora_batch(
            surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmospheric_vars,
            lat=latitude_var, lon=longitude_var, mode="roll")

        time_tupe = tuple(datetime.fromisoformat(t) for t in timestamps)
        # print(lat, lon)
        metadata = Metadata(
            lat=lat,
            lon=lon,
            time=time_tupe,
            atmos_levels=pressure_levels,
        )

        batch = Batch(
            metadata=metadata,
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
        )
        return batch

    def __getitem__(self, idx):
        fpath_x = self.files[idx]
        if self.mode == "pretrain":
            fpath_y = self.files[idx + 1]
            x = self.load_and_process_files(fpath_x)
            y = self.load_and_process_files(fpath_y)
            return x, y
        else:  # finetune
            x = self.load_and_process_files(fpath_x)
            return x

    def scale_batch(self, batch: dict | Batch, direction: Literal["original", "scaled"] = "scaled"):
        """
        Scale a batch of data back or forward.
        """
        if not self.scaling_settings.enabled:
            return batch
        convert_to_batch = False
        if isinstance(batch, Batch):
            # convert from NamedTuple to dict
            batch = batch._asdict()
            convert_to_batch = True
        _rescale_recursive(
            batch,
            self.scaling_statistics,
            dimensions_to_keep_by_key=dimensions_to_keep_monthly,
            mode=self.scaling_settings.mode,
            direction=direction,
        )
        if convert_to_batch:
            # convert back to NamedTuple
            batch = Batch(**batch)
        return batch

    def rename_surface_vars(self, surface_vars: dict) -> dict:
        """
        Convert ERA/ECMWF raw names (t2m, u10, …) into the short tokens Aurora expects.
        Returns a new dict so the caller may keep the original untouched.
        """
        RENAME_SURF = {"t2m": "2t", "u10": "10u", "v10": "10v", "msl": "msl"}

        return {RENAME_SURF[k]: surface_vars[k]
                for k in RENAME_SURF if k in surface_vars}
                

def extract_atmospheric_levels(
    atmos_vars: Dict[str, torch.Tensor],
    all_levels: List[int],
    desired_levels: List[int],
    level_dim: int = 2,
) -> Dict[str, torch.Tensor]:
    """
    Given a dict mapping variable names → tensors that have a 'levels'
    axis at position `level_dim`, return a new dict where each tensor
    has been sliced to KEEP ONLY those levels in `desired_levels`.

    - atmos_vars: e.g. {"temperature": Tensor[B,C,D,H,W], "humidity": …}
    - all_levels: the full list of level values (length D)
    - desired_levels: the subset of levels you want to extract, e.g. [50, 60, 100]
    - level_dim: the tensor dimension index where levels live (default 2)

    Returns a new dict with the same keys, but each Tensor is now
    shape (..., len(desired_levels), ...).
    """
    # Map desired level values -> their integer indices in all_levels
    idxs: List[int] = []
    for lvl in desired_levels:
        try:
            idxs.append(all_levels.index(lvl))
        except ValueError:
            raise ValueError(f"Level {lvl!r} not found in available levels {all_levels}")

    # Build a 1D index tensor on the same device our data
    device = next(iter(atmos_vars.values())).device
    idx_tensor = torch.tensor(idxs, dtype=torch.long, device=device)

    # Slice each variable via index_select
    filtered: Dict[str, torch.Tensor] = {}
    for name, tensor in atmos_vars.items():
        filtered[name] = tensor.index_select(dim=level_dim, index=idx_tensor)

    return filtered


def aurora_batch_collate(samples: list[Batch]) -> Batch:
    ref = samples[0]
    def stack(group: str) -> dict[str, torch.Tensor]:
        return {
            k: torch.stack([getattr(s, group)[k] for s in samples], dim=0)
            for k in getattr(ref, group)
        }
    static_clean = {
        k: v.squeeze()
        for k, v in ref.static_vars.items()
    }
    return Batch(
        surf_vars  = stack("surf_vars"),
        atmos_vars = stack("atmos_vars"),
        static_vars= static_clean,
        metadata   = ref.metadata,
    )