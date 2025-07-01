from typing import Literal

import numpy as np
import torch
from aurora.batch import Batch, Metadata
from torch.utils.data import DataLoader, Dataset, default_collate

from typing import Dict, Callable, Literal, Tuple

# Custom collate function to merge a list of Batch objects.
def collate_batches(batch_list):
    # Merge surf_vars, static_vars, and atmos_vars by stacking their tensor values.
    surf_vars = {
        k: torch.stack([b.surf_vars[k] for b in batch_list], dim=0)
        for k in batch_list[0].surf_vars.keys()
    }
    # static_vars = {
    #     k: torch.stack([b.static_vars[k] for b in batch_list], dim=0)
    #     for k in batch_list[0].static_vars.keys()
    # }
    atmos_vars = {
        k: torch.stack([b.atmos_vars[k] for b in batch_list], dim=0)
        for k in batch_list[0].atmos_vars.keys()
    }
    # For static_vars, we simply take the one from the first sample.
    static_vars = batch_list[0].static_vars
    # For metadata, we assume they are the same across samples.
    metadata: Metadata = batch_list[0].metadata
    metadata.time = tuple(
        el.metadata.time for el in batch_list
    )  # metadata.time needs to be merged
    return Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=metadata,
    )


def collate_batches_dict(batch_list):
    # Merge surf_vars, static_vars, and atmos_vars by stacking their tensor values.
    species_distribution = torch.stack(
        [b["species_distribution"] for b in batch_list], dim=0
    )
    # For metadata, we assume they are the same across samples.
    metadata = batch_list[0]["metadata"]
    metadata["time"] = tuple(
        el["metadata"]["time"] for el in batch_list
    )  # metadata.time needs to be merged
    return {
        "species_distribution": species_distribution,
        "metadata": metadata,
    }


def custom_collate_fn(samples):
    # Each sample is a dict with keys "batch" and "target".
    batch_list = [s["batch"] for s in samples]
    collated_batch = collate_batches_dict(batch_list)
    targets = default_collate([s["target"] for s in samples])
    return {"batch": collated_batch, "target": targets}


def manage_negative_lon(
    batch: Batch, mode: Literal["roll", "exclude", "translate"]
) -> Batch:
    raise NotImplementedError()


def manage_negative_lon_dict(
    batch: dict, mode: Literal["roll", "exclude", "translate", "ignore"]
) -> dict:
    # print("manage_negative_lon_dict", mode)
    if mode == "exclude":
        lon_range = batch["metadata"]["lon"].numpy()
        lon_neg_loc = np.where(lon_range < 0)[0]
        max_neg = lon_neg_loc.max()
        # print("max_neg", max_neg, lon_range[max_neg])
        lon_range = lon_range[max_neg + 1 :]
        batch["metadata"]["lon"] = torch.Tensor(lon_range)
        # also crop species distribution
        species_matrix = batch["species_distribution"]
        species_matrix = species_matrix[..., max_neg + 1 :]
        batch["species_distribution"] = torch.Tensor(species_matrix)
        pass
    elif mode == "roll":
        # roll negative after
        lon_range = batch["metadata"]["lon"].numpy()
        lon_neg_loc = np.where(lon_range < 0)[0]
        max_neg = lon_neg_loc.max()
        # print("max_neg", max_neg)
        lon_range[: max_neg + 1] += 360
        lon_range = np.roll(lon_range, shift=-(max_neg + 1))
        batch["metadata"]["lon"] = torch.Tensor(lon_range)
        # also roll species_matrix
        species_matrix = batch["species_distribution"]
        # print(species_matrix.shape)
        lon_axis = species_matrix.dim() - 1  # longitude axis is always the last one
        species_matrix = np.roll(species_matrix, shift=-(max_neg + 1), axis=lon_axis)
        batch["species_distribution"] = torch.Tensor(species_matrix)
    elif mode == "translate":
        # just shift it, faking metadata
        min_value = batch["metadata"]["lon"].numpy().min()
        if min_value < 0:
            batch["metadata"]["lon"] = (
                batch["metadata"]["lon"] - min_value
            )  # min_value is negative
        pass
    return batch


def _apply_lastdim(var_dict: Dict[str, torch.Tensor],
                   fn: Callable[[torch.Tensor], torch.Tensor],
                   size: int) -> None:
    """Apply fn to every tensor whose last dimension == size."""
    for k, t in var_dict.items():
        if t.ndim and t.shape[-1] == size:
            var_dict[k] = fn(t)

def _apply_latdim(var_dict: Dict[str, torch.Tensor],
                  fn: Callable[[torch.Tensor], torch.Tensor],
                  size: int) -> None:
    """Apply fn to every tensor whose second-to-last dim == size."""
    for k, t in var_dict.items():
        if t.ndim >= 2 and t.shape[-2] == size:
            var_dict[k] = fn(t)

def manage_negative_lon_aurora_batch(
    surf_vars: Dict[str, torch.Tensor],
    static_vars: Dict[str, torch.Tensor],
    atmos_vars: Dict[str, torch.Tensor],
    lat: torch.Tensor,
    lon: torch.Tensor,
    mode: Literal["roll", "exclude", "translate", "ignore"] = "ignore",
    lon_precision: int = 4,
):
    """
    Fix longitude and latitude axes **in-place** for three variable dicts.

    Returns the same objects so the caller can simply re-assign.

    Longitude modes are identical to the GeoLifeclef version.
    Latitude is always forced to strictly decreasing order.

    Parameters
    ----------
    surf_vars / static_vars / atmos_vars : dict[str, Tensor]
        Tensors shaped (...,  H, W) where H matches lat and W matches lon.
    lat : 1-D Tensor (H) - will be flipped if increasing.
    lon : 1-D Tensor(W) - will be modified according to mode.
    lon_precision : int - number of decimals kept in lon after rounding.
    """
    # ensure latitude strictly decreasing
    if lat[0] < lat[-1]: # increasing -> flip
        lat = torch.flip(lat, dims=[0])
        flip_fn = lambda t: torch.flip(t, dims=[-2])
        for g in (surf_vars, static_vars, atmos_vars):
            _apply_latdim(g, flip_fn, lat.shape[0])

    if mode != "ignore":
        lon_np   = lon.cpu().numpy()
        neg_idx  = np.where(lon_np < 0)[0]
        if neg_idx.size:
            max_neg = int(neg_idx.max())
            W       = lon.shape[-1]
            groups  = (surf_vars, static_vars, atmos_vars)

            if mode == "exclude":  # drop western hemi
                keep = slice(max_neg + 1, None)
                lon  = lon[keep]
                cut  = lambda t: t[..., keep]
                for g in groups: _apply_lastdim(g, cut, W)

            elif mode == "roll": # 0-360 ° grid
                lon360 = lon_np.copy()
                lon360[:max_neg+1] += 360 
                shift = -(max_neg + 1)
                lon  = torch.as_tensor(np.roll(lon360, shift),
                                                 dtype=lon.dtype, device=lon.device)
                roll_fn = lambda t: torch.roll(t, shifts=shift, dims=-1)
                for g in groups: _apply_lastdim(g, roll_fn, W)

            elif mode == "translate":
                min_val = lon.min()
                if min_val < 0:
                    lon = lon - min_val  # keep order, just shift baseline

            else:
                raise ValueError(f"unknown mode {mode}")

    lon = torch.round(lon * (10 ** lon_precision)) / (10 ** lon_precision)
    torch.set_printoptions(sci_mode=False, precision=lon_precision,
                           linewidth=200)           

    return surf_vars, static_vars, atmos_vars, lat, lon