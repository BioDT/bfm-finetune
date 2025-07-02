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



def fix_aurora_coords(lat: torch.Tensor,
                      lon: torch.Tensor,
                      precision: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return lat, lon coordinates that satisfy Aurora's requirements:

    -lat strictly descending-ordered.
    -lon strictly ascending-ordered in [0, 360).

    The algorithm is deterministic and independent of the
    incoming order or sign of lon.
    """
    lat_out = lat.flip(0) if lat[0] < lat[-1] else lat.clone()
    if not torch.all(lat_out[1:] < lat_out[:-1]):
        raise ValueError("Latitude still not strictly decreasing after flip.")

    lon_out = torch.remainder(lon, 360.0)

    # Drop duplicates & sort ascending
    lon_out = torch.unique(lon_out, sorted=True)

    if lon_out.numel() > 1 and lon_out[0] == 0.0:
        pass
    else:
        # Make it start at 0 by rolling so the largest gap is at the end.
        diffs = torch.diff(torch.cat([lon_out, lon_out[:1] + 360]))
        gap_idx = int(torch.argmax(diffs)) + 1
        lon_out = torch.roll(lon_out, shifts=-gap_idx, dims=0)

    if not torch.all(lon_out[1:] > lon_out[:-1]):
        raise ValueError("Longitude still not strictly increasing (unexpected).")
    if not (0.0 <= lon_out.min() and lon_out.max() < 360.0):
        raise ValueError("Longitude outside [0,360) (unexpected).")

    lon_out = torch.round(lon_out, decimals=precision)
    lat_out = torch.round(lat_out, decimals=precision)
    
    return lat_out, lon_out