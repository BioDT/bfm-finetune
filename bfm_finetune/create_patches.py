#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import numpy as np
import torch


def create_four_anchor_patches(
    species_distribution: torch.Tensor,
    patch_lat_size: int = 64,
    patch_lon_size: int = 128,
    vertical_anchor: str = "center",  # Options: "top", "center", "bottom"
    horizontal_anchor: str = "center",  # Options: "left", "center", "right"
    lat_ascending: bool = True,  # True if lat increases (e.g. from 34 to 72)
) -> tuple:
    """
    Extract a contiguous region of size (2*patch_lat_size, 2*patch_lon_size)
    from the full grid. The input tensor can have shape either:
        - [S, 152, 320]  (no time dimension)
        - [T, S, 152, 320] (with T timesteps, here T=2)
    The cropping is applied to the last two dimensions.

    Returns:
      (patches, row_start, col_start)

      - patches is a list of 4 tensors. If the input is 4D, then each patch will have shape
        [T, S, patch_lat_size, patch_lon_size]. If the input is 3D, then each patch will have shape
        [S, patch_lat_size, patch_lon_size].
      - row_start and col_start are the starting indices (in the height and width dimensions) used for cropping.
    """
    if species_distribution.ndim == 4:
        # Input shape: [T, S, H, W]
        T, S, full_lat, full_lon = species_distribution.shape
    elif species_distribution.ndim == 3:
        # Input shape: [S, H, W]
        S, full_lat, full_lon = species_distribution.shape
        T = None
    else:
        raise ValueError("Input tensor must be either 3D or 4D.")

    assert (
        full_lat == 152 and full_lon == 320
    ), f"Expected grid shape (H,W) = (152,320), got ({full_lat},{full_lon})"

    # Overall crop dimensions to be split into 4 patches.
    crop_height = 2 * patch_lat_size  # e.g., 128 if patch_lat_size=64
    crop_width = 2 * patch_lon_size  # e.g., 256 if patch_lon_size=128

    if vertical_anchor == "center":
        row_start = (full_lat - crop_height) // 2
    elif vertical_anchor == "bottom":
        row_start = 0 if lat_ascending else full_lat - crop_height
    elif vertical_anchor == "top":
        row_start = full_lat - crop_height if lat_ascending else 0
    else:
        raise ValueError("vertical_anchor must be 'top', 'center', or 'bottom'.")

    if horizontal_anchor == "center":
        col_start = (full_lon - crop_width) // 2
    elif horizontal_anchor == "left":
        col_start = 0
    elif horizontal_anchor == "right":
        col_start = full_lon - crop_width
    else:
        raise ValueError("horizontal_anchor must be 'left', 'center', or 'right'.")

    # Function to slice either a 3D or 4D tensor along H and W.
    def slice_tensor(x, r0, r1, c0, c1):
        if x.ndim == 4:
            return x[:, :, r0:r1, c0:c1]
        else:
            return x[:, r0:r1, c0:c1]

    patch_top_left = slice_tensor(
        species_distribution,
        row_start,
        row_start + patch_lat_size,
        col_start,
        col_start + patch_lon_size,
    )
    patch_top_right = slice_tensor(
        species_distribution,
        row_start,
        row_start + patch_lat_size,
        col_start + patch_lon_size,
        col_start + 2 * patch_lon_size,
    )
    patch_bottom_left = slice_tensor(
        species_distribution,
        row_start + patch_lat_size,
        row_start + 2 * patch_lat_size,
        col_start,
        col_start + patch_lon_size,
    )
    patch_bottom_right = slice_tensor(
        species_distribution,
        row_start + patch_lat_size,
        row_start + 2 * patch_lat_size,
        col_start + patch_lon_size,
        col_start + 2 * patch_lon_size,
    )

    patches = [patch_top_left, patch_top_right, patch_bottom_left, patch_bottom_right]
    return patches, row_start, col_start


def process_pt_file(
    file_path: Path,
    output_subfolder: Path,
    vertical_anchor: str = "center",
    horizontal_anchor: str = "center",
    lat_ascending: bool = True,
) -> list:
    """
    Processes a single .pt file by:
      - Loading the file.
      - Extracting the species_distribution which has shape [T, S, 152, 320] (T=2 timesteps).
      - Creating 4 patches from the entire tensor (so patches will have shape [T, S, 64, 128]).
      - Embedding the spatial metadata (including coordinate subsets) and the timestamps.
      - Saving each patch as a .pt file in the output_subfolder.

    Returns a list of saved file paths.
    """
    data = torch.load(file_path, map_location="cpu")
    # Expecting species_distribution with shape [T, S, 152, 320]; T should be 2.
    distribution = data["species_distribution"]
    if distribution.shape[0] < 2:
        raise ValueError(f"File {file_path} does not have at least 2 timesteps.")
    # Now pass the entire distribution; patchification will be applied along last two dims.
    # So the resulting patches will have shape [T, S, 64, 128].
    patches, row_start, col_start = create_four_anchor_patches(
        distribution,
        patch_lat_size=64,
        patch_lon_size=128,
        vertical_anchor=vertical_anchor,
        horizontal_anchor=horizontal_anchor,
        lat_ascending=lat_ascending,
    )
    # Load full coordinate arrays from metadata.
    lat_array = np.asarray(data["metadata"]["lat"])  # shape (152,)
    lon_array = np.asarray(data["metadata"]["lon"])  # shape (320,)

    timestamps = {}
    if "lead_time" in data:
        timestamps["lead_time"] = data["lead_time"]
    if "input_time" in data:
        timestamps["input_time"] = data["input_time"]

    base_name = file_path.stem  # e.g., "yearly_species_2017-2018"
    saved_files = []
    for i, patch in enumerate(patches):
        if i == 0:  # Top-left
            rs = row_start
            cs = col_start
        elif i == 1:  # Top-right
            rs = row_start
            cs = col_start + 128
        elif i == 2:  # Bottom-left
            rs = row_start + 64
            cs = col_start
        elif i == 3:  # Bottom-right
            rs = row_start + 64
            cs = col_start + 128

        # Extract the coordinate subsets for this patch.
        lat_patch = lat_array[rs : rs + 64]
        lon_patch = lon_array[cs : cs + 128]

        patch_metadata = {
            "row_start": rs,
            "col_start": cs,
            "lat_start": float(lat_array[rs]),
            "lon_start": float(lon_array[cs]),
            "patch_lat_size": 64,
            "patch_lon_size": 128,
            "lat_patch": lat_patch.tolist(),
            "lon_patch": lon_patch.tolist(),
            "timestamps": timestamps,  # Contains both timesteps' info, if available.
            "lat_array": lat_array,
            "lon_array": lon_array,
        }
        out_dict = {
            "patch": patch,  # Note: patch shape is [T, S, 64, 128]
            "metadata": patch_metadata,
        }
        out_file = (
            output_subfolder
            / f"{base_name}_patch_{i}_rs{rs}_cs{cs}_lat{lat_array[rs]:.2f}_lon{lon_array[cs]:.2f}.pt"
        )
        torch.save(out_dict, out_file)
        print(f"Saved patch {i} from {file_path.name} to {out_file}")
        saved_files.append(out_file)
    return saved_files


def process_dataset_folder(
    input_folder: Path,
    output_folder: Path,
    vertical_anchor: str = "center",
    horizontal_anchor: str = "center",
    lat_ascending: bool = True,
):
    """
    Processes an input folder with subfolders "train" and "val". For each .pt file,
    applies patchification (creating 4 patches per file with the full 2 timesteps),
    and saves the new patches (with embedded metadata and timestamps) into corresponding
    subfolders in the output_folder.
    """
    for sub in ["train", "val"]:
        input_subfolder = input_folder / sub
        output_subfolder = output_folder / sub
        output_subfolder.mkdir(parents=True, exist_ok=True)
        pt_files = sorted(input_subfolder.glob("*.pt"))
        print(f"Processing {len(pt_files)} files in {input_subfolder}")
        for pt_file in pt_files:
            process_pt_file(
                pt_file,
                output_subfolder,
                vertical_anchor,
                horizontal_anchor,
                lat_ascending,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Patchify GeoLifeCLEF .pt files (with 'train' and 'val' subfolders) that contain 2 timesteps "
        "into 4 patches per file (handling the entire 2-timestep tensor) while embedding coordinate and timestamp metadata."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to input folder with 'train' and 'val' subfolders.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to output folder where patched files will be saved.",
    )
    parser.add_argument(
        "--vertical_anchor",
        type=str,
        default="bottom",
        choices=["top", "center", "bottom"],
        help="Vertical anchor for cropping (default: center).",
    )
    parser.add_argument(
        "--horizontal_anchor",
        type=str,
        default="center",
        choices=["left", "center", "right"],
        help="Horizontal anchor for cropping (default: center).",
    )
    parser.add_argument(
        "--lat_ascending",
        action="store_true",
        default=False,
        help="Flag indicating that the latitude array is in ascending order (default).",
    )
    parser.add_argument(
        "--lat_descending",
        dest="lat_ascending",
        action="store_false",
        help="Flag indicating that the latitude array is in descending order.",
    )
    parser.set_defaults(lat_ascending=True)

    args = parser.parse_args()
    input_folder = Path("data/finetune/geolifeclef24/aurorashape_species")
    output_folder = Path("data_patched")

    process_dataset_folder(
        input_folder,
        output_folder,
        args.vertical_anchor,
        args.horizontal_anchor,
        args.lat_ascending,
    )
