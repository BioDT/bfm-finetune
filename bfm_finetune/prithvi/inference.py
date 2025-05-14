import argparse
import os
from pathlib import Path
from typing import Tuple

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from gravity_wave_finetuning.gravity_wave_model import UNetWithTransformer
from matplotlib.colors import TwoSlopeNorm
from torch.nn.parallel import DistributedDataParallel as DDP

from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoCLEFDataModulePrithvi,
    GeoLifeCLEFSpeciesDatasetPrithvi,
)
from bfm_finetune.prithvi.utils import (
    prithvi_inference_results_path,
    prithvi_output_checkpoint_path,
    prithvi_species_patches_location,
)

local_rank = int(os.environ["LOCAL_RANK"])
# local_rank = 0
# rank = int(os.environ["RANK"])
# rank = 0
device = f"cuda:{local_rank}"
dtype = torch.float32


def stitch_and_plot(
    patch_outputs: list,
    lat_full: np.ndarray,
    lon_full: np.ndarray,
    results_dir: Path,
    vertical_anchor: str = "center",
    horizontal_anchor: str = "center",
    lat_ascending: bool = True,
    patch_lat_size: int = 64,
    patch_lon_size: int = 128,
    n_species_to_plot: int = 5,
):
    """
    Given a list of 4 patch predictions (each with shape [1, 500, 64, 128]),
    stitches them together and plots the stitched prediction overlay on top of the full map.

    The full grid is assumed to be 152 (lat) x 320 (lon). The anchor parameters are used
    to compute the starting indices for the cropped region.

    Parameters:
      patch_outputs: list of 4 tensors, each with shape [1, S, patch_lat_size, patch_lon_size].
      lat_full: full latitude array (length 152)
      lon_full: full longitude array (length 320)
      vertical_anchor: one of {"top", "center", "bottom"}
      horizontal_anchor: one of {"left", "center", "right"}
      lat_ascending: if True, the latitude array is in increasing order.

    Returns:
      stitched_data: tensor of shape [S, 2*patch_lat_size, 2*patch_lon_size]
                     (e.g. [500, 128, 256])
    """
    # Convert to NumPy arrays and squeeze to ensure 1D.
    lat_full = np.asarray(lat_full).squeeze()
    lon_full = np.asarray(lon_full).squeeze()
    full_H, full_W = 152, 320
    crop_height = 2 * patch_lat_size  # 128
    crop_width = 2 * patch_lon_size  # 256

    # Compute row_start based on vertical_anchor:
    if vertical_anchor == "center":
        row_start = (full_H - crop_height) // 2
    elif vertical_anchor == "bottom":
        row_start = 0 if lat_ascending else full_H - crop_height
    elif vertical_anchor == "top":
        row_start = full_H - crop_height if lat_ascending else 0
    else:
        raise ValueError("vertical_anchor must be 'top', 'center' or 'bottom'.")

    # Compute col_start based on horizontal_anchor:
    if horizontal_anchor == "center":
        col_start = (full_W - crop_width) // 2
    elif horizontal_anchor == "left":
        col_start = 0
    elif horizontal_anchor == "right":
        col_start = full_W - crop_width
    else:
        raise ValueError("horizontal_anchor must be 'left', 'center' or 'right'.")

    # Remove batch dimension (assuming each patch is [1, S, 64, 128])
    def squeeze_patch(p):
        return p[0] if p.ndim == 4 else p

    patch0 = squeeze_patch(patch_outputs[0])  # top-left: [S, 64, 128]
    patch1 = squeeze_patch(patch_outputs[1])  # top-right: [S, 64, 128]
    patch2 = squeeze_patch(patch_outputs[2])  # bottom-left: [S, 64, 128]
    patch3 = squeeze_patch(patch_outputs[3])  # bottom-right: [S, 64, 128]

    # Stitch patches along width and height.
    top_row = torch.cat([patch0, patch1], dim=-1)  # shape [S, 64, 256]
    bottom_row = torch.cat([patch2, patch3], dim=-1)  # shape [S, 64, 256]
    stitched_data = torch.cat([top_row, bottom_row], dim=-2)  # shape [S, 128, 256]

    # Now, compute the geographic coordinates for the cropped (stitched) region.
    # We assume that lat_full and lon_full are one-dimensional arrays.
    lat_full = np.asarray(lat_full)
    lon_full = np.asarray(lon_full)
    sub_lat = lat_full[row_start : row_start + crop_height]
    sub_lon = lon_full[col_start : col_start + crop_width]

    stitched_data_cpu = stitched_data.detach().cpu()

    # For plotting, select one channel (e.g., channel 0).
    for channel_to_plot in tqdm.tqdm(range(n_species_to_plot), desc="Plotting species"):
        data_for_plot = stitched_data_cpu[channel_to_plot].numpy()  # shape (128, 256)

        fig, ax = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 6)
        )
        # Make sure to pass scalar floats
        # print("lon_ful ", lon_full, "\n lat full", lat_full)
        ax.set_extent(
            [
                float(lon_full[0]),
                float(lon_full[-1]),
                float(lat_full[0]),
                float(lat_full[-1]),
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.coastlines(resolution="50m")
        ax.set_title("Stitched Prediction Overlay")

        Lon_grid, Lat_grid = np.meshgrid(sub_lon, sub_lat)
        norm = TwoSlopeNorm(
            vmin=data_for_plot.min(), vcenter=0, vmax=data_for_plot.max()
        )
        cf = ax.contourf(
            Lon_grid,
            Lat_grid,
            data_for_plot,
            levels=60,
            cmap="RdBu_r",
            norm=norm,
            alpha=0.6,
            transform=ccrs.PlateCarree(),
        )
        # Optionally add a rectangle indicating the cropped area

        fig.colorbar(cf, ax=ax, orientation="vertical", label="Prediction Value")
        plt.tight_layout()
        plt.savefig(str(results_dir / f"inference_channel_{channel_to_plot}.png"))
        plt.show()
    print(f"Saving result fig at {results_dir}")
    return stitched_data, row_start, col_start


def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)


def cleanup():
    dist.destroy_process_group()


def load_checkpoint(model, ckpt_singular):

    print("Loading weights from", ckpt_singular)
    state_dict = torch.load(f=ckpt_singular, map_location=device, weights_only=True)

    ignore_layers = [
        "input_scalers_mu",
        "input_scalers_sigma",
        "static_input_scalers_mu",
        "static_input_scalers_sigma",
        "patch_embedding.proj.weight",
        "patch_embedding_static.proj.weight",
        "unembed.weight",
        "unembed.bias",
        "output_scalers",
    ]

    for layer in ignore_layers:
        state_dict.pop(layer, None)
    model.load_state_dict(state_dict)
    print("Loaded weights")
    return model


def get_model(cfg, vartype, ckpt_singular: str) -> torch.nn.Module:
    model: torch.nn.Module = UNetWithTransformer(
        lr=cfg.lr,
        hidden_channels=cfg.hidden_channels,
        in_channels=500,
        out_channels=500,
        n_lats_px=cfg.n_lats_px,
        n_lons_px=cfg.n_lons_px,
        in_channels_static=cfg.in_channels_static,
        mask_unit_size_px=cfg.mask_unit_size_px,
        patch_size_px=cfg.patch_size_px,
        device=device,
    )
    model = DDP(model.to(local_rank, dtype=dtype), device_ids=[local_rank])
    model = load_checkpoint(model, ckpt_singular)

    return model


def get_data() -> Tuple[torch.utils.data.DataLoader, GeoLifeCLEFSpeciesDatasetPrithvi]:
    datamodule = GeoCLEFDataModulePrithvi(
        batch_size=1,
        num_data_workers=1,
    )
    datamodule.setup(stage="predict")
    dataloader = datamodule.predict_dataloader()
    dataset = datamodule.dataset_predict
    return dataloader, dataset


def main(cfg, vartype, ckpt_path: str, results_dir: Path, n_species_to_plot: int):
    setup()

    model: torch.nn.Module = get_model(cfg, vartype, ckpt_singular=ckpt_path)
    dataloader, val_dataset = get_data()

    patch_outputs = []  # to store output from each patch inference.
    # Main prediction loop
    total: int = len(dataloader)
    pbar = tqdm.tqdm(iterable=enumerate(dataloader), total=total)
    for i, batch in pbar:
        batch = {
            k: v.to(device="cuda") for k, v in batch.items()
        }  # move data to the same device as the model

        with torch.inference_mode():
            output: torch.Tensor = model(batch)  # run inference
            print("Output shape:", output.shape)  # [1, 500, 64, 128]
            unnormalized_preds = val_dataset.scale_species_distribution(
                output.clone(), unnormalize=True
            )
            patch_outputs.append(unnormalized_preds)
            # Report loss
            loss: torch.Tensor = F.mse_loss(input=output, target=batch["target"])
            pbar.set_postfix(
                ordered_dict={
                    # "t_slice": f"{t_slice.start}-{t_slice.stop}",  # time slice
                    "loss": loss.item(),
                }
            )

    lat_full = batch["lat_original"].cpu().numpy()  # Expected shape: (152,)
    lon_full = batch["lon_original"].cpu().numpy()  # Expected shape: (320,)

    # Specify anchor settings (should match those used during patchification).
    vertical_anchor = "center"
    horizontal_anchor = "center"
    lat_ascending = True

    # Stitch the 4 patch outputs and plot the stitched prediction overlay.
    if len(patch_outputs) == 4:
        # Now stitch and plot the predictions.
        stitched_data, row_start, col_start = stitch_and_plot(
            patch_outputs,
            lat_full,
            lon_full,
            results_dir=results_dir,
            vertical_anchor=vertical_anchor,
            horizontal_anchor=horizontal_anchor,
            lat_ascending=lat_ascending,
            patch_lat_size=64,
            patch_lon_size=128,
            n_species_to_plot=n_species_to_plot,
        )
    else:
        print("Warning: Expected 4 patches for stitching but got", len(patch_outputs))

    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="uvtp122",
    )
    parser.add_argument(
        "--ckpt_path",
        default=str(prithvi_output_checkpoint_path),
    )
    parser.add_argument(
        "--data_path",
        default=str(prithvi_species_patches_location / "val"),
    )
    parser.add_argument(
        "--results_dir",
        default=str(prithvi_inference_results_path),
    )
    parser.add_argument(
        "--n_species_to_plot",
        default=50,
    )
    args = parser.parse_args()

    from config import get_cfg

    cfg = get_cfg()
    os.makedirs(name=args.results_dir, exist_ok=True)
    main(
        cfg,
        args.split,
        args.ckpt_path,
        results_dir=Path(args.results_dir),
        n_species_to_plot=args.n_species_to_plot,
    )
    cleanup()
