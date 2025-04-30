import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path
import torch

# Define the bounding box for Europe: [lon_min, lon_max, lat_min, lat_max].
EUROPE_EXTENT = [-30, 40, 34.25, 72]

def plot_eval(
    batch,
    prediction_species: torch.Tensor | None,
    out_dir: Path,
    n_species_to_plot: int = 20,
    save: bool = True,
    region_extent=None
):
    """
    Plots evaluation results for species predictions, focusing on a specific bounding box (Europe by default).
    
    Args:
        batch (Batch): A Batch object with metadata and surf_vars.
        prediction_species (torch.Tensor): Model predictions with shape [B, T, S, H, W].
        out_dir (Path): Directory where to save figures.
        n_species_to_plot (int): Number of species channels to plot.
        save (bool): If True, figures are saved to out_dir.
        region_extent (list or None): If not None, a list [lon_min, lon_max, lat_min, lat_max]
                                     specifying the bounding box to display. Defaults to Europe.
    """
    if region_extent is None:
        region_extent = EUROPE_EXTENT  # Default bounding box for Europe

    metadata = batch["metadata"]
    lat = metadata["lat"].cpu().numpy()  # shape (152,)
    lon = metadata["lon"].cpu().numpy()  # shape (320,)
    print(lon.min(), lon.max())

    time = metadata["time"]  # e.g. (datetime1, datetime2)

    # shape [B, T=2, S, H, W]
    species_distribution = batch["species_distribution"]
    t0_species = species_distribution[:, 0, :, :, :]   # shape [B, S, H, W]
    target_species = species_distribution[:, 1, :, :, :]
    # shape [B, T=1, S, H, W] => take first time dim
    if prediction_species is not None:
        prediction_species = prediction_species[:, 0, :, :, :]

    for single_b in range(t0_species.shape[0]):
        # For each batch element, plot the species.
        plot_single(
            t0_species[single_b],
            target_species[single_b],
            prediction_species[single_b] if prediction_species is not None else None,
            times=time[single_b],
            lat=lat,
            lon=lon,
            n_species_to_plot=n_species_to_plot,
            region_extent=region_extent,
            out_dir=out_dir,
            save=save,
        )

def create_subfig(fig, ax, lat, lon, matrix, title, label, region_extent, roll_back=False):
    """
    Creates an individual subplot, using a fixed bounding box (region_extent) for Europe.
    
    Args:
        fig (Figure): Matplotlib figure.
        ax (Axes): Matplotlib axes with projection=ccrs.PlateCarree().
        lat (np.ndarray): 1D array of latitudes, shape (H,).
        lon (np.ndarray): 1D array of longitudes, shape (W,).
        matrix (np.ndarray): 2D data array of shape [H, W].
        title (str): Title for the subplot.
        label (str): Colorbar label text.
        region_extent (list): [lon_min, lon_max, lat_min, lat_max].
        roll_back (bool): If True, roll longitudes back to -180 to 180 range.
    """
    if roll_back:
        # Roll longitudes back to -180 to 180 range
        lon = np.where(lon > 180, lon - 360, lon)
        sort_idx = np.argsort(lon)
        lon = lon[sort_idx]
        matrix = matrix[:, sort_idx]

    # Create a meshgrid from the original lat/lon (assuming lat and lon match matrix's shape).
    Lon, Lat = np.meshgrid(lon, lat, indexing="xy")

    # Set the bounding box to the region_extent
    ax.set_extent(region_extent, crs=ccrs.PlateCarree())

    try:
        ax.coastlines(resolution="50m")
    except Exception as e:
        print("Error drawing coastlines:", e)

    # Clip matrix values to avoid extreme artifacts in visualization
    # disabled because species are very sparse and local
    # matrix = np.clip(matrix, np.percentile(matrix, 1), np.percentile(matrix, 99))

    # Plot the data as a filled contour, with 60 levels.
    cf2 = ax.contourf(
        Lon,
        Lat,
        matrix,
        levels=60,
        cmap="viridis",
        transform=ccrs.PlateCarree()
    )
    ax.set_title(title)
    fig.colorbar(cf2, ax=ax, orientation="vertical", label=label)

def plot_single(
    t0_species: torch.Tensor,
    target_species: torch.Tensor,
    prediction_species: torch.Tensor | None,
    times,
    lat: np.ndarray,
    lon: np.ndarray,
    n_species_to_plot: int,
    region_extent,
    out_dir: Path,
    save: bool
):
    """
    Plots T0, Target, and Prediction for each species channel, focusing on the region_extent bounding box.
    
    Args:
        t0_species (torch.Tensor): shape [S, H, W].
        target_species (torch.Tensor): shape [S, H, W].
        prediction_species (torch.Tensor): shape [S, H, W].
        times (tuple): e.g. (datetime1, datetime2).
        lat (np.ndarray): shape (H,).
        lon (np.ndarray): shape (W,).
        n_species_to_plot (int): Number of channels (species) to plot.
        region_extent (list): [lon_min, lon_max, lat_min, lat_max].
        out_dir (Path): Directory for saving plots.
        save (bool): Whether to save the figure.
    """
    for species_i in range(n_species_to_plot):
        fig, axes = plt.subplots(
            1, 3,
            figsize=(18, 6),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )

        # shape [H, W]
        t0_vals = t0_species[species_i, :, :].cpu().numpy()
        target_vals = target_species[species_i, :, :].cpu().numpy()
        if prediction_species is not None:
            prediction_vals = prediction_species[species_i, :, :].cpu().numpy()

        create_subfig(
            fig=fig, ax=axes[0],
            lat=lat, lon=lon, matrix=t0_vals,
            title=f"Species {species_i}: T0 = {times[0]}",
            label="Value",
            region_extent=region_extent
        )
        create_subfig(
            fig=fig, ax=axes[1],
            lat=lat, lon=lon, matrix=target_vals,
            title=f"Species {species_i}: Target = {times[1]}",
            label="Value",
            region_extent=region_extent
        )
        if prediction_species is not None:
            create_subfig(
                fig=fig, ax=axes[2],
                lat=lat, lon=lon, matrix=prediction_vals,
                title=f"Species {species_i}: Prediction = {times[1]}",
                label="Value",
                region_extent=region_extent,
                roll_back=False
            )

        plt.tight_layout()
        if save:
            filename = out_dir / f"eval_species_{species_i}_{times[0]}-{times[1]}.jpeg"
            plt.savefig(filename, dpi=300, bbox_inches="tight")

        plt.show()
        plt.close(fig)
