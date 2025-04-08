import math
from pathlib import Path
from typing import List

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from aurora.batch import Batch
from plotly.subplots import make_subplots


def plot_df_latlon(
    df: pd.DataFrame,
    title: str,
    lat_key: str = "lat",
    lon_key: str = "lon",
    value_keys: List[str] = ["value"],
    radius: int = 10,
):
    min_lat = df[lat_key].min()
    max_lat = df[lat_key].max()
    min_lon = df[lon_key].min()
    max_lon = df[lon_key].max()
    print("lat", min_lat, max_lat)
    print("lon", min_lon, max_lon)
    # fig = go.Figure()
    columns = math.ceil(math.sqrt(len(value_keys)))
    rows = math.ceil(len(value_keys) / columns)
    fig = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=value_keys,
        specs=[[{"type": "scattergeo"}] * columns] * rows,
    )
    max_value = max(df[value_keys].max().tolist())
    for i, value_key in enumerate(value_keys):
        df_filtered = df[df[value_key] > 0.0]
        fig.add_trace(
            # go.Densitymap(
            #     lat=df[lat_key],
            #     lon=df[lon_key],
            #     z=df[value_key],
            #     radius=radius,
            #     name=value_key,
            # )
            go.Scattergeo(
                lat=df_filtered[lat_key],
                lon=df_filtered[lon_key],
                text=df_filtered[value_key],
                # marker_color=df_filtered[value_key],
                # radius=radius,
                marker=dict(
                    size=8,
                    opacity=df_filtered[value_key] / max_value,
                ),
                name=value_key,
            ),
            col=i % columns + 1,
            row=math.floor(i / columns) + 1,
        )
    # for c in range(columns)
    # for c in range(columns)
    geo = dict(
        scope="europe",
        resolution=50,
        lonaxis=dict(range=[min_lon, max_lon]),
        lataxis=dict(range=[min_lat, max_lat]),
    )
    geo_dict = {f"geo{i+1}": geo for i in range(len(value_keys))}
    # fig.update_layout(  # TODO: only works for the first subplot???
    #     #     # map_style="open-street-map",
    #     #     map_center_lon=(min_lon + max_lon) / 2,
    #     #     map_center_lat=(min_lat + max_lat) / 2,
    #     #     map_bounds_north=max_lat,
    #     #     map_bounds_south=min_lat,
    #     #     map_bounds_east=max_lon,
    #     #     map_bounds_west=min_lon,
    #     # geo=geo,
    # )
    fig.update_layout(geo_dict)
    fig.update_layout(title=title)
    # fig.update_layout(
    #     margin={"r": 0, "t": 0, "l": 0, "b": 0},
    #     # width=(max_lon - min_lon) * 20,
    #     # height=(max_lat - min_lat) * 20 + 400,
    # )
    # fig.show()
    return fig


def plot_eval(
    batch: Batch,
    prediction_species: torch.Tensor,
    out_dir: Path,
    n_species_to_plot: int = 20,
    save: str = True
):
    metadata = batch.metadata
    lat = metadata.lat.cpu().numpy()
    lon = metadata.lon.cpu().numpy()
    extent = {
        "lat": lat,
        "lon": lon,
    }
    time = metadata.time # tuple with shape [B, T]
    # print("time", len(time))
    species_distribution = batch.surf_vars["species_distribution"]
    # print("species_distribution", species_distribution.shape) # [B, T=2, S, H, W]
    t0_species = species_distribution[:, 0, :, :, :] # [B, S, H, W]
    target_species = species_distribution[:, 1, :, :, :]
    # print("prediction", prediction_species.shape) # [B, T=1, S, H, W]
    prediction_species = prediction_species[:, 0, :, :, :]

    lon_above_loc = np.where(lon > 180)[0]
    count_above = len(lon_above_loc)
    # print("count_above", count_above)
    for single_b in range(prediction_species.shape[0]):
        plot_single(
            t0_species[single_b],
            target_species[single_b],
            prediction_species[single_b],
            times=time[single_b],
            extent=extent,
            n_species_to_plot=n_species_to_plot,
            count_above=count_above,
            out_dir=out_dir,
            save=save,
        )
    # columns = 3
    # rows = n_species_to_plot
    # fig = make_subplots(
    #     rows=rows,
    #     cols=columns,
    #     subplot_titles=[
    #         f"T0: {time[0]}",
    #         f"T1 (target): {time[1]}",
    #         f"T1 (prediction): {time[1]}",
    #     ],
    #     specs=[[{"type": "scattergeo"}] * columns] * rows,
    # )
    # for species_i in range(n_species_to_plot):
    #     t0_vals = t0_species[species_i, :, :].cpu().numpy()
    #     fig.add_trace(
    #         go.Scattergeo(
    #             lat=df_filtered[lat_key],
    #             lon=df_filtered[lon_key],
    #             text=df_filtered[value_key],
    #             # marker_color=df_filtered[value_key],
    #             # radius=radius,
    #             marker=dict(
    #                 size=8,
    #                 opacity=df_filtered[value_key] / max_value,
    #             ),
    #             name=value_key,
    #         ),
    #         col=i % columns + 1,
    #         row=math.floor(i / columns) + 1,
    #     )
    # geo = dict(
    #     scope="europe",
    #     resolution=50,
    #     lonaxis=dict(range=[min_lon, max_lon]),
    #     lataxis=dict(range=[min_lat, max_lat]),
    # )
    # geo_dict = {f"geo{i+1}": geo for i in range(columns * rows)}

    # fig.update_layout(geo_dict)
    # fig.update_layout(title="Predictions")

def create_subfig(fig, ax, extent, matrix, title, label="Value"):
    extent_array = extent["extent_array"]
    lat = extent["lat"]
    lon = extent["lon"]
    # print("extent_array", extent_array)
    # europe_extent = [-30, 40, 34.25, 72]
    Lon, Lat = np.meshgrid(lon, np.sort(lat), indexing="xy")
    ax.set_extent(extent_array, crs=ccrs.PlateCarree())
    try:
        ax.coastlines(resolution="50m")
    except Exception as e:
        print("Error drawing coastlines on Timestep 2:", e)
    cf2 = ax.contourf(
        Lon, Lat, matrix, levels=60, cmap="viridis", transform=ccrs.PlateCarree()
    )
    ax.set_title(title)
    fig.colorbar(cf2, ax=ax, orientation="vertical", label=label)


def plot_single(t0_species, target_species, prediction_species, times, extent, n_species_to_plot: int, count_above: int, out_dir: Path, save: str):
    # europe_extent = [-30, 40, 34.25, 72]
    # lat_fixed = np.linspace(72, 34.25, 152)
    # lat_fixed = np.sort(lat_fixed)  # Now from 34.25 to 72.0.
    # Longitude: 320 points linearly spaced from -30.0 to 40.0.
    # lon_fixed = np.linspace(-30, 40, 320)
    # Create a meshgrid.
    # Lon, Lat = np.meshgrid(lon_fixed, lat_fixed, indexing="xy")
    
    for species_i in range(n_species_to_plot):
        fig, axes = plt.subplots(
            1, 3, figsize=(18, 6), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        t0_vals = t0_species[species_i, :, :].cpu().numpy()
        target = target_species[species_i, :, :].cpu().numpy()
        prediction = prediction_species[species_i, :, :].cpu().numpy()
        # roll
        # now put lon+360
        lat = extent["lat"].copy()
        lon = extent["lon"].copy()
        # print(lon)
        lon_above_loc = np.where(lon > 180)[0]
        count_above = len(lon_above_loc)
        lon[-count_above:] -= 360
        # lon = np.sort(lon)
        lon = np.roll(lon, shift=+(count_above))
        # print("LON NOW", lon)
        extent_array = [
            lon.min(),
            lon.max(),
            lat.min(),
            lat.max(),
        ]
        new_extent = {
            "extent_array": extent_array,
            "lat": lat,
            "lon": lon
        }

        # roll the [180,360) portion to [-180, 0)
        t0_vals = np.roll(t0_vals, shift=+(count_above), axis=1)
        target = np.roll(target, shift=+(count_above), axis=1)
        prediction = np.roll(prediction, shift=+(count_above), axis=1)
        # and filp the lat axis (y increasing with lat decreasing)
        t0_vals = np.flip(t0_vals, axis=0)
        target = np.flip(target, axis=0)
        prediction = np.flip(prediction, axis=0)
        # print(t0_vals.shape, target.shape, prediction.shape)

        # subfig 1
        create_subfig(fig=fig, ax = axes[0], extent=new_extent, matrix=t0_vals, title=f"species {species_i}: T0={times[0]}")

        # subfig 2
        create_subfig(fig=fig, ax = axes[1], extent=new_extent, matrix=target, title=f"species {species_i}: target ={times[1]}")

        # subfig 3
        create_subfig(fig=fig, ax = axes[2], extent=new_extent, matrix=prediction, title=f"species {species_i}: prediction ={times[1]}")

        plt.tight_layout()
        fig.canvas.draw()
        if save:
            filename = out_dir / f"eval_species_{species_i}_{times[0]}-{times[1]}.jpeg"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
