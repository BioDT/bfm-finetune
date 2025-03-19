import math
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
