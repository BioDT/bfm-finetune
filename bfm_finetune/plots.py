import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_matrix(lat_range, lon_range, matrix: np.ndarray):
    assert matrix.shape[0] == len(
        lat_range
    ), f"lat_range and matrix.shape[0] mismatch: {len(lat_range)}, {matrix.shape[0]}"
    assert matrix.shape[1] == len(
        lon_range
    ), f"lon_range and matrix.shape[1] mismatch: {len(lon_range)}, {matrix.shape[1]}"
    unrolled = []
    for lat_i, lat_val in enumerate(lat_range):
        for lon_i, lon_val in enumerate(lon_range):
            unrolled.append(
                {"lat": lat_val, "lon": lon_val, "value": matrix[lat_i, lon_i]}
            )
    df = pd.DataFrame(unrolled)
    return plot_df_latlon(df)


def plot_df_latlon(
    df: pd.DataFrame,
    lat_key: str = "lat",
    lon_key: str = "lon",
    value_key: str = "value",
    radius: int = 10,
):
    min_lat = df[lat_key].min()
    max_lat = df[lat_key].max()
    min_lon = df[lon_key].min()
    max_lon = df[lon_key].max()
    print("lat", min_lat, max_lat)
    print("lon", min_lon, max_lon)
    fig = go.Figure(
        go.Densitymap(lat=df[lat_key], lon=df[lon_key], z=df[value_key], radius=radius)
    )
    fig.update_layout(
        map_style="open-street-map",
        map_center_lon=(min_lon + max_lon) / 2,
        map_center_lat=(min_lat + max_lat) / 2,
        map_bounds_north=max_lat,
        map_bounds_south=min_lat,
        map_bounds_east=max_lon,
        map_bounds_west=min_lon,
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        # width=(max_lon - min_lon) * 20,
        # height=(max_lat - min_lat) * 20 + 400,
    )
    # fig.show()
    return fig
