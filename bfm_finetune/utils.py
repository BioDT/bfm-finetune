import numpy as np
import pandas as pd
from tqdm import tqdm


def get_lat_lon_ranges(
    min_lon: float = -30.0,
    max_lon: float = 50.0,
    min_lat: float = 34.0,
    max_lat: float = 72.0,
    lon_step: float = 0.25,
    lat_step: float = 0.25,
):
    """
    Get latitude and longitude ranges.

    Args:
        min_lon (float): The minimum longitude.
        max_lon (float): The maximum longitude.
        min_lat (float): The minimum latitude.
        max_lat (float): The maximum latitude.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The latitude and longitude ranges.
    """

    lat_range = np.arange(min_lat, max_lat + lat_step, lat_step)
    lon_range = np.arange(min_lon, max_lon + lon_step, lon_step)
    # reverse lat_range to go from North to South
    lat_range = lat_range[::-1]

    return lat_range, lon_range


def aggregate_into_latlon_grid(
    df: pd.DataFrame, lat_range: np.ndarray, lon_range: np.ndarray, step: float
) -> np.ndarray:
    matrix = np.zeros((len(lat_range), len(lon_range)))
    # slower loop
    # for lat_i, lat_val in enumerate(tqdm(lat_range)):
    #     for lon_i, lon_val in enumerate(lon_range):
    #         df_here = df[
    #             (df["lat"] >= lat_val - step / 2)
    #             & (df["lat"] < lat_val + step / 2)
    #             & (df["lon"] >= lon_val - step / 2)
    #             & (df["lon"] < lon_val + step / 2)
    #         ]
    #         # print(len(df_here))
    #         matrix[lat_i, lon_i] = len(df_here)
    lat_range_list = lat_range.tolist()
    lon_range_list = lon_range.tolist()
    # faster loop
    for index, row in tqdm(df.iterrows()):
        lat = row["lat"]
        lon = row["lon"]
        lat_i = next(i for i, val in enumerate(lat_range_list) if val <= lat + step / 2)
        lon_i = next(i for i, val in enumerate(lon_range_list) if val >= lon - step / 2)
        # print(lat_i, lat, lat_range)
        # raise ValueError(123)
        matrix[lat_i, lon_i] += 1.0

    return matrix


def unroll_matrix_into_df(lat_range, lon_range, matrix: np.ndarray):
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
    return df
