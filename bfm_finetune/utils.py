import os
import numpy as np
import pandas as pd
import torch

def get_lat_lon_ranges(
    min_lon: float = -30.0,
    max_lon: float = 50.0,
    min_lat: float = 34.0,
    max_lat: float = 72.0,
    lon_step: float = 0.25,
    lat_step: float = 0.25,
    crop_multiple: int = 4,
    lon_positive: bool = False,
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
    # make it compatible with patch_size
    if crop_multiple != 1:
        lat_cnt = lat_range.shape[0] // crop_multiple * crop_multiple
        lat_range = lat_range[:lat_cnt]
        lon_cnt = lon_range.shape[0] // crop_multiple * crop_multiple
        lon_range = lon_range[:lon_cnt]

    if lon_positive:
        # negative longitudes are +360 and rotated at the end to have sorted lon
        lon_range[lon_range < 0] += 360.0
        lon_range.sort()

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
    for index, row in df.iterrows():
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


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1} with loss {loss:.4f}")

def load_checkpoint(model, optimizer, checkpoint_path="checkpoint.pth"):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["loss"]
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {start_epoch}, loss: {best_loss:.4f})")
        return start_epoch, best_loss
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, float("inf")


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True