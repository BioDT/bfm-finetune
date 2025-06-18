import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score


def get_lat_lon_ranges(
    min_lon: float = -25.0,
    max_lon: float = 45.0,
    min_lat: float = 32.0,
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

    # copy avoids negative strides (not supported by torch.Tensor)
    return lat_range.copy(), lon_range.copy()


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
        lat_i = next(
            (
                i
                for i, val in enumerate(lat_range_list)
                if val < lat + step / 2 and val >= lat - step / 2
            ),
            None,
        )
        lon_i = next(
            (
                i
                for i, val in enumerate(lon_range_list)
                if val < lon + step / 2 and val >= lon - step / 2
            ),
            None,
        )
        if lat_i == None or lon_i == None:
            # not in the grid
            continue
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


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_folder):
    if not checkpoint_folder:
        print("checkpoint_folder not set. not saving.")
        return
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    file_path = os.path.join(checkpoint_folder, "best_checkpoint.pth")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at epoch {epoch+1} with loss {loss:.4f} to {file_path}")


def get_supersampling_target_lat_lon(
    supersampling_config,
) -> Tuple[np.ndarray, np.ndarray] | None:
    if supersampling_config == None:
        return None
    if supersampling_config.enabled:
        return get_lat_lon_ranges(
            min_lon=supersampling_config.target_region.min_lon,
            max_lon=supersampling_config.target_region.max_lon,
            min_lat=supersampling_config.target_region.min_lat,
            max_lat=supersampling_config.target_region.max_lat,
        )
    else:
        return None


def load_checkpoint(
    model, optimizer, checkpoint_folder, strict=True, load_optim_state=False
):
    if not checkpoint_folder:
        print("checkpoint_folder not set. Starting from scratch.")
        return 0, float("inf")
    file_path = os.path.join(checkpoint_folder, "best_checkpoint.pth")
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if load_optim_state:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["loss"]
        print(
            f"Loaded checkpoint from {file_path} (epoch {start_epoch}, loss: {best_loss:.4f})"
        )
        return start_epoch, best_loss
    else:
        print("No checkpoint found in folder. Starting from scratch.")
        return 0, float("inf")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_config(output_dir: str | Path, config_file_name: str = "config.yaml"):
    config_path = Path(output_dir) / ".hydra"
    # with initialize(version_base=None, config_path=str(config_path), job_name="test_app"):
    #     cfg = compose(config_name=config_file_name, overrides=[])
    #     return cfg
    cfg = OmegaConf.load(str(config_path / config_file_name))
    return cfg


######## FOR CLASSIFICATION - Like MALPOLON DOES https://arxiv.org/pdf/2409.18102


def compute_top25_metrics_for_sample(pred, target):
    """
    Computes Top-25 metrics for one sample.

    Args:
        pred (np.array): 1D array of predictions.
        target (np.array): 1D array of target values.

    Returns:
        dict: Contains precision, recall, f1, auc and the intersection count.
    """
    # Determine top-25 indices for predictions and targets.
    top25_pred_idx = np.argsort(pred)[-25:]
    top25_target_idx = np.argsort(target)[-25:]
    pred_set = set(top25_pred_idx)
    target_set = set(top25_target_idx)
    intersection = len(pred_set.intersection(target_set))

    precision = intersection / 25.0
    recall = intersection / 25.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    # For AUC: create binary labels (1 for top-25 indices, 0 otherwise).
    binary_labels = np.zeros_like(target, dtype=int)
    binary_labels[top25_target_idx] = 1
    try:
        auc = roc_auc_score(binary_labels, pred)
    except ValueError:
        auc = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "intersection": intersection,
    }


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """
    Performs one training epoch and computes additional metrics.

    Returns:
        avg_loss (float): Average loss for the epoch.
        metrics (dict): Dictionary with sample (macro) and micro averaged metrics.
    """
    model.train()
    total_loss = 0.0
    all_metrics = []
    global_true = []
    global_scores = []

    for batch in dataloader:
        batch = batch.to(device)  # shape: [B, T, C_in, H, W]
        optimizer.zero_grad()
        output = model(batch)  # expected shape: [B, T, C_out, H, W] with T=1
        loss = loss_fn(output, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        output_np = output.detach().cpu().numpy()  # shape: [B, 1, C_out, H, W]
        target_np = batch.detach().cpu().numpy()  # shape: [B, T, C_in, H, W]
        B = output_np.shape[0]
        for i in range(B):
            pred_sample = output_np[i, 0].flatten()  # flatten [C_out, H, W]
            target_sample = target_np[i, 0].flatten()  # flatten [C_in, H, W]
            sample_metrics = compute_top25_metrics_for_sample(
                pred_sample, target_sample
            )
            all_metrics.append(sample_metrics)
            # For micro aggregation: build a binary ground truth vector per sample.
            binary_labels = np.zeros_like(target_sample, dtype=int)
            top25_idx = np.argsort(target_sample)[-25:]
            binary_labels[top25_idx] = 1
            global_true.append(binary_labels)
            global_scores.append(pred_sample)

    avg_loss = total_loss / len(dataloader)
    # Sample (macro) averages.
    sample_precision = np.mean([m["precision"] for m in all_metrics])
    sample_recall = np.mean([m["recall"] for m in all_metrics])
    sample_f1 = np.mean([m["f1"] for m in all_metrics])
    sample_auc = np.mean([m["auc"] for m in all_metrics])

    # Micro averages: concatenate all samples.
    global_true_concat = np.concatenate(global_true)
    global_scores_concat = np.concatenate(global_scores)
    try:
        micro_auc = roc_auc_score(global_true_concat, global_scores_concat)
    except ValueError:
        micro_auc = 0.0
    total_intersection = sum([m["intersection"] for m in all_metrics])
    micro_precision = total_intersection / (25 * len(all_metrics))
    micro_recall = total_intersection / (25 * len(all_metrics))
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    metrics = {
        "sample_precision": sample_precision,
        "sample_recall": sample_recall,
        "sample_f1": sample_f1,
        "sample_auc": sample_auc,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "micro_auc": micro_auc,
    }
    return avg_loss, metrics


def validate_epoch(model, dataloader, loss_fn, device):
    """
    Performs one validation epoch (without gradient updates) and computes metrics.

    Returns:
        avg_loss (float): Average validation loss.
        metrics (dict): Dictionary with sample (macro) and micro averaged metrics.
    """
    model.eval()
    total_loss = 0.0
    all_metrics = []
    global_true = []
    global_scores = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss = loss_fn(output, batch)
            total_loss += loss.item()

            output_np = output.detach().cpu().numpy()  # shape: [B, 1, C_out, H, W]
            target_np = batch.detach().cpu().numpy()  # shape: [B, T, C_in, H, W]
            B = output_np.shape[0]
            for i in range(B):
                pred_sample = output_np[i, 0].flatten()
                target_sample = target_np[i, 0].flatten()
                sample_metrics = compute_top25_metrics_for_sample(
                    pred_sample, target_sample
                )
                all_metrics.append(sample_metrics)
                binary_labels = np.zeros_like(target_sample, dtype=int)
                top25_idx = np.argsort(target_sample)[-25:]
                binary_labels[top25_idx] = 1
                global_true.append(binary_labels)
                global_scores.append(pred_sample)

    avg_loss = total_loss / len(dataloader)
    sample_precision = np.mean([m["precision"] for m in all_metrics])
    sample_recall = np.mean([m["recall"] for m in all_metrics])
    sample_f1 = np.mean([m["f1"] for m in all_metrics])
    sample_auc = np.mean([m["auc"] for m in all_metrics])

    global_true_concat = np.concatenate(global_true)
    global_scores_concat = np.concatenate(global_scores)
    try:
        micro_auc = roc_auc_score(global_true_concat, global_scores_concat)
    except ValueError:
        micro_auc = 0.0
    total_intersection = sum([m["intersection"] for m in all_metrics])
    micro_precision = total_intersection / (25 * len(all_metrics))
    micro_recall = total_intersection / (25 * len(all_metrics))
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    metrics = {
        "sample_precision": sample_precision,
        "sample_recall": sample_recall,
        "sample_f1": sample_f1,
        "sample_auc": sample_auc,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "micro_auc": micro_auc,
    }
    return avg_loss, metrics
