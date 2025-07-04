import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
from properscoring import crps_ensemble
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import f1_score

EPS = 1e-8


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes RMSE over the entire tensor.
    Both pred and target are expected to have shape [B, 1, C, H, W].
    """
    return torch.sqrt(torch.mean((pred - target) ** 2))  # .item() if want the value


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes MAE over the entire tensor.
    """
    return torch.mean(torch.abs(pred - target))


def compute_mape(
    pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8
) -> float:
    """
    Computes MAPE (in percentage) over the entire tensor.
    """
    return torch.mean(torch.abs((pred - target) / (target + epsilon))) * 100


def compute_acc(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes the Anomaly Correlation Coefficient (ACC) per sample.
    For each sample, the anomaly is computed by subtracting the sample mean
    (computed over C, H, W) from both pred and target.
    Returns the average Pearson correlation over the batch.

    Both pred and target have shape [B, 1, C, H, W].
    """
    B = pred.shape[0]
    correlations = []
    for b in range(B):
        # Squeeze the time dimension; shape: [C, H, W]
        pred_sample = pred[b, 0]
        target_sample = target[b, 0]
        pred_flat = pred_sample.flatten()
        target_flat = target_sample.flatten()
        mean_pred = pred_flat.mean()
        mean_target = target_flat.mean()
        std_pred = pred_flat.std()
        std_target = target_flat.std()
        if std_pred.item() == 0 or std_target.item() == 0:
            correlations.append(0)
        else:
            cov = torch.mean((pred_flat - mean_pred) * (target_flat - mean_target))
            corr = (cov / (std_pred * std_target)).item()
            correlations.append(corr)
    return np.mean(correlations)


def compute_msss(
    pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8
) -> float:
    """
    Computes the Mean Squared Skill Score (MSSS) relative to a baseline.
    For each sample, the baseline is the climatology (i.e. the sample mean over C, H, W).
    MSSS = 1 - (MSE_model / MSE_baseline)

    Both pred and target have shape [B, 1, C, H, W].
    Returns the average MSSS over the batch.
    """
    B = pred.shape[0]
    msss_vals = []
    for b in range(B):
        pred_sample = pred[b, 0]
        target_sample = target[b, 0]
        mse_model = torch.mean((pred_sample - target_sample) ** 2)
        climatology = target_sample.mean()
        mse_baseline = torch.mean((target_sample - climatology) ** 2)
        msss_vals.append(1 - (mse_model / (mse_baseline + epsilon)).item())
    return np.mean(msss_vals)


def compute_ssim_metric(
    pred: torch.Tensor, target: torch.Tensor, data_range: float = None
) -> float:
    """
    Computes the Structural Similarity Index (SSIM) for spatial fields.
    For each sample and channel, SSIM is computed on the 2D field (H, W).
    Both pred and target have shape [B, 1, C, H, W]. The time dim is squeezed.

    Returns the average SSIM over all samples and channels.
    """
    B, _, C, H, W = pred.shape
    ssim_vals = []
    pred_np = pred.squeeze(1).detach().cpu().numpy()  # shape: [B, C, H, W]
    target_np = target.squeeze(1).detach().cpu().numpy()  # shape: [B, C, H, W]
    for b in range(B):
        for c in range(C):
            # Determine data_range if not provided
            dr = (
                data_range
                if data_range is not None
                else (pred_np[b, c].max() - pred_np[b, c].min())
            )
            ssim_val = ssim(pred_np[b, c], target_np[b, c], data_range=dr)
            ssim_vals.append(ssim_val)
    return np.mean(ssim_vals)


def compute_spc(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes the Spatial Pattern Correlation (SPC).
    For each sample and channel, flatten the 2D spatial field and compute Pearson correlation.

    Both pred and target have shape [B, 1, C, H, W].
    Returns the average SPC over all samples and channels.
    """
    B, _, C, H, W = pred.shape
    correlations = []
    pred_np = pred.squeeze(1).detach().cpu().numpy()
    target_np = target.squeeze(1).detach().cpu().numpy()
    for b in range(B):
        for c in range(C):
            p_flat = pred_np[b, c].flatten()
            t_flat = target_np[b, c].flatten()
            if np.std(p_flat) == 0 or np.std(t_flat) == 0:
                correlations.append(0)
            else:
                corr = np.corrcoef(p_flat, t_flat)[0, 1]
                correlations.append(corr)
    return np.mean(correlations)


def compute_psnr(
    pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0
) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) in dB.
    For each sample and channel, PSNR is computed and then averaged.

    Both pred and target have shape [B, 1, C, H, W].
    """
    B, _, C, H, W = pred.shape
    psnr_vals = []
    pred_np = pred.squeeze(1).detach().cpu().numpy()
    target_np = target.squeeze(1).detach().cpu().numpy()
    for b in range(B):
        for c in range(C):
            mse = np.mean((pred_np[b, c] - target_np[b, c]) ** 2)
            if mse == 0:
                psnr_vals.append(float("inf"))
            else:
                psnr_vals.append(20 * np.log10(max_val) - 10 * np.log10(mse))
    return np.mean(psnr_vals)


def compute_fss(
    pred: torch.Tensor, target: torch.Tensor, window_size: int = 3
) -> float:
    """
    Computes the Fractions Skill Score (FSS) for spatial fields.
    For each sample and channel, a uniform filter is applied to compute the fractional coverage.

    Both pred and target have shape [B, 1, C, H, W].
    Returns the average FSS over all samples and channels.
    """
    B, _, C, H, W = pred.shape
    fss_vals = []
    pred_np = pred.squeeze(1).detach().cpu().numpy()  # [B, C, H, W]
    target_np = target.squeeze(1).detach().cpu().numpy()  # [B, C, H, W]
    for b in range(B):
        for c in range(C):
            pred_field = pred_np[b, c]
            target_field = target_np[b, c]
            pred_fraction = scipy.ndimage.uniform_filter(
                pred_field.astype(np.float32), size=window_size
            )
            target_fraction = scipy.ndimage.uniform_filter(
                target_field.astype(np.float32), size=window_size
            )
            mse = np.mean((pred_fraction - target_fraction) ** 2)
            denom = np.mean((np.abs(pred_fraction) + np.abs(target_fraction)) ** 2)
            fss_vals.append(1 - mse / (denom + 1e-8))
    return np.mean(fss_vals)


def plot_metrics(metrics_dict: dict):
    """
    Plots a dictionary of metric values over epochs.

    Args:
        metrics_dict (dict): Keys are metric names; values are lists of metric values.
    """
    plt.figure(figsize=(12, 8))
    for metric_name, values in metrics_dict.items():
        plt.plot(values, label=metric_name)
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Validation Metrics Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


## TESTING SOME FUNCTIONS


def create_constant_tensor(value, shape=(2, 1, 3, 32, 32)):
    """Creates a tensor filled with 'value' of shape [B, T=1, C, H, W]."""
    return torch.full(shape, value, dtype=torch.float32)


def generate_synthetic_data(B=3, T=1, C=1, H=64, W=64, noise_std=0.1, shift=2):
    """
    Generates synthetic spatial forecast data.
    The target is defined as a sinusoidal pattern over a spatial grid.
    The prediction is defined as the target shifted to the right by 'shift' pixels
    plus some additive Gaussian noise.

    Returns:
        target, pred: torch.Tensor of shape [B, T, C, H, W]
    """
    x = np.linspace(0, 2 * np.pi, W)
    y = np.linspace(0, 2 * np.pi, H)
    X, Y = np.meshgrid(x, y)
    base_pattern = np.sin(X) * np.cos(Y)  # shape: [H, W]

    # Create target tensor: same pattern repeated for each sample.
    target = np.tile(base_pattern, (B, T, C, 1, 1)).astype(np.float32)

    # Create prediction: shift the pattern to the right and add noise.
    pred = np.empty_like(target)
    for b in range(B):
        for t in range(T):
            for c in range(C):
                shifted = np.roll(base_pattern, shift, axis=1)  # shift horizontally
                noisy = shifted + np.random.normal(0, noise_std, size=shifted.shape)
                pred[b, t, c] = noisy

    return torch.tensor(target), torch.tensor(pred)


# CRPS – Monte‑Carlo version that works for any Poisson λ
def _sampled_crps_poisson(
    lam: torch.Tensor, obs: torch.Tensor, n_samples: int = 30
) -> torch.Tensor:
    """
    Draw Monte-Carlo samples from Poisson(λ) and call properscoring.
    Shapes are broadcast automatically; runs on GPU then moves to CPU
    only for the final scoring step to keep memory low.
    """
    lam = lam.clamp(min=EPS)  # avoid negative
    obs = obs.clamp(min=0)

    # (..., n_samples) synthetic ensemble
    samples = torch.poisson(lam.unsqueeze(-1).expand(*lam.shape, n_samples))

    ens = samples.reshape(-1, n_samples).cpu().numpy()
    obs1d = obs.reshape(-1).cpu().numpy()
    return torch.as_tensor(crps_ensemble(obs1d, ens).mean())


def crps_(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    TODO FIX ITS TOO SLOW
    pred can be
      - deterministic mean λ  [B, S, H, W]   -> Monte-Carlo CRPS
      - ensemble samples      [B, S, H, W, E] -> exact ensemble CRPS
    """
    if pred.ndim == 5:  # for ensamble
        ens = pred.permute(4, 0, 1, 2, 3).reshape(pred.shape[-1], -1).cpu()
        obs = target.reshape(-1).cpu()
        return torch.as_tensor(crps_ensemble(obs.numpy(), ens.numpy()).mean())
    else:  # draw synthetic ensemble
        return _sampled_crps_poisson(pred, target)


def crps(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Deterministic CRPS (≡ MAE). Works on any tensor shape.
    Reduces over all dims: batch, species, lat, lon.
    """
    return torch.mean(torch.abs(pred - target))


def _sanitize(t: torch.Tensor) -> torch.Tensor:
    """Ensure finite, non-negative values before taking logs."""
    t = torch.nan_to_num(t, nan=0.0, neginf=0.0, posinf=0.0)
    return t.clamp(min=EPS)


def poisson_deviance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred, target = _sanitize(pred), _sanitize(target)
    term = target * torch.log((target + EPS) / pred) - (target - pred)
    return 2.0 * torch.mean(term)


def explained_deviance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dev_model = poisson_deviance(pred, target)
    # “null” model = one global mean rate over the whole spatiotemporal cube
    # TODO Adapt the global model to something meaningful
    mean_rate = torch.mean(_sanitize(target))
    dev_null = poisson_deviance(mean_rate.expand_as(target), target)
    return 1.0 - dev_model / (dev_null + EPS)


# True‑Skill Statistic
def tss(pred: torch.Tensor, target: torch.Tensor, threshold: int = 1) -> torch.Tensor:

    pred_bin = _sanitize(pred) >= threshold
    targ_bin = _sanitize(target) >= threshold
    tp = (pred_bin & targ_bin).sum()
    tn = (~pred_bin & ~targ_bin).sum()
    fp = (pred_bin & ~targ_bin).sum()
    fn = (~pred_bin & targ_bin).sum()
    sens = tp / (tp + fn + EPS)
    spec = tn / (tn + fp + EPS)
    return sens + spec - 1.0


def make_test():
    # Test data: Two samples, 1 timestep, 3 channels, 32x32 spatial dimensions.
    B, T, C, H, W = 2, 1, 3, 32, 32

    # Let target be zeros and pred be ones. Expected RMSE = 1, MAE = 1.
    target_zeros = create_constant_tensor(0, shape=(B, T, C, H, W))
    pred_ones = create_constant_tensor(1, shape=(B, T, C, H, W))
    rmse_val = compute_rmse(pred_ones, target_zeros)
    mae_val = compute_mae(pred_ones, target_zeros)
    print("Test RMSE (expected 1):", rmse_val)
    print("Test MAE (expected 1):", mae_val)

    # Let target be ones and pred be zeros, so absolute percentage error = 100%.
    target_ones = create_constant_tensor(1, shape=(B, T, C, H, W))
    pred_zeros = create_constant_tensor(0, shape=(B, T, C, H, W))
    mape_val = compute_mape(pred_zeros, target_ones)
    print("Test MAPE (expected 100):", mape_val)

    # For identical tensors, ACC should be 1.
    acc_val = compute_acc(target_ones, target_ones)
    print("Test ACC (expected 1):", acc_val)

    # For perfect forecast (identical to target), MSSS = 1.
    msss_val = compute_msss(target_ones, target_ones)
    print("Test MSSS (expected 1):", msss_val)

    # For identical images, SSIM should be 1.
    # ssim_val = compute_ssim_torch(target_ones, target_ones)
    # print("Test SSIM (expected 1):", ssim_val)

    ssim_val = compute_ssim_metric(target_ones, target_ones)
    print("Test SSIM sci-image (expected 1):", ssim_val)

    # For identical images, PSNR should be infinite.
    # psnr_val = compute_psnr_torch(target_ones, target_ones)
    # print("Test PSNR (expected inf):", psnr_val)

    psnr_val = compute_psnr(target_ones, target_ones)
    print("Test PSNR Classic (expected inf):", psnr_val)

    # For identical images, SPC should be 1.
    spc_val = compute_spc(target_ones, target_ones)
    print("Test SPC (expected 1):", spc_val)

    # For identical images, FSS should be 1.
    fss_val = compute_fss(target_ones, target_ones)
    print("Test FSS (expected 1):", fss_val)

    target_tensor, pred_tensor = generate_synthetic_data()

    # Compute Metrics on Synthetic Data
    print("Realistic Synthetic Data Metrics:")
    print("RMSE:", compute_rmse(pred_tensor, target_tensor))
    print("MAE:", compute_mae(pred_tensor, target_tensor))
    print("MAPE:", compute_mape(pred_tensor, target_tensor))
    print("ACC:", compute_acc(pred_tensor, target_tensor))
    print("MSSS:", compute_msss(pred_tensor, target_tensor))
    print("SSIM:", compute_ssim_metric(pred_tensor, target_tensor))
    print("PSNR:", compute_psnr(pred_tensor, target_tensor))
    print("SPC:", compute_spc(pred_tensor, target_tensor))
    print("FSS:", compute_fss(pred_tensor, target_tensor))

    # Create a dummy metrics dictionary with 4 epochs.
    dummy_metrics = {
        "RMSE": [rmse_val, rmse_val * 0.9, rmse_val * 1.1, rmse_val * 0.95],
        "MAE": [mae_val, mae_val * 0.95, mae_val * 1.05, mae_val],
        "SSIM": [ssim_val, ssim_val * 0.98, ssim_val, ssim_val * 1.0],
    }

    print("Plotting metrics...")
    plot_metrics(dummy_metrics)


# Define a function to compute F1 score for your predictions
def compute_f1(pred, target, threshold=0.5):
    """
    Compute F1 score for species presence/absence prediction.

    Args:
        pred: Tensor of predictions [B, 1, C, H, W]
        target: Tensor of ground truth [B, 1, C, H, W]
        threshold: Threshold to convert predictions to binary (default: 0.5)

        Returns:
        F1 score as a float
    """
    # Convert tensors to numpy arrays
    pred_np = pred.squeeze(1).detach().cpu().numpy()  # shape: [B, C, H, W]
    target_np = target.squeeze(1).detach().cpu().numpy()  # shape: [B, C, H, W]

    # Flatten all dimensions (sample, channel, height, width)
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()

    # Convert predictions to binary using threshold
    pred_binary = (pred_flat > threshold).astype(np.int32)
    target_binary = (target_flat > 0).astype(np.int32)

    # Calculate F1 score
    f1 = f1_score(target_binary, pred_binary, zero_division=0)
    return f1


def compute_geolifeclef_f1(pred, target, threshold=0.5):
    """
    Compute F1 score using the formula:
    F1 = (1/N) ∑ TP_i / (TP_i + (FP_i + FN_i)/2)

    Overview of GeoLifeCLEF 2024: Species Composition
    Prediction with High Spatial Resolution at Continental
    Scale using Remote Sensing
    From: https://ceur-ws.org/Vol-3740/paper-186.pdf

    where:
    - TP_i = number of correctly predicted species
    - FP_i = number of species predicted but not observed
    - FN_i = number of species not predicted but present

    Args:
        pred: Tensor of predictions [B, 1, C, H, W]
        target: Tensor of ground truth [B, 1, C, H, W]
        threshold: Threshold to convert predictions to binary (default: 0.5)

    Returns:
        Custom F1 score as a float
    """
    # Convert tensors to numpy arrays
    pred_np = pred.squeeze(1).detach().cpu().numpy()  # shape: [B, C, H, W]
    target_np = target.squeeze(1).detach().cpu().numpy()  # shape: [B, C, H, W]

    B, C, H, W = pred_np.shape
    f1_scores = []

    # Process each sample in the batch
    for b in range(B):
        sample_f1_scores = []

        # For each location (H, W coordinates)
        for h in range(H):
            for w in range(W):
                # Get predicted and target species at this location
                pred_species = pred_np[b, :, h, w]
                target_species = target_np[b, :, h, w]

                # Binarize predictions based on threshold
                pred_binary = (pred_species > threshold).astype(int)
                target_binary = (target_species > 0).astype(int)

                # Calculate TP, FP, FN
                TP = np.sum(pred_binary & target_binary)
                FP = np.sum(pred_binary & ~target_binary)
                FN = np.sum(~pred_binary & target_binary)

                # Calculate F1 score for this location
                if TP + FP + FN == 0:  # No species predicted or observed
                    location_f1 = (
                        1.0  # Perfect score when both pred and target are empty
                    )
                else:
                    location_f1 = TP / (TP + (FP + FN) / 2) if TP > 0 else 0

                sample_f1_scores.append(location_f1)

        # Average F1 over all locations in this sample
        if sample_f1_scores:
            f1_scores.append(np.mean(sample_f1_scores))

    # Return average F1 over all samples
    return np.mean(f1_scores) if f1_scores else 0.0
