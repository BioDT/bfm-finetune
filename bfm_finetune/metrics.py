import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import scipy.ndimage
import math
import matplotlib.pyplot as plt

def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes RMSE over the entire tensor.
    Both pred and target are expected to have shape [B, 1, C, H, W].
    """
    return torch.sqrt(torch.mean((pred - target) ** 2)) # .item() if want the value

def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes MAE over the entire tensor.
    """
    return torch.mean(torch.abs(pred - target))

def compute_mape(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> float:
    """
    Computes MAPE (in percentage) over the entire tensor.
    """
    return (torch.mean(torch.abs((pred - target) / (target + epsilon))) * 100)

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

def compute_msss(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> float:
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

def compute_ssim_metric(pred: torch.Tensor, target: torch.Tensor, data_range: float = None) -> float:
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
            dr = data_range if data_range is not None else (pred_np[b, c].max() - pred_np[b, c].min())
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

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
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
                psnr_vals.append(float('inf'))
            else:
                psnr_vals.append(20 * np.log10(max_val) - 10 * np.log10(mse))
    return np.mean(psnr_vals)

def compute_fss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 3) -> float:
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
            pred_fraction = scipy.ndimage.uniform_filter(pred_field.astype(np.float32), size=window_size)
            target_fraction = scipy.ndimage.uniform_filter(target_field.astype(np.float32), size=window_size)
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
        "SSIM": [ssim_val, ssim_val * 0.98, ssim_val, ssim_val * 1.0]
    }

    print("Plotting metrics...")
    plot_metrics(dummy_metrics)
