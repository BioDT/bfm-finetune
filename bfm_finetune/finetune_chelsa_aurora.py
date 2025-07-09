import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanSquaredError, R2Score
from tqdm import tqdm


class CHELSAAuroraDataset(Dataset):
    """Dataset for CHELSA target vs Aurora predictions comparison."""

    def __init__(self, netcdf_path):
        """
        Initialize dataset from the comparison NetCDF file.

        Args:
            netcdf_path: Path to the NetCDF file with chelsa_tas and aurora_2t
        """
        self.ds = xr.open_dataset(netcdf_path)

        # Extract CHELSA target data (tas)
        self.chelsa_targets = self.ds[
            "chelsa_tas"
        ].values  # Shape: (time, target_pixel)

        # Extract Aurora predictions - interpolate to match CHELSA if needed
        aurora_data = self.ds["aurora_2t"]  # Shape: (time, lat, lon)

        # Check if we need to align time dimensions
        if "aurora_time" in aurora_data.dims:
            # Different time dimensions - need to align
            common_times = np.intersect1d(
                self.ds.chelsa_tas.time.values, aurora_data.aurora_time.values
            )
            if len(common_times) > 0:
                self.chelsa_targets = (
                    self.ds["chelsa_tas"].sel(time=common_times).values
                )
                self.aurora_preds = aurora_data.sel(aurora_time=common_times).values
            else:
                # No common times - use first N samples
                min_len = min(
                    len(self.ds.chelsa_tas.time), len(aurora_data.aurora_time)
                )
                self.chelsa_targets = self.chelsa_targets[:min_len]
                self.aurora_preds = aurora_data.values[:min_len]
        else:
            # Same time dimension
            self.aurora_preds = aurora_data.values

        # Flatten Aurora predictions to match CHELSA pixel format
        if self.aurora_preds.ndim == 3:  # (time, lat, lon)
            self.aurora_preds = self.aurora_preds.reshape(
                self.aurora_preds.shape[0], -1
            )

        # Ensure same number of samples
        min_samples = min(len(self.chelsa_targets), len(self.aurora_preds))
        self.chelsa_targets = self.chelsa_targets[:min_samples]
        self.aurora_preds = self.aurora_preds[:min_samples]

        # Pre-normalize both datasets to handle unit mismatches
        print("Normalizing datasets...")

        # Normalize CHELSA targets (subtract mean, divide by std)
        chelsa_mean = self.chelsa_targets.mean()
        chelsa_std = self.chelsa_targets.std()
        self.chelsa_targets_norm = (self.chelsa_targets - chelsa_mean) / chelsa_std

        # Normalize Aurora predictions (subtract mean, divide by std)
        aurora_mean = self.aurora_preds.mean()
        aurora_std = self.aurora_preds.std()
        self.aurora_preds_norm = (self.aurora_preds - aurora_mean) / aurora_std

        # Store normalization stats for later use
        self.chelsa_stats = {"mean": chelsa_mean, "std": chelsa_std}
        self.aurora_stats = {"mean": aurora_mean, "std": aurora_std}

        # For this comparison, we'll use Aurora predictions as "inputs" and CHELSA as "targets"
        # This allows us to see how well we can predict CHELSA from Aurora
        self.inputs = torch.tensor(self.aurora_preds_norm, dtype=torch.float32)
        self.targets = torch.tensor(self.chelsa_targets_norm, dtype=torch.float32)

        # Input dimension is the flattened Aurora spatial dimension
        self.input_dim = self.inputs.shape[-1]

        print(f"Dataset initialized:")
        print(f"  CHELSA targets shape: {self.chelsa_targets.shape}")
        print(f"  Aurora predictions shape: {self.aurora_preds.shape}")
        print(f"  Input dimension: {self.input_dim}")
        print(f"  Number of samples: {len(self.inputs)}")
        print(f"  CHELSA stats - mean: {chelsa_mean:.2f}, std: {chelsa_std:.2f}")
        print(f"  Aurora stats - mean: {aurora_mean:.2f}, std: {aurora_std:.2f}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def get_original_data(self, idx):
        """Get original unnormalized data for analysis."""
        return self.aurora_preds[idx], self.chelsa_targets[idx]

    def denormalize_chelsa(self, normalized_data):
        """Convert normalized CHELSA data back to original units."""
        return normalized_data * self.chelsa_stats["std"] + self.chelsa_stats["mean"]

    def denormalize_aurora(self, normalized_data):
        """Convert normalized Aurora data back to original units."""
        return normalized_data * self.aurora_stats["std"] + self.aurora_stats["mean"]


class AuroraRegressor(nn.Module):
    """Neural network to map Aurora predictions to CHELSA targets."""

    def __init__(self, input_dim, hidden_dim=512, output_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim  # Default to same dimension as input

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.model(x)


def train_model(
    netcdf_path,
    batch_size=1,
    epochs=10,
    device="cuda",
    sanity_check=True,
    save_results=True,
):
    """Train model to predict CHELSA from Aurora predictions."""

    dataset = CHELSAAuroraDataset(netcdf_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = dataset.input_dim
    output_dim = dataset.targets.shape[-1]  # CHELSA target dimension

    model = AuroraRegressor(input_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    r2 = R2Score().to(device)
    rmse = MeanSquaredError(squared=False).to(device)

    # Compute normalization stats for training (additional standardization)
    inputs_np = dataset.inputs.numpy()
    targets_np = dataset.targets.numpy()

    # Since data is already pre-normalized, we can use simpler stats or skip this step
    # But we'll keep it for additional standardization during training
    x_mean = torch.tensor(inputs_np.mean(axis=0), dtype=torch.float32).to(device)
    x_std = torch.tensor(inputs_np.std(axis=0) + 1e-8, dtype=torch.float32).to(
        device
    )  # Add epsilon to avoid division by zero
    y_mean = torch.tensor(targets_np.mean(axis=0), dtype=torch.float32).to(device)
    y_std = torch.tensor(targets_np.std(axis=0) + 1e-8, dtype=torch.float32).to(
        device
    )  # Add epsilon to avoid division by zero

    print(f"Model architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training normalization stats:")
    print(f"  Input mean: {x_mean.mean().item():.4f}, std: {x_std.mean().item():.4f}")
    print(f"  Target mean: {y_mean.mean().item():.4f}, std: {y_std.mean().item():.4f}")

    # --- Sanity check: Overfit a small batch ---
    if sanity_check:
        print("Running sanity check...")
        x, y = next(iter(dataloader))
        x, y = x.to(device), y.to(device)

        # Normalize x and y (data is already pre-normalized, this is additional standardization)
        x = (x - x_mean) / x_std
        y = (y - y_mean) / y_std

        for i in range(1000):
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Sanity check step {i}: Loss={loss.item():.6f}")

        # --- Visualize predictions vs. targets ---
        pred = model(x).detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()

        # Also create a plot with original units
        pred_orig = dataset.denormalize_chelsa(pred)
        y_true_orig = dataset.denormalize_chelsa(y_true)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Normalized data
        ax1.scatter(y_true.flatten(), pred.flatten(), alpha=0.5)
        ax1.set_xlabel("True CHELSA (normalized)")
        ax1.set_ylabel("Predicted from Aurora (normalized)")
        ax1.set_title("Sanity check: Normalized data")
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")

        # Plot 2: Original units
        ax2.scatter(y_true_orig.flatten(), pred_orig.flatten(), alpha=0.5)
        ax2.set_xlabel("True CHELSA (K)")
        ax2.set_ylabel("Predicted from Aurora (K)")
        ax2.set_title("Sanity check: Original units")
        ax2.plot(
            [y_true_orig.min(), y_true_orig.max()],
            [y_true_orig.min(), y_true_orig.max()],
            "r--",
        )

        plt.tight_layout()
        plt.savefig("/home/tkhan/bfm-finetune/outputs/aurora_sanity_check.png")
        plt.close()

        if not save_results:
            return

    # Main training loop
    print("Starting main training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        r2.reset()
        rmse.reset()

        # Wrap dataloader with tqdm for progress bar
        for batch_idx, (x, y) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        ):
            x, y = x.to(device), y.to(device)

            # Normalize x and y (data is already pre-normalized, this is additional standardization)
            x = (x - x_mean) / x_std
            y = (y - y_mean) / y_std

            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            r2.update(pred, y)
            rmse.update(pred, y)
            total_loss += loss.item()

            if save_results and batch_idx % 10 == 0:
                # Store results to txt file
                with open(
                    "/home/tkhan/bfm-finetune/outputs/aurora_training_results.txt", "a"
                ) as f:
                    f.write(
                        f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, R2: {r2.compute().item():.4f}, RMSE: {rmse.compute().item():.4f}\n"
                    )

        epoch_loss = total_loss / len(dataloader)
        epoch_r2 = r2.compute()
        epoch_rmse = rmse.compute()

        print(
            f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, R2={epoch_r2:.4f}, RMSE={epoch_rmse:.4f}"
        )

        if save_results:
            with open(
                "/home/tkhan/bfm-finetune/outputs/aurora_training_results.txt", "a"
            ) as f:
                f.write(
                    f"EPOCH_SUMMARY {epoch+1}: Loss={epoch_loss:.4f}, R2={epoch_r2:.4f}, RMSE={epoch_rmse:.4f}\n"
                )

    # Save final model
    if save_results:
        torch.save(
            model.state_dict(),
            "/home/tkhan/bfm-finetune/outputs/aurora_regressor_model.pth",
        )
        print("Model saved to outputs/aurora_regressor_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model to predict CHELSA from Aurora predictions"
    )
    parser.add_argument(
        "--netcdf_path",
        type=str,
        required=True,
        help="Path to the NetCDF file with chelsa_tas and aurora_2t",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training"
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help="Run a sanity check to overfit a small batch of data",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save training results to a text file",
    )
    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.netcdf_path):
        raise FileNotFoundError(f"NetCDF file not found: {args.netcdf_path}")

    train_model(
        netcdf_path=args.netcdf_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        sanity_check=args.sanity_check,
        save_results=args.save_results,
    )
