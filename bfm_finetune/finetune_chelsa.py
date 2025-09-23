import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, R2Score
from tqdm import tqdm
from bfm_finetune.dataloaders.chelsa.dataloader import LatentCHELSADataset


class ClimateRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
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
    dataset = LatentCHELSADataset(netcdf_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = dataset.input_dim  # <-- input_dim is already an int
    model = ClimateRegressor(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    r2 = R2Score().to(device)
    rmse = MeanSquaredError(squared=False).to(device)

    # Compute normalization stats
    inputs = dataset.inputs  # <-- use attribute, not dict-style access
    targets = dataset.targets if hasattr(dataset, "targets") else dataset.labels

    # Convert to numpy arrays if needed
    if hasattr(inputs, "values"):
        inputs_np = inputs.values
    else:
        inputs_np = inputs
    if hasattr(targets, "values"):
        targets_np = targets.values
    else:
        targets_np = targets

    x_mean = torch.tensor(inputs_np.mean(axis=(0, 1)), dtype=torch.float32).to(device)
    x_std = torch.tensor(inputs_np.std(axis=(0, 1)), dtype=torch.float32).to(device)
    y_mean = torch.tensor(targets_np.mean(axis=(0, 1)), dtype=torch.float32).to(device)
    y_std = torch.tensor(targets_np.std(axis=(0, 1)), dtype=torch.float32).to(device)

    # --- Sanity check: Overfit a small batch ---
    if sanity_check:
        x, y = next(iter(dataloader))
        x, y = x.to(device), y.to(device)
        x = (x - x_mean) / (x_std + 1e-8)
        y = (y - y_mean) / (y_std + 1e-8)
        x = x.mean(dim=1)
        if y.ndim == 3:
            y = y.mean(dim=1)
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
        plt.figure(figsize=(12, 4))
        plt.scatter(y_true.flatten(), pred.flatten(), alpha=0.5)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title("Sanity check: True vs. Predicted")
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
        plt.tight_layout()
        plt.savefig("sanity_check_true_vs_pred.png")
        plt.close()
        # Only return if not also saving results (i.e., skip main training loop only if not save_results)
        if not save_results:
            return

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        r2.reset()
        rmse.reset()

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            # Normalize x and y
            x = (x - x_mean) / (x_std + 1e-8)
            y = (y - y_mean) / (y_std + 1e-8)
            # Mean-pool the spatial dimension
            x = x.mean(dim=1)  # [batch, 256]
            if y.ndim == 3:
                y = y.mean(dim=1)  # [batch, 2]
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r2.update(pred, y)
            rmse.update(pred, y)
            total_loss += loss.item()

            if save_results:
                # store results to txt file
                with open(
                    "/home/tkhan/bfm-finetune/outputs/chelsa_training_results.txt", "a"
                ) as f:
                    f.write(
                        f"Epoch {epoch+1}, Loss: {loss.item():.4f}, R2: {r2.compute().item():.4f}, RMSE: {rmse.compute().item():.4f}\n"
                    )

        print(
            f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, R2={r2.compute():.4f}, RMSE={rmse.compute():.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--netcdf_path", type=str, required=True)
    parser.add_argument(
        "--latent_type",
        type=str,
        required=False,
        default="backbone_output",
        choices=["encoder_output", "backbone_output"],
        help="Which latent representation to use for training",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
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

    if not os.path.exists(args.netcdf_path):
        raise FileNotFoundError(f"NetCDF file not found: {args.netcdf_path}")

    train_model(
        netcdf_path=args.netcdf_path,
        # latent_type=args.latent_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        sanity_check=args.sanity_check,
        save_results=args.save_results,
    )
