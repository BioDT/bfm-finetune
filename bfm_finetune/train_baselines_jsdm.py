from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bfm_finetune.dataloaders.dataloader_utils import custom_collate_fn
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoLifeCLEFSpeciesDataset,
)
from bfm_finetune.metrics import compute_geolifeclef_f1, compute_rmse
from bfm_finetune.utils import seed_everything


# Baseline 1: Latent-factor MLP jSDM (per-pixel, non-spatial)
class LatentMLPjSDM(nn.Module):
    """
    Site-independent latent-factor jSDM:

    Input:  [B * H * W, C]  presence / features at time t
    Output: [B * H * W, C]  logits for presence at time t+1
    """

    def __init__(self, n_species: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Linear(n_species, latent_dim, bias=False)
        self.activation = nn.ReLU()
        self.decoder = nn.Linear(latent_dim, n_species)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.encoder(x))
        logits = self.decoder(z)
        return logits  # BCEWithLogitsLoss will be applied outside


# Baseline 2: ConvLSTM jSDM with latent species embedding
class ConvLSTMCell(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True
    ):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, state):
        """
        x:     [B, C_in, H, W]
        state: (h, c), each [B, hidden_dim, H, W]
        """
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    """
    Single-layer ConvLSTM over a sequence of feature maps.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, H, W]
        returns: last hidden state h_T: [B, hidden_dim, H, W]
        """
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        c = torch.zeros_like(h)

        for t in range(T):
            h, c = self.cell(x[:, t], (h, c))

        return h


class ConvLSTM_JSDM(nn.Module):
    """
    Spatiotemporal jSDM:

    Input:  [B, T, C, H, W] species presence up to time t
            (here T=1: only the first year)
    Output: [B, C, H, W] logits for presence at time t+1
    """

    def __init__(
        self,
        n_species: int,
        latent_dim: int = 64,
        hidden_dim: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.n_species = n_species
        self.latent_dim = latent_dim

        # Species encoder: C -> K latent “community factors”
        self.species_encoder = nn.Linear(n_species, latent_dim, bias=False)
        self.act = nn.ReLU()

        # Spatiotemporal ConvLSTM over latent factors
        self.convlstm = ConvLSTM(
            input_dim=latent_dim, hidden_dim=hidden_dim, kernel_size=kernel_size
        )

        # Decoder from ConvLSTM hidden state to species logits
        self.species_decoder = nn.Linear(hidden_dim, n_species)

    def encode_species(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, H, W] -> [B, T, K, H, W]
        """
        B, T, C, H, W = x.shape
        # Flatten (B, T, H, W) as batch for linear layer
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(-1, C)  # [B*T*H*W, C]
        z_flat = self.act(self.species_encoder(x_flat))  # [B*T*H*W, K]
        z = z_flat.view(B, T, H, W, self.latent_dim).permute(0, 1, 4, 2, 3)
        return z  # [B, T, K, H, W]

    def decode_species(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, hidden_dim, H, W] -> logits: [B, C, H, W]
        """
        B, hidden_dim, H, W = h.shape
        h_flat = h.permute(0, 2, 3, 1).reshape(-1, hidden_dim)  # [B*H*W, hidden_dim]
        logits_flat = self.species_decoder(h_flat)  # [B*H*W, C]
        logits = logits_flat.view(B, H, W, self.n_species).permute(0, 3, 1, 2)
        return logits  # [B, C, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, H, W]
        """
        z = self.encode_species(x)  # [B, T, K, H, W]
        h_last = self.convlstm(z)  # [B, hidden_dim, H, W]
        logits = self.decode_species(h_last)
        return logits


# Training and evaluation utilities
def train_baseline1(
    train_loader: DataLoader,
    n_species: int,
    latent_dim: int = 64,
    num_epochs: int = 5,
    lr: float = 1e-3,
    device: torch.device | str = "cuda",
) -> LatentMLPjSDM:
    """
    Train Baseline 1: LatentMLPjSDM on GeoLifeCLEFSpeciesDataset.
    """
    device = torch.device(device)
    model = LatentMLPjSDM(n_species=n_species, latent_dim=latent_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        n_samples = 0

        for batch_dict in train_loader:
            # species_distribution: [B, 2, C, H, W]
            species_dist = batch_dict["batch"]["species_distribution"].to(device)
            # target: [B, 1, C, H, W]
            target = batch_dict["target"].to(device)

            # Use year 0 as input, year 1 as target
            x = species_dist[:, 0]  # [B, C, H, W]
            y = target[:, 0]  # [B, C, H, W]

            B, C, H, W = x.shape

            # Flatten spatial dims; each pixel is a sample
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
            # Binary targets for BCE
            y_flat = (y > 0).float().permute(0, 2, 3, 1).reshape(-1, C)

            optimizer.zero_grad()
            logits = model(x_flat)  # [B*H*W, C]
            loss = criterion(logits, y_flat)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_flat.size(0)
            n_samples += x_flat.size(0)

        epoch_loss = running_loss / max(1, n_samples)
        print(f"[Baseline1] Epoch {epoch+1}/{num_epochs} - loss: {epoch_loss:.4f}")

    return model


def train_baseline2(
    train_loader: DataLoader,
    n_species: int,
    latent_dim: int = 64,
    hidden_dim: int = 64,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device | str = "cuda",
) -> ConvLSTM_JSDM:
    """
    Train Baseline 2: ConvLSTM_JSDM on GeoLifeCLEFSpeciesDataset.
    """
    device = torch.device(device)
    model = ConvLSTM_JSDM(
        n_species=n_species,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        kernel_size=3,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        n_batches = 0

        for batch_dict in train_loader:
            species_dist = batch_dict["batch"]["species_distribution"].to(
                device
            )  # [B, 2, C, H, W]

            # Input sequence: only year 0 for now, but keep T dim
            x = species_dist[:, 0:1]  # [B, 1, C, H, W]
            y = species_dist[:, 1]  # [B, C, H, W]
            y_bin = (y > 0).float()  # [B, C, H, W]

            optimizer.zero_grad()
            logits = model(x)  # [B, C, H, W]
            loss = criterion(logits, y_bin)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            n_batches += x.size(0)

        epoch_loss = running_loss / max(1, n_batches)
        print(f"[Baseline2] Epoch {epoch+1}/{num_epochs} - loss: {epoch_loss:.4f}")

    return model


@torch.no_grad()
def evaluate_baseline1(
    model: LatentMLPjSDM,
    val_loader: DataLoader,
    device: torch.device | str = "cuda",
) -> Tuple[float, float]:
    """
    Evaluate Baseline 1 on validation set.
    Returns (GeoLifeCLEF F1, RMSE).
    """
    device = torch.device(device)
    model.eval()

    f1_list = []
    rmse_list = []

    for batch_dict in val_loader:
        species_dist = batch_dict["batch"]["species_distribution"].to(
            device
        )  # [B, 2, C, H, W]
        target = batch_dict["target"].to(device)  # [B, 1, C, H, W]

        # Input: year 0
        x = species_dist[:, 0]  # [B, C, H, W]
        B, C, H, W = x.shape

        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        logits_flat = model(x_flat)
        probs_flat = torch.sigmoid(logits_flat)  # [B*H*W, C]

        probs = probs_flat.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        preds = probs.unsqueeze(1)  # [B, 1, C, H, W]

        target_bin = (target > 0).float()

        f1 = compute_geolifeclef_f1(preds, target_bin)
        rmse = compute_rmse(preds, target_bin)

        f1_list.append(f1)
        rmse_list.append(rmse)

    return float(np.mean(f1_list)), float(np.mean(rmse_list))


@torch.no_grad()
def evaluate_baseline2(
    model: ConvLSTM_JSDM,
    val_loader: DataLoader,
    device: torch.device | str = "cuda",
) -> Tuple[float, float]:
    """
    Evaluate Baseline 2 on validation set.
    Returns (GeoLifeCLEF F1, RMSE).
    """
    device = torch.device(device)
    model.eval()

    f1_list = []
    rmse_list = []

    for batch_dict in val_loader:
        species_dist = batch_dict["batch"]["species_distribution"].to(
            device
        )  # [B, 2, C, H, W]
        target = batch_dict["target"].to(device)  # [B, 1, C, H, W]

        # Input: year 0 as sequence of length 1
        x = species_dist[:, 0:1]  # [B, 1, C, H, W]
        logits = model(x)  # [B, C, H, W]
        probs = torch.sigmoid(logits).unsqueeze(1)  # [B, 1, C, H, W]

        target_bin = (target > 0).float()

        f1 = compute_geolifeclef_f1(probs, target_bin)
        rmse = compute_rmse(probs, target_bin)

        f1_list.append(f1)
        rmse_list.append(rmse)

    return float(np.mean(f1_list)), float(np.mean(rmse_list))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Hyperparameters
    num_species = 500
    batch_size_train = 1  # we flatten spatially; 1 cube per batch is fine
    batch_size_val = 1
    latent_dim = 64
    hidden_dim = 64
    epochs_baseline1 = 100
    epochs_baseline2 = 50
    lr = 1e-3

    data_dir = Path(
        "/home/thanasis.trantas/github_projects/bfm-finetune/aurorashape_species"
    )

    train_dataset = GeoLifeCLEFSpeciesDataset(
        data_dir=data_dir,
        num_species=num_species,
        mode="train",
        unnormalize=False,  # we binarize targets ourselves
        negative_lon_mode="ignore",
    )
    val_dataset = GeoLifeCLEFSpeciesDataset(
        data_dir=data_dir,
        num_species=num_species,
        mode="val",
        unnormalize=False,
        negative_lon_mode="ignore",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    print("=== Training Baseline 1 (LatentMLPjSDM) ===")
    model1 = train_baseline1(
        train_loader=train_loader,
        n_species=num_species,
        latent_dim=latent_dim,
        num_epochs=epochs_baseline1,
        lr=lr,
        device=device,
    )
    print("=== Evaluating Baseline 1 ===")
    f1_1, rmse_1 = evaluate_baseline1(model1, val_loader, device=device)
    print(f"Baseline 1 - GeoLifeCLEF F1: {f1_1:.4f}, RMSE: {rmse_1:.4f}")

    print("\n=== Training Baseline 2 (ConvLSTM_JSDM) ===")
    model2 = train_baseline2(
        train_loader=train_loader,
        n_species=num_species,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_epochs=epochs_baseline2,
        lr=lr,
        device=device,
    )
    print("=== Evaluating Baseline 2 ===")
    f1_2, rmse_2 = evaluate_baseline2(model2, val_loader, device=device)
    print(f"Baseline 2 - GeoLifeCLEF F1: {f1_2:.4f}, RMSE: {rmse_2:.4f}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    seed_everything(42)
    main()
