import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import R2Score, MeanSquaredError
from bfm_finetune.dataloaders.chelsa.dataloader import LatentCHELSADataset
import argparse
import os

class ClimateRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def train_model(netcdf_path, latent_type="backbone_output", batch_size=1, epochs=10, device="cuda"):
    dataset = LatentCHELSADataset(netcdf_path, latent_type=latent_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = dataset.latents.shape[-1]
    model = ClimateRegressor(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    r2 = R2Score().to(device)
    rmse = MeanSquaredError(squared=False).to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        r2.reset()
        rmse.reset()

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x = x.view(-1, input_dim)
            y = y.view(-1, 2)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r2.update(pred, y)
            rmse.update(pred, y)
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, R2={r2.compute():.4f}, RMSE={rmse.compute():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--netcdf_path", type=str, required=True)
    parser.add_argument("--latent_type", type=str, default="backbone_output", 
                      choices=["encoder_output", "backbone_output"], 
                      help="Which latent representation to use for training")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.netcdf_path):
        raise FileNotFoundError(f"NetCDF file not found: {args.netcdf_path}")

    train_model(
        netcdf_path=args.netcdf_path,
        latent_type=args.latent_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device
    )