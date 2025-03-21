from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from aurora import Aurora, AuroraSmall
from aurora.batch import Batch, Metadata
from torch.utils.data import DataLoader, Dataset, default_collate

from bfm_finetune.aurora_mod import AuroraModified


# Custom collate function to merge a list of Batch objects.
def collate_batches(batch_list):
    # Merge surf_vars, static_vars, and atmos_vars by stacking their tensor values.
    surf_vars = {
        k: torch.stack([b.surf_vars[k] for b in batch_list], dim=0)
        for k in batch_list[0].surf_vars.keys()
    }
    # static_vars = {
    #     k: torch.stack([b.static_vars[k] for b in batch_list], dim=0)
    #     for k in batch_list[0].static_vars.keys()
    # }
    atmos_vars = {
        k: torch.stack([b.atmos_vars[k] for b in batch_list], dim=0)
        for k in batch_list[0].atmos_vars.keys()
    }
    # For static_vars, we simply take the one from the first sample.
    static_vars = batch_list[0].static_vars
    # For metadata, we assume they are the same across samples.
    metadata = batch_list[0].metadata
    return Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=metadata,
    )


def custom_collate_fn(samples):
    # Each sample is a dict with keys "batch" and "target".
    batch_list = [s["batch"] for s in samples]
    collated_batch = collate_batches(batch_list)
    targets = default_collate([s["target"] for s in samples])
    return {"batch": collated_batch, "target": targets}


class ToyClimateDataset(Dataset):
    def __init__(self, num_samples=200, new_input_channels=10, num_species=10000):
        self.num_samples = num_samples
        self.new_input_channels = new_input_channels
        self.num_species = num_species
        # Define latitude and longitude grids.
        self.lat = torch.linspace(90, -90, 17)  # 17 latitude points
        self.lon = torch.linspace(0, 360, 33)[:-1]  # 32 longitude points
        self.metadata = Metadata(
            lat=self.lat,
            lon=self.lon,
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(100, 250, 500, 850),
        )
        self.H, self.W = len(self.lat), len(self.lon)
        print(f"Lat {self.H} Long {self.W}")
        # Set history length T to 2 (as required by the Aurora encoder).
        self.T = 2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate new finetuning input with shape (T, new_input_channels, H, W).
        new_input = torch.randn(self.T, self.new_input_channels, self.H, self.W)
        surf_vars = {"new_input": new_input}
        static_vars = {k: torch.randn(self.H, self.W) for k in ("lsm", "z", "slt")}
        atmos_vars = {
            k: torch.randn(2, 4, self.H, self.W) for k in ("z", "u", "v", "t", "q")
        }
        batch = Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=self.metadata,
        )
        # Target: high-dimensional variable with shape (num_species, H, W).
        target = torch.randn(self.num_species, self.H, self.W)
        return {"batch": batch, "target": target}


def finetune_new_variables(use_small=True):
    if use_small:
        base_model = AuroraSmall()
        base_model.load_checkpoint(
            "microsoft/aurora", "aurora-0.25-small-pretrained.ckpt"
        )
        embed_dim = 256
    else:
        base_model = AuroraSmall()
        base_model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
        embed_dim = 512
    base_model.to("cuda")
    # config = {
    #     "embed_dim": embed_dim,
    # }
    new_input_channels = 10  # Our new finetuning dataset has 10 channels.

    model = AuroraModified(
        base_model=base_model,
        new_input_channels=new_input_channels,
        use_new_head=True,
        # **config,
    )
    model.to("cuda")

    # Freeze all pretrained parts. We already froze base_model inside AuroraModified.
    # Ensure that in the input adapter, only the LoRA parameters are trainable.
    for name, param in model.input_adapter.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False

    # Optimizer on the LoRA adapter parameters and new head.
    params_to_optimize = (
        list(model.input_adapter.lora_A.parameters())
        + list(model.input_adapter.lora_B.parameters())
        + list(model.new_head.parameters())
    )
    optimizer = optim.AdamW(params_to_optimize, lr=1e-3)
    criterion = nn.MSELoss()

    dataset = ToyClimateDataset(
        num_samples=1000, new_input_channels=new_input_channels, num_species=10000
    )
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=15,
    )

    model.train()
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for sample in dataloader:
            batch = sample["batch"]
            targets = sample["target"].to("cuda")
            optimizer.zero_grad()
            outputs = model(batch)  # outputs: (B, 10000, H, W)
            # print(f"output shape {outputs.shape}")
            # print(f"target shape {targets.shape}")
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")


if __name__ == "__main__":
    finetune_new_variables()
