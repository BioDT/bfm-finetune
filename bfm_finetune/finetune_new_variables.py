from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from aurora import Aurora, AuroraSmall
from aurora.batch import Batch, Metadata
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm

from bfm_finetune.aurora_mod import AuroraModified, AuroraExtend, AuroraFlex
from bfm_finetune.dataloaders.dataloader_utils import custom_collate_fn
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoLifeCLEFSpeciesDataset,
)
from bfm_finetune.dataloaders.toy_dataset.dataloader import ToyClimateDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def finetune_new_variables(use_small=True, use_toy=True):
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
    base_model.to(device)

    num_species = 500  # Our new finetuning dataset has 1000 channels.
    geo_size = (152, 320)  # WORKS
    # geo_size = (17, 32)  # WORKS
    batch_size = 1 if use_toy else 2
    latent_dim = 12160


    if use_toy:
        dataset = ToyClimateDataset(
            num_samples=100,
            new_input_channels=num_species,
            num_species=num_species,
            geo_size=geo_size,
        )
    else:
        dataset = GeoLifeCLEFSpeciesDataset(num_species=num_species)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=15,
    )
    #######   V1
    # model = AuroraModified(
    #     base_model=base_model,
    #     new_input_channels=num_species,
    #     use_new_head=True,
    #     target_size=geo_size,
    #     latent_dim=latent_dim,
    # )

    # # Freeze all pretrained parts. We already froze base_model inside AuroraModified.
    # # Ensure that in the input adapter, only the LoRA parameters are trainable.
    # for name, param in model.input_adapter.named_parameters():
    #     if "lora_A" not in name and "lora_B" not in name:
    #         param.requires_grad = False

    # # Optimizer on the LoRA adapter parameters and new head.
    # params_to_optimize = (
    #     list(model.input_adapter.lora_A.parameters())
    #     + list(model.input_adapter.lora_B.parameters())
    #     + list(model.new_head.parameters())
    # )

    ####### V2
    # model = AuroraExtend(base_model=base_model,
    #                      latent_dim=latent_dim,
    #                      in_channels=num_species,
    #                      hidden_channels=128,
    #                      out_channels=num_species,
    #                      target_size=geo_size)
    # params_to_optimize = model.parameters()

    ###### V3
    model = AuroraFlex(base_model=base_model, in_channels=num_species, out_channels=num_species)
    params_to_optimize = model.parameters()
    
    model.to(device)
    
    optimizer = optim.AdamW(params_to_optimize, lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    num_epochs = 20
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for sample in dataloader:
            batch = sample["batch"].to(device)
            targets = sample["target"].to(device)
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
    # finetune_new_variables(use_toy=True)
    finetune_new_variables(use_toy=True)
