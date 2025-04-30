import os
from pathlib import Path

import hydra
import mlflow
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from aurora import Aurora, AuroraSmall
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from bfm_finetune.aurora_mod import AuroraFlex, AuroraRaw
from bfm_finetune.dataloaders.dataloader_utils import custom_collate_fn
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoLifeCLEFSpeciesDataset,
)
from bfm_finetune.dataloaders.toy_dataset.dataloader import ToyClimateDataset
from bfm_finetune.metrics import compute_rmse, compute_spc, compute_ssim_metric
from bfm_finetune.plots import plot_eval
from bfm_finetune.utils import load_checkpoint, save_checkpoint, seed_everything

# TODO Make the configurable - maybe from bash script
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"


def compute_statio_temporal_loss(outputs, targets):
    criterion = nn.MSELoss()
    weight_1 = 10.0
    weight_2 = 5.0
    weight_3 = 0.5
    # print(f"prediction shape : {outputs.shape} | target shape : {targets.shape} ")
    ssim = compute_ssim_metric(outputs, targets)
    spc = compute_spc(outputs, targets)
    rmse_t = criterion(outputs, targets)
    rmse = compute_rmse(outputs, targets)
    # print(f"SSIM: {ssim} | SPC: {spc} | RMSE: {rmse}")
    loss = (
        weight_3 * rmse + weight_1 * (1.0 - ssim) + weight_2 * (1.0 - (spc + 1.0) / 2.0)
    )
    return loss


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    for sample in dataloader:
        batch = sample["batch"].to(device)
        targets = sample["target"].to(device)
        optimizer.zero_grad()
        outputs = model(batch)  # e.g., outputs shape: [B, 10000, H, W]
        loss = criterion(outputs, targets)
        # loss = compute_statio_temporal_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(dataloader)
    return epoch_loss


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    with torch.inference_mode():
        for sample in dataloader:
            batch = sample["batch"].to(device)
            targets = sample["target"].to(device)
            outputs = model(batch)
            loss = criterion(outputs, targets)
            # loss = compute_statio_temporal_loss(outputs, targets)
            epoch_loss += loss.item()
    epoch_loss /= len(dataloader)
    return epoch_loss


def main_worker(rank, world_size, cfg, output_dir, gpu_ids):
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    torch.cuda.set_device(gpu_ids[rank])
    device = torch.device(f"cuda:{gpu_ids[rank]}")

    if rank == 0:
        print(f"Output directory: {output_dir}")
        print(f"Using GPUs: {gpu_ids}")
        print(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set the internal precision for TensorCores (H100s)
    torch.set_float32_matmul_precision(cfg.training.precision_in)

    # Seed the experiment for numpy, torch and python.random.
    # TODO: sometimes linked to loss nan???
    # seed_everything(0)

    if cfg.model.base_small:
        base_model = AuroraSmall()
        base_model.load_checkpoint(
            "microsoft/aurora", "aurora-0.25-small-pretrained.ckpt"
        )
        atmos_levels = (100, 250, 500, 850)
    elif cfg.model.big:
        base_model = Aurora(use_lora=False)  # stabilise_level_agg=True
        base_model.load_checkpoint(
            "microsoft/aurora", "aurora-0.25-pretrained.ckpt"
        )  # strict=False
        atmos_levels = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    elif cfg.model.big_ft:
        base_model = Aurora(use_lora=False)
        base_model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")

    base_model.to(device)

    num_species = (
        cfg.dataset.num_species
    )  # Our new finetuning dataset has 500 channels.
    geo_size = (152, 320)  # WORKS
    # geo_size = (17, 32)  # WORKS
    if cfg.model.supersampling:
        geo_size = (721, 1440)  # NOT WORK

    if rank == 0:
        print(f"Map size: {geo_size}")
    latent_dim = 12160
    num_epochs = cfg.training.epochs

    if cfg.dataset.toy:
        dataset = ToyClimateDataset(
            num_samples=100,
            new_input_channels=num_species,
            num_species=num_species,
            geo_size=geo_size,
        )
    else:
        train_dataset = GeoLifeCLEFSpeciesDataset(num_species=num_species, mode="train")
        val_dataset = GeoLifeCLEFSpeciesDataset(num_species=num_species, mode="val")

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,
        collate_fn=custom_collate_fn,
        num_workers=cfg.dataset.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=cfg.dataset.num_workers,
    )
    ###### V3
    model = AuroraFlex(
        base_model=base_model,
        in_channels=num_species,
        hidden_channels=cfg.model.hidden_dim,
        out_channels=num_species,
        geo_size=geo_size,
        atmos_levels=atmos_levels,
        supersampling=cfg.model.supersampling,
    )
    params_to_optimize = model.parameters()

    model.to(device)

    if cfg.training.backend.lower() == "fsdp":
        model = FSDP(model, use_orig_params=True)
        if rank == 0:
            print("Using FSDP for distributed training.")
    else:
        model = DDP(model, device_ids=[gpu_ids[rank]])
        if rank == 0:
            print("Using DDP for distributed training.")

    optimizer = optim.AdamW(params_to_optimize, lr=cfg.training.lr)
    criterion = nn.MSELoss()

    checkpoint_save_path = Path(output_dir) / "checkpoints"

    # Load checkpoint if available
    _, best_loss = load_checkpoint(model, optimizer, cfg.training.checkpoint_path)
    val_loss = 1_000_000
    mlflow.set_experiment("BFM_Finetune")

    if rank == 0:
        plots_dir = Path(output_dir) / "plots"
        os.makedirs(plots_dir, exist_ok=True)

    with mlflow.start_run():
        mlflow.log_param("num_epochs", cfg.training.epochs)
        mlflow.log_param("learning_rate", cfg.training.lr)
        mlflow.log_param("batch_size", cfg.training.batch_size)
        for epoch in range(1, num_epochs):
            # Ensure a different shuffling for each epoch
            train_sampler.set_epoch(epoch)

            train_loss = train_epoch(
                model, train_dataloader, optimizer, criterion, device
            )

            if epoch % cfg.training.val_every == 0:
                val_loss = validate_epoch(model, val_dataloader, criterion, device)
                if rank == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")
                    mlflow.log_metric("val_loss", val_loss, step=epoch + 1)

                if val_loss < best_loss and rank == 0:
                    best_loss = val_loss
                    save_checkpoint(
                        model, optimizer, epoch, best_loss, checkpoint_save_path
                    )
                    mlflow.log_metric("best_loss", best_loss, step=epoch + 1)

            if rank == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
                mlflow.log_metric("train_loss", train_loss, step=epoch + 1)

    # final evaluate
    if rank == 0:
        for sample in val_dataloader:
            batch = sample["batch"].to(device)
            target = sample["target"]
            with torch.inference_mode():
                # For DDP, use model.module; for FSDP, unwrapping may be needed.
                if cfg.training.backend.lower() == "fsdp":
                    prediction = (
                        model.module.forward(batch)
                        if hasattr(model, "module")
                        else model.forward(batch)
                    )
                else:
                    prediction = model.module.forward(batch)
            plot_eval(
                batch=batch, prediction_species=prediction, out_dir=plots_dir, save=True
            )

    dist.destroy_process_group()


@hydra.main(version_base=None, config_path="", config_name="finetune_config")
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    gpu_ids = cfg.training.gpus
    world_size = len(gpu_ids)
    mp.spawn(
        main_worker,
        args=(world_size, cfg, output_dir, gpu_ids),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
