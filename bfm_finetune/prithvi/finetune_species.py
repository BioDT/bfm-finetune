import argparse
import os
from typing import Literal

import torch
import torch.distributed as dist
import tqdm
from gravity_wave_finetuning.distributed import print0
from gravity_wave_finetuning.gravity_wave_model import UNetWithTransformer
from torch.nn.parallel import DistributedDataParallel as DDP

# import wandb
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (  # TODO: test that it works properly
    GeoCLEFDataModulePrithvi,
)
from bfm_finetune.prithvi.utils import prithvi_output_checkpoint_path

# local_rank = int(os.environ["LOCAL_RANK"])
local_rank = 0
# global_rank = int(os.environ["RANK"])
global_rank = 0
device = f"cuda:{local_rank}"
dtype = torch.float32


def count_forward_pass_parameters(model):
    """Count the total number of parameters in a model that are used in the forward pass
    and have `requires_grad=True`.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: The total number of parameters used in the forward pass.
    """
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
    return total_params


def setup():
    """Initialize the process group for distributed training and set the CUDA device."""
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)


def cleanup():
    """Destroy the process group to clean up resources after training."""
    dist.destroy_process_group()


def train(cfg, rank):
    """Train the model using the specified configuration and rank.

    Args:
        cfg: The configuration object containing hyperparameters and paths.
        rank: The rank of the process in distributed training.
    """

    # Setup dataloaders
    vartype: Literal["uvtp122"] = cfg.vartype
    print0(f"Loading NetCDF data for variable type: {vartype}")

    # Setup Weights and Biases (wandb) logger
    # if rank == 0:
    #     wandb.init(
    #         entity="define-entity",
    #         project="gravity-wave-flux",
    #         dir="logs",
    #         name=f"gwf_14_pre_{vartype}",
    #         mode=cfg.wandb_mode,
    #     )

    setup()

    # Initialize the data module and setup the dataset for training
    datamodule = GeoCLEFDataModulePrithvi(
        batch_size=cfg.batch_size,
        num_data_workers=cfg.num_data_workers,
    )
    datamodule.setup(stage="fit")

    # Initialize the model and optimizer
    model: torch.nn.Module = UNetWithTransformer(
        lr=cfg.lr,
        hidden_channels=cfg.hidden_channels,
        in_channels=500,  # Num of species
        out_channels=500,
        n_lats_px=cfg.n_lats_px,
        n_lons_px=cfg.n_lons_px,
        in_channels_static=cfg.in_channels_static,
        mask_unit_size_px=cfg.mask_unit_size_px,
        patch_size_px=cfg.patch_size_px,
        device=device,
        ckpt_singular=cfg.singular_sharded_checkpoint,
    )
    optimizer: torch.optim.Optimizer = model.configure_optimizers()

    # Wrap model in DistributedDataParallel for multi-GPU training
    model = DDP(model.to(rank, dtype=dtype), device_ids=[rank])

    # Count and log the number of trainable parameters
    total_params = count_forward_pass_parameters(model)
    print0(f"TOTAL TRAINING PARAMETERS: {total_params:,}")

    # Start finetuning the model
    if rank == 0:
        print("Starting to finetune")
    # Initialize best_loss to a very high value
    best_loss = 1_000_000
    for epoch in tqdm.trange(cfg.max_epochs):
        model.train()

        # Training loop
        pbar_train = tqdm.tqdm(
            iterable=datamodule.train_dataloader(), disable=(rank != 0)
        )
        for batch in pbar_train:
            # Move batch data to the appropriate device
            batch = {key: val.to(rank, dtype=dtype) for key, val in batch.items()}
            optimizer.zero_grad()

            # Forward pass
            y_hat: torch.Tensor = model.forward(batch)
            # print("Prediction and target shapes", y_hat.shape, batch["target"].shape)
            # Compute loss and metrics
            loss: torch.Tensor = torch.nn.functional.mse_loss(
                input=y_hat, target=batch["target"]
            )
            print(f"Train loss {loss.item()}")

            # Log training loss to wandb
            if rank == 0:
                pbar_train.set_postfix(ordered_dict={"train/loss": float(loss)})
                # wandb.log(data={"train/loss": float(loss)})

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

        # Validation loop
        val_losses = []
        pbar_val = tqdm.tqdm(iterable=datamodule.val_dataloader(), disable=(rank != 0))
        with torch.no_grad():
            model.eval()

            for batch in pbar_val:
                # Move batch data to the appropriate device
                batch = {key: val.to(rank, dtype=dtype) for key, val in batch.items()}

                # Forward pass
                y_hat: torch.Tensor = model.forward(batch)

                # Compute validation loss and metrics
                val_loss: torch.Tensor = torch.nn.functional.mse_loss(
                    input=y_hat, target=batch["target"]
                )
                val_losses.append(val_loss.item())

                print(f"Val loss {val_loss.item()}")

                # Log validation loss to wandb
                if rank == 0:
                    pbar_val.set_postfix(ordered_dict={"val/loss": float(val_loss)})
                    # wandb.log(data={"val/loss": float(val_loss)})

            # Compute average validation loss for the epoch
            avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch}: Average Val Loss {avg_val_loss:.4f}")
        # Save model checkpoint after each epoch
        if rank == 0:
            # wandb.log(data={"epoch/avg_val_loss": avg_val_loss, "epoch": epoch})
            # Save checkpoint only if the validation loss improves
            if avg_val_loss < best_loss and epoch % cfg.val_every == 0:
                best_loss = avg_val_loss
                ckpt_path: str = str(prithvi_output_checkpoint_path)
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(obj=model.state_dict(), f=ckpt_path)
                print(f"Improved validation loss, checkpoint saved to {ckpt_path}")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="uvtp122",
        help="determines which dataset to use for training",
    )
    args = parser.parse_args()

    if args.split == "uvtp122":
        from config import get_cfg
    else:
        raise NotImplementedError

    cfg = get_cfg()
    train(cfg, local_rank)
