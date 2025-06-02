import importlib
import math
import os
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
from bfm_model.bfm.dataloader_monthly import (
    LargeClimateDataset,
    batch_to_device,
    custom_collate,
)
from bfm_model.bfm.test_lighting import BFM_lighting
from bfm_model.bfm.train_lighting import BFM_lighting as BFM_lighting_t
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from bfm_finetune import bfm_mod
from bfm_finetune.dataloaders.dataloader_utils import custom_collate_fn
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoLifeCLEFSpeciesDataset,
)
from bfm_finetune.dataloaders.toy_dataset.dataloader import ToyClimateDataset
from bfm_finetune.finetune_new_variables import (
    save_checkpoint,
    train_epoch,
    validate_epoch,
)
from bfm_finetune.paths import REPO_FOLDER, STORAGE_DIR
from bfm_finetune.plots_v2 import plot_eval
from bfm_finetune.utils import (
    get_lat_lon_ranges,
    load_checkpoint,
)

checkpoint_file = STORAGE_DIR / "weights" / "epoch=268-val_loss=0.00493.ckpt"
if not os.path.exists(checkpoint_file):
    raise ValueError(f"checkpoint not found: {checkpoint_file}")

bfm_config_path = REPO_FOLDER / "bfm-model/bfm_model/bfm/configs"
cwd = Path(os.getcwd())
bfm_config_path = str(bfm_config_path.relative_to(cwd))
bfm_config_path = f"../bfm-model/bfm_model/bfm/configs"
print(bfm_config_path)
with initialize(version_base=None, config_path=bfm_config_path, job_name="test_app"):
    bfm_cfg = compose(config_name="train_config.yaml")

swin_params = {}
if bfm_cfg.model.backbone == "swin":
    selected_swin_config = bfm_cfg.model_swin_backbone[bfm_cfg.model.swin_backbone_size]
    swin_params = {
        "swin_encoder_depths": tuple(selected_swin_config.encoder_depths),
        "swin_encoder_num_heads": tuple(selected_swin_config.encoder_num_heads),
        "swin_decoder_depths": tuple(selected_swin_config.decoder_depths),
        "swin_decoder_num_heads": tuple(selected_swin_config.decoder_num_heads),
        "swin_window_size": tuple(selected_swin_config.window_size),
        "swin_mlp_ratio": selected_swin_config.mlp_ratio,
        "swin_qkv_bias": selected_swin_config.qkv_bias,
        "swin_drop_rate": selected_swin_config.drop_rate,
        "swin_attn_drop_rate": selected_swin_config.attn_drop_rate,
        "swin_drop_path_rate": selected_swin_config.drop_path_rate,
        "swin_use_lora": selected_swin_config.use_lora,
    }

# BFM args
bfm_args = dict(
    surface_vars=(bfm_cfg.model.surface_vars),
    edaphic_vars=(bfm_cfg.model.edaphic_vars),
    atmos_vars=(bfm_cfg.model.atmos_vars),
    climate_vars=(bfm_cfg.model.climate_vars),
    species_vars=(bfm_cfg.model.species_vars),
    vegetation_vars=(bfm_cfg.model.vegetation_vars),
    land_vars=(bfm_cfg.model.land_vars),
    agriculture_vars=(bfm_cfg.model.agriculture_vars),
    forest_vars=(bfm_cfg.model.forest_vars),
    redlist_vars=(bfm_cfg.model.redlist_vars),
    misc_vars=(bfm_cfg.model.misc_vars),
    atmos_levels=bfm_cfg.data.atmos_levels,
    species_num=bfm_cfg.data.species_number,
    H=bfm_cfg.model.H,
    W=bfm_cfg.model.W,
    num_latent_tokens=bfm_cfg.model.num_latent_tokens,
    backbone_type=bfm_cfg.model.backbone,
    patch_size=bfm_cfg.model.patch_size,
    embed_dim=bfm_cfg.model.embed_dim,
    num_heads=bfm_cfg.model.num_heads,
    head_dim=bfm_cfg.model.head_dim,
    depth=bfm_cfg.model.depth,
    batch_size=bfm_cfg.evaluation.batch_size,
    **swin_params,
)


class BFMWithLatent(BFM_lighting):
    # overridden to return latents
    def forward(self, batch, lead_time=2, batch_size: int = 1):
        encoded = self.encoder(batch, lead_time, batch_size)
        num_patches_h = self.H // self.encoder.patch_size
        num_patches_w = self.W // self.encoder.patch_size
        total_patches = num_patches_h * num_patches_w  # noqa
        depth = encoded.shape[1] // (num_patches_h * num_patches_w)
        patch_shape = (
            depth,  # depth dimension matches sequence length / (H*W)
            num_patches_h,  # height in patches
            num_patches_w,  # width in patches
        )

        if self.backbone_type == "mvit":
            encoded = encoded.view(encoded.size(0), -1, self.encoder.embed_dim)
            print(f"Reshaped encoded for MViT: {encoded.shape}")
        backbone_output = self.backbone(
            encoded, lead_time=lead_time, rollout_step=0, patch_shape=patch_shape
        )
        output = self.decoder(backbone_output, batch, lead_time)
        return encoded, backbone_output, output

    # overridden to return latents
    def predict_step(self, batch, batch_idx):
        records = []
        x, y = batch
        encoded, backbone_output, output = self(
            x, self.lead_time, batch_size=self.batch_size
        )
        records.append(
            {
                "idx": batch_idx,
                "pred": output,
                "gt": y,
                "encoded": encoded,
                "backbone_output": backbone_output,
            }
        )
        return records


model = BFMWithLatent.load_from_checkpoint(checkpoint_path=checkpoint_file, **bfm_args)

# Batch path (overriden) # bfm_cfg.evaluation.test_data
bfm_cfg.evaluation.test_data = str(STORAGE_DIR / "monthly_batches" / "batches")
bfm_cfg.data.scaling.stats_path = str(
    STORAGE_DIR
    / "monthly_batches"
    / "statistics"
    / "monthly_batches_stats_splitted_channels.json"
)

test_dataset = LargeClimateDataset(
    data_dir=bfm_cfg.evaluation.test_data,
    scaling_settings=bfm_cfg.data.scaling,
    num_species=bfm_cfg.data.species_number,
    atmos_levels=bfm_cfg.data.atmos_levels,
    model_patch_size=bfm_cfg.model.patch_size,
    max_files=3,
)
print("Reading test data from :", bfm_cfg.evaluation.test_data)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=bfm_cfg.evaluation.batch_size,
    num_workers=bfm_cfg.training.workers,
    collate_fn=custom_collate,
    drop_last=True,
    shuffle=False,
)


# batch = next(iter(test_dataloader))
# batch.to(model.device)
# batch_0 = batch[0]
# batch_0 = batch_to_device(batch_0, device=model.device)

# # do a forward pass
# latent = model.encoder(batch_0, lead_time=2, batch_size=1)
# print(latent.shape)
# after_backbone = model.backbone(latent)
# print(after_backbone.shape)

trainer = L.Trainer(
    accelerator=bfm_cfg.training.accelerator,
    devices=bfm_cfg.training.devices,
    precision=bfm_cfg.training.precision,
    log_every_n_steps=bfm_cfg.training.log_steps,
    # limit_test_batches=1,
    limit_predict_batches=12,  # TODO Change this to select how many consecutive months you want to predict
    logger=[],  # [mlf_logger_in_hydra_folder, mlf_logger_in_current_folder],
    enable_checkpointing=False,
    enable_progress_bar=True,
)


predictions = trainer.predict(
    model=model, ckpt_path=checkpoint_file, dataloaders=test_dataloader
)
pred = predictions[0][0]  # 11 (batches) x 3 (batch size)

# The encoded (only after encoder)
print(pred["encoded"].shape)  # torch.Size([1, 64400, 256])
print(pred["backbone_output"].shape)  # torch.Size([1, 64400, 256])
