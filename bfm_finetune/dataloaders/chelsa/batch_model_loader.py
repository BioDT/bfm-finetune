import os
from pathlib import Path

import torch
from hydra import compose, initialize
from torch.utils.data import DataLoader
import lightning as L

from bfm_model.bfm.dataloader_monthly import (
    LargeClimateDataset,
    custom_collate,
)
from bfm_model.bfm.test_lighting import BFM_lighting
from bfm_finetune.paths import REPO_FOLDER, STORAGE_DIR


class BFMWithLatent(BFM_lighting):
    """Same class as in bfm_get_latent.py - extracts latents during forward pass"""
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


def get_bfm_model_and_dataloader(bfm_cfg):
    """Function to load BFM model and dataloader as specified in the config"""
    checkpoint_file = STORAGE_DIR / "weights" / "epoch=268-val_loss=0.00493.ckpt"
    if not os.path.exists(checkpoint_file):
        raise ValueError(f"checkpoint not found: {checkpoint_file}")
    
    # Setup Swin parameters if needed
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
    
    # BFM args - same as in bfm_get_latent.py
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
    
    # Load model from checkpoint - exactly as in bfm_get_latent.py
    model = BFMWithLatent.load_from_checkpoint(checkpoint_path=checkpoint_file, **bfm_args)
    
    # Setup dataset and dataloader
    test_dataset = LargeClimateDataset(
        data_dir=bfm_cfg.evaluation.test_data,
        scaling_settings=bfm_cfg.data.scaling,
        num_species=bfm_cfg.data.species_number,
        atmos_levels=bfm_cfg.data.atmos_levels,
        model_patch_size=bfm_cfg.model.patch_size,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=bfm_cfg.evaluation.batch_size,
        num_workers=bfm_cfg.training.workers,
        collate_fn=custom_collate,
        drop_last=True,
        shuffle=False,
    )
    
    # Extract filenames to use as time indices - instead of calling get_time_index()
    # Get file list from dataset
    time_index = []
    data_dir = bfm_cfg.evaluation.test_data
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")]
    files.sort()  # Sort to ensure consistent ordering
    
    # Extract dates from filenames (assuming format like "batch_2018-06-01_to_2018-07-01.pt")
    for file in files:
        filename = os.path.basename(file)
        # Try to extract date range from filename
        parts = filename.split('_')
        if len(parts) >= 3 and "to" in filename:
            # e.g., batch_2018-06-01_to_2018-07-01.pt
            date_range = "_".join(parts[1:]).replace(".pt", "")
            start_date = date_range.split("_to_")[0]
            time_index.append(start_date)
        else:
            # If filename doesn't match expected pattern, use filename as index
            time_index.append(filename)
            
    
    trainer = L.Trainer(
        accelerator=bfm_cfg.training.accelerator,
        devices=bfm_cfg.training.devices,
        precision=bfm_cfg.training.precision,
        log_every_n_steps=bfm_cfg.training.log_steps,
        # limit_test_batches=1,
        #limit_predict_batches=228,  # TODO Change this to select how many consecutive months you want to predict
        logger=[],  # [mlf_logger_in_hydra_folder, mlf_logger_in_current_folder],
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    print("The lnegth of the test_dataloader:", len(test_dataloader))  # Force dataset to load and check for errors
    
    predictions = trainer.predict(
    model=model, ckpt_path=checkpoint_file, dataloaders=test_dataloader
    )
    # Generate time_index for all months from 2000-01 to 2019-12
    import pandas as pd
    time_index = pd.date_range("2000-01-01", "2020-12-01", freq="MS").strftime("%Y-%m-%d").tolist()

    return model, test_dataloader, time_index, predictions