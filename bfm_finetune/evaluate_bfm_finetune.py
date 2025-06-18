import os
from pathlib import Path

import torch
import torch.nn as nn
from bfm_model.bfm.rollout_finetuning import BFM_Forecastinglighting as BFM_forecast
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bfm_finetune import bfm_mod
from bfm_finetune.dataloaders.dataloader_utils import custom_collate_fn
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoLifeCLEFSpeciesDataset,
)
from bfm_finetune.finetune_new_variables import (
    save_checkpoint,
    train_epoch,
    validate_epoch,
)
from bfm_finetune.paths import REPO_FOLDER, STORAGE_DIR
from bfm_finetune.utils import (
    get_lat_lon_ranges,
    load_checkpoint,
)

bfm_config_path = REPO_FOLDER / "bfm-model/bfm_model/bfm/configs"
cwd = Path(os.getcwd())
bfm_config_path = str(bfm_config_path.relative_to(cwd))
bfm_config_path = f"../bfm-model/bfm_model/bfm/configs"
print(bfm_config_path)
with initialize(version_base=None, config_path=bfm_config_path, job_name="test_app"):
    cfg = compose(config_name="train_config.yaml")

swin_params = {}
if cfg.model.backbone == "swin":
    selected_swin_config = cfg.model_swin_backbone[cfg.model.swin_backbone_size]
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

# TODO: Add your path here
checkpoint_file = (
    "/home/atrantas/bfm-finetune/bfm_finetune/outputs_bfm_finetune_48800/checkpoints/"
)

# BFM args
bfm_args = dict(
    surface_vars=(cfg.model.surface_vars),
    edaphic_vars=(cfg.model.edaphic_vars),
    atmos_vars=(cfg.model.atmos_vars),
    climate_vars=(cfg.model.climate_vars),
    species_vars=(cfg.model.species_vars),
    vegetation_vars=(cfg.model.vegetation_vars),
    land_vars=(cfg.model.land_vars),
    agriculture_vars=(cfg.model.agriculture_vars),
    forest_vars=(cfg.model.forest_vars),
    redlist_vars=(cfg.model.redlist_vars),
    misc_vars=(cfg.model.misc_vars),
    atmos_levels=cfg.data.atmos_levels,
    species_num=cfg.data.species_number,
    H=cfg.model.H,
    W=cfg.model.W,
    num_latent_tokens=cfg.model.num_latent_tokens,
    backbone_type=cfg.model.backbone,
    patch_size=cfg.model.patch_size,
    embed_dim=cfg.model.embed_dim,
    num_heads=cfg.model.num_heads,
    head_dim=cfg.model.head_dim,
    depth=cfg.model.depth,
    learning_rate=cfg.finetune.lr,
    weight_decay=cfg.finetune.wd,
    batch_size=cfg.finetune.batch_size,
    td_learning=cfg.finetune.td_learning,
    ground_truth_dataset=None,
    # strict=False,  # False if loading from a pre-trained with PEFT checkpoint
    peft_r=cfg.finetune.rank,
    lora_alpha=cfg.finetune.lora_alpha,
    d_initial=cfg.finetune.d_initial,
    peft_dropout=cfg.finetune.peft_dropout,
    peft_steps=cfg.finetune.rollout_steps,
    peft_mode=cfg.finetune.peft_mode,
    use_lora=cfg.finetune.use_lora,
    use_vera=cfg.finetune.use_vera,
    rollout_steps=cfg.finetune.rollout_steps,
    # lora_steps=cfg.finetune.rollout_steps, # 1 month
    # lora_mode=cfg.finetune.lora_mode, # every step + layers #single
    **swin_params,
)

base_model = BFM_forecast(**bfm_args)


finetune_config_path = "."  # f"bfm_finetune"
with initialize(
    version_base=None, config_path=finetune_config_path, job_name="test_app"
):
    finetune_cfg = compose(config_name="finetune_config.yaml")

num_species = finetune_cfg.dataset.num_species
model = bfm_mod.BFMRaw(base_model=base_model, n_species=num_species)

device = base_model.device
model.to(device)

val_dataset = GeoLifeCLEFSpeciesDataset(
    num_species=num_species,
    mode="val",
    negative_lon_mode=finetune_cfg.dataset.negative_lon_mode,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=finetune_cfg.dataset.num_workers,
)

params_to_optimize = model.parameters()

optimizer = torch.optim.AdamW(
    params_to_optimize,
    lr=finetune_cfg.training.lr,
    weight_decay=0.0001,
    betas=(0.9, 0.95),
    eps=1e-8,
)

criterion = nn.L1Loss()

_, best_loss = load_checkpoint(
    model, optimizer, finetune_cfg.training.checkpoint_path, strict=False
)

# final evaluate
for sample in val_dataloader:
    batch = sample["batch"]  # .to(device)
    batch["species_distribution"] = batch["species_distribution"].to(device)
    target = sample["target"]
    with torch.inference_mode():
        prediction, encoder, backbone = model.forward(batch)
    unnormalized_preds = val_dataset.scale_species_distribution(
        prediction.clone(), unnormalize=True
    )
    # TODO These are the backbone's latents! For the test sample
    print(backbone)
    # save_path = predictions_dir / "finetune_predictions.pt"
    # torch.save(unnormalized_preds, save_path)
    # plot_eval(
    #     batch=batch,
    #     # prediction_species=prediction,
    #     prediction_species=unnormalized_preds,
    #     out_dir=plots_dir,
    #     save=True,
    # )
print("DONE")
