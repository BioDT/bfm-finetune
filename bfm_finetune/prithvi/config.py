from yacs.config import CfgNode as CN

from bfm_finetune.paths import REPO_FOLDER
from bfm_finetune.prithvi.utils import prithvi_species_patches_location

_CN = CN()

_CN.wandb_mode = "disabled"
_CN.vartype = "uvtp122"
# _CN.train_data_path = (
#     # "data/finetune/geolifeclef24/aurorashape_species/train"
#     str(prithvi_species_patches_location / "train")
# )
# _CN.valid_data_path = (
#     # "data/finetune/geolifeclef24/aurorashape_species/val"
#     str(prithvi_species_patches_location / "val")
# )
_CN.singular_sharded_checkpoint = (
    # "prithvi_wxc/v0.8.50.rollout_step3.1.pth"
    str(REPO_FOLDER / "checkpoints/prithvi.wxc.rollout.2300m.v1.pt")
)
_CN.file_glob_pattern = "wxc_input_u_v_t_p_output_theta_uw_vw_era5_*.nc"

_CN.lr = 0.01
_CN.hidden_channels = 160
_CN.n_lats_px = 64  # 152 #64
_CN.n_lons_px = 128  # 320 #128
_CN.in_channels_static = 3
_CN.mask_unit_size_px = [8, 16]
_CN.patch_size_px = [1, 1]
_CN.val_every = 5


### Training Params

_CN.max_epochs = 50  # 520
_CN.batch_size = 4
_CN.num_data_workers = 8


def get_cfg():
    return _CN.clone()
