import numpy as np
import torch
import hydra
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
from pathlib import Path
from torchvision.transforms import Resize
from aurora import AuroraSmall
from bfm_finetune.aurora_mod import AuroraFlex, AuroraRaw2, AuroraRaw
from bfm_finetune.aurora_feature_extractor import extract_features
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoLifeCLEFSpeciesDataset,
)
from bfm_finetune.unet_classification import dict_to_batch, to_device
from bfm_finetune.utils import get_supersampling_target_lat_lon
from torch import pca_lowrank


@hydra.main(config_path=".", config_name="spatial_autocorr")
def main(cfg: DictConfig):
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")
    # load backbone and model
    backbone = AuroraSmall(use_lora=False, autocast=True) # TODO: set Lora to True, set AuroraBig
    backbone.load_checkpoint(cfg.aurora.repo, cfg.aurora.checkpoint) # TODO: set strict = False
    backbone.to(device)
    if cfg.model.supersampling:
        lat_lon = get_supersampling_target_lat_lon(True)
    else:
        # Use keyword args so default data_dir is not clobbered
        ds0 = GeoLifeCLEFSpeciesDataset(
            num_species=cfg.dataset.num_species,
            mode=cfg.dataset.mode,
            negative_lon_mode=cfg.dataset.negative_lon_mode,
        )
        lat_lon = ds0.get_lat_lon()
    model = AuroraRaw2(
        base_model=backbone,
        lat_lon=lat_lon,
        in_channels=cfg.model.in_channels,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=cfg.model.out_channels,
    ).to(device)

    # prepare transform and dataset
    resize = Resize(lat_lon[0].shape[::-1])
    dataset = GeoLifeCLEFSpeciesDataset(
        num_species=cfg.dataset.num_species,
        mode=cfg.dataset.mode,
        negative_lon_mode=cfg.dataset.negative_lon_mode,
    )
    N = min(cfg.run.num_samples or len(dataset), len(dataset))

    features = []
    for i in range(N):
        samp = dataset[i]
        sd = resize(samp["batch"]["species_distribution"])
        if sd.dim() == 4:
            sd = sd.unsqueeze(1)
        samp["batch"]["species_distribution"] = sd
        batch_obj = dict_to_batch(to_device(samp["batch"], device))
        feats = extract_features(
            model.base_model, batch_obj.surf_vars["species_distribution"]
        )
        features.append(feats.cpu().numpy())
    feat_mat = np.vstack(features)  # shape [N, D]

    # PCA reduction if needed
    X = torch.from_numpy(feat_mat).to(device)
    N, D = X.shape
    max_k = min(N, D)
    requested_k = cfg.run.corr_dim or D
    k = min(requested_k, max_k)
    if k < D:
        U, S, V = pca_lowrank(X, q=k)
        Z = U[:, :k] * S[:k]  # [N, k]
        corr = np.corrcoef(Z.cpu().numpy().T)
    else:
        corr = np.corrcoef(feat_mat.T, rowvar=True)

    # plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, cbar_kws={"shrink": 0.5})
    out_dir = Path(cfg.run.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    save_path = out_dir / "feature_correlation_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved correlation heatmap to {save_path}")


if __name__ == "__main__":
    main()
