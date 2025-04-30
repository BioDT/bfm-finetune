import os
from pathlib import Path
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from aurora import AuroraSmall
from aurora.batch import Batch
from torchvision.transforms import Resize

from bfm_finetune.aurora_feature_extractor import extract_features
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoLifeCLEFSpeciesDataset,
)
from bfm_finetune.unet_classification import dict_to_batch, to_device
from bfm_finetune.aurora_mod import AuroraFlex, AuroraRaw2
from bfm_finetune.utils import get_supersampling_target_lat_lon


def naive_morans_i(features: torch.Tensor, coords: torch.Tensor):
    """
    Compute Moran’s I for each feature channel then return mean±std.
    features: [N_feats, D]
    coords:   [N_coords, 2]
    """
    # truncate features to match coords length
    C = coords.shape[0]
    feats = features[:C]
    N, D = feats.shape

    # build weight matrix
    W = torch.zeros((N, N), device=feats.device)
    for i in range(N):
        for j in range(N):
            dist = torch.norm(coords[i] - coords[j])
            W[i, j] = 1.0 / (dist + 1e-5)
    W_sum = W.sum()

    I_vals = []
    for d in range(D):
        x = feats[:, d]
        x_bar = x.mean()
        num = (W * (x[:, None] - x_bar) * (x[None, :] - x_bar)).sum()
        den = ((x - x_bar) ** 2).sum()
        I_vals.append((N / W_sum) * (num / den))

    I_tensor = torch.stack(I_vals)
    return I_tensor.mean().item(), I_tensor.std().item()


@hydra.main(config_path=".", config_name="spatial_autocorr")
def main(cfg: DictConfig):
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")
    # load aurora backbone
    backbone = AuroraSmall(use_lora=False, autocast=True)
    backbone.load_checkpoint(cfg.aurora.repo, cfg.aurora.checkpoint)
    backbone.to(device)

    # supersampling lat/lon if enabled
    if cfg.model.supersampling:
        lat_lon = get_supersampling_target_lat_lon(True)
    else:
        ds = GeoLifeCLEFSpeciesDataset(
            num_species=cfg.dataset.num_species,
            mode=cfg.dataset.mode,
            negative_lon_mode=cfg.dataset.negative_lon_mode,
        )
        lat_lon = ds.get_lat_lon()

    # build model
    model = AuroraRaw2(
        base_model=backbone,
        lat_lon=lat_lon,
        in_channels=cfg.model.in_channels,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=cfg.model.out_channels,
    ).to(device)

    # prepare data
    resize = Resize(lat_lon[0].shape[::-1])  # (lon, lat) → (W, H)
    dataset = GeoLifeCLEFSpeciesDataset(
        num_species=cfg.dataset.num_species,
        mode=cfg.dataset.mode,
        negative_lon_mode=cfg.dataset.negative_lon_mode,
    )

    # Report and optionally limit number of samples
    total = len(dataset)
    print(f"Total samples available: {total}")
    N = cfg.run.num_samples or total
    N = min(N, total)
    print(f"Processing {N} samples")

    features_list = []
    coords_list = []

    for i in range(N):
        sample = dataset[i]
        batch = sample["batch"]
        # resize species_distribution
        sd = resize(batch["species_distribution"])
        if sd.dim() == 4:
            sd = sd.unsqueeze(1)
        batch["species_distribution"] = sd

        # optional static_vars resize
        if "static_vars" in batch:
            for k, v in batch["static_vars"].items():
                batch["static_vars"][k] = resize(v)

        batch_obj = dict_to_batch(to_device(batch, device))
        feats = extract_features(
            model.base_model, batch_obj.surf_vars["species_distribution"]
        )
        features_list.append(feats.cpu())  # collect on CPU
        coords_list.append(torch.tensor(sample["coords"]))  # 1D lat/lon

    # Align counts if any mismatch
    M_feat = len(features_list)
    M_coord = len(coords_list)
    M = min(M_feat, M_coord)
    if M != N:
        print(
            f"Warning: using {M} samples for Moran’s I (features={M_feat}, coords={M_coord})"
        )
    # stack into tensors [M, feat_dim] and [M,2]
    features_tensor = torch.cat(features_list[:M], dim=0).to(device)
    coords_tensor = torch.stack(coords_list[:M], dim=0).to(device)

    # compute Moran’s I mean and std
    mi_mean, mi_std = naive_morans_i(features_tensor, coords_tensor)
    print(f"Global Moran’s I over {M} samples: {mi_mean:.6f} ± {mi_std:.6f}")

    # save two values
    out_dir = Path(cfg.run.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    np.savetxt(out_dir / "moran_I.txt", [mi_mean, mi_std], fmt="%.6f")
    print(f"Saved Moran’s I mean and std to {out_dir/'moran_I.txt'}")


if __name__ == "__main__":
    main()
