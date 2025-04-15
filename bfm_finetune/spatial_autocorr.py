import numpy as np
import torch
from aurora.batch import Batch
from bfm_finetune.aurora_feature_extractor import extract_features
from bfm_finetune.dataloaders.geolifeclef_species.dataloader import GeoLifeCLEFSpeciesDataset
from bfm_finetune.unet_classification import dict_to_batch, to_device
from bfm_finetune.aurora_mod import AuroraExtend
from aurora import AuroraSmall

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def naive_morans_i(features: torch.Tensor, coords: torch.Tensor):
    """
    Simplistic Moran’s I calculation for demonstration.
    features: [B, feat_dim]
    coords:   [B, 2] lat/lon or x/y for each sample
    """
    N = features.shape[0]
    W = torch.zeros((N, N), device=features.device)
    for i in range(N):
        for j in range(N):
            dist = torch.norm(coords[i] - coords[j])
            W[i, j] = 1.0 / (dist + 1e-5)
    W_sum = torch.sum(W)
    x = features.mean(dim=1)  # simple average if multi-dim
    x_bar = x.mean()
    num = 0
    den = torch.sum((x - x_bar)**2)
    for i in range(N):
        for j in range(N):
            num += W[i, j] * (x[i] - x_bar) * (x[j] - x_bar)
    I = (N / W_sum) * (num / den)
    return I.item()

def main():
    # Load Pretrained Aurora Backbone
    backbone = AuroraSmall(use_lora=False, autocast=True)
    backbone.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
    backbone.to(device)

    # Build the model using Aurora as backbone via AuroraExtend
    num_species = 500
    target_size = (152, 320)
    latent_dim = 12160
    model = AuroraExtend(
        base_model=backbone,
        latent_dim=latent_dim,
        in_channels=num_species,
        hidden_channels=160,
        out_channels=num_species,
        target_size=target_size,
    )
    model.to(device)

    # Load dataset
    dataset = GeoLifeCLEFSpeciesDataset(num_species=num_species, mode="train")
    sample = dataset[0]  # single sample
    batch_dict = to_device(sample["batch"], device)
    batch = dict_to_batch(batch_dict)

    # Extract features from Aurora model
    features = extract_features(model.base_model, batch.surf_vars["species_distribution"])

    # Assume we have coordinates for each sample
    coords = sample["coords"]  # shape [B, 2], each row is (lat, lon)
    coords_t = torch.tensor(coords, device=features.device, dtype=features.dtype)

    # Compute naive Moran’s I
    mi_value = naive_morans_i(features, coords_t)
    print(f"Naive Moran’s I: {mi_value}")

if __name__ == "__main__":
    main()