"""
Feature extractor utility for Aurora models
"""

import torch
import torch.nn as nn
from aurora.batch import Batch


def extract_features(model, encoded_tensor, latent_dim=12160):
    """
    Extract features from an Aurora model's backbone without running full inference.

    Args:
        model: Aurora model instance
        encoded_tensor: Tensor of shape [B, 4, H, W] representing the 4 surface variables
        latent_dim: Dimension of the latent features to return

    Returns:
        Features tensor of shape [B, latent_dim]
    """
    # Create a minimal batch with only the necessary surface variables
    B, C, H, W = encoded_tensor.shape
    surf_vars = {
        "2t": encoded_tensor[:, 0:1],
        "10u": encoded_tensor[:, 1:2],
        "10v": encoded_tensor[:, 2:3],
        "msl": encoded_tensor[:, 3:4],
    }

    # Create placeholder static vars (will be ignored in feature extraction)
    static_vars = {
        k: torch.zeros((H, W), device=encoded_tensor.device, dtype=encoded_tensor.dtype)
        for k in ("lsm", "z", "slt")
    }

    # Create placeholder atmos vars (minimal required for feature extraction)
    atmos_vars = {
        k: torch.zeros(
            (B, 1, 4, H, W), device=encoded_tensor.device, dtype=encoded_tensor.dtype
        )
        for k in ("z", "u", "v", "t", "q")
    }

    # Create a minimal batch
    batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=model.metadata,  # Use model's metadata
    )

    # Extract backbone features (use only what's needed for feature extraction)
    with torch.no_grad():
        batch_normalized = batch.normalise(surf_stats=model.surf_stats)
        batch_cropped = batch_normalized.crop(patch_size=model.patch_size)

        # Get spatial shape after cropping
        H, W = batch_cropped.spatial_shape
        patch_res = (
            model.encoder.latent_levels,
            H // model.encoder.patch_size,
            W // model.encoder.patch_size,
        )

        # Run only the encoder and backbone
        x_enc = model.encoder(batch_cropped, lead_time=model.timestep)
        x_features = model.backbone(
            x_enc,
            lead_time=model.timestep,
            patch_res=patch_res,
            rollout_step=batch.metadata.rollout_step,
        )

        # Return flat features
        features = x_features.view(B, -1)

        # If necessary, adjust feature dimension
        if features.shape[1] != latent_dim:
            features = nn.Linear(features.shape[1], latent_dim).to(features.device)(
                features
            )

    return features
