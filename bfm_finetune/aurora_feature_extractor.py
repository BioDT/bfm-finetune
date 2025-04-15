"""
Feature extractor utility for Aurora models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from aurora.batch import Batch, Metadata


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
    # Squeeze out time dimension if it is 1
    if encoded_tensor.dim() == 5 and encoded_tensor.shape[1] == 1:
        encoded_tensor = encoded_tensor.squeeze(1)  # Now shape [B, C, H, W]

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

    # Create a minimal metadata containing atmos_levels used by normalise()
    dummy_metadata = Metadata(
        time=torch.zeros(1),
        lat=torch.linspace(90, -90, steps=H),  # strictly decreasing from 90 to -90
        lon=torch.linspace(0, 359, steps=W),  # strictly increasing in [0, 360)
        rollout_step=0,
        atmos_levels=[100, 250, 500, 850],
    )

    batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=dummy_metadata,
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
            rollout_step=0,
        )

        # Ensure spatial dimensions match before concatenation
        if x_surf.shape[-2:] != x_static.shape[-2:]:
            x_static = F.interpolate(x_static, size=(x_surf.shape[-2], x_surf.shape[-1]), mode="bilinear", align_corners=False)

        # Concatenate tensors along dimension 2
        x_surf = torch.cat((x_surf, x_static), dim=2)  # (B, T, V_S + V_Static, H, W)

        # Return flat features
        features = x_features.view(B, -1)

    return features
