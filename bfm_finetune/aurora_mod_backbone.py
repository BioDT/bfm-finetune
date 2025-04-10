import contextlib
import dataclasses
from typing import Tuple

import torch
import torch.nn as nn
from aurora.batch import Batch

from bfm_finetune.lora_adapter import LoRAAdapter
from bfm_finetune.new_variable_decoder import (
    InputMapper,
    NewModalityEncoder,
    NewVariableHead,
    OutputMapper,
    VectorDecoder,
)
from bfm_finetune.aurora_feature_extractor import extract_features


class ChannelAdapter(nn.Module):
    def __init__(self, in_channels=1, out_channels=128):
        super(ChannelAdapter, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AuroraModified(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        new_input_channels: int,
        target_size: Tuple[int, int],
        use_new_head: bool = True,
        latent_dim: int = 12160,
    ):
        """
        Wraps a pretrained Aurora model (e.g. AuroraSmall) to adapt a new input with different channels
        and produce a new high-dimensional output.
        """
        super().__init__()
        self.base_model = base_model  # Pre-instantiated AuroraSmall model.
        self.new_input_channels = new_input_channels  # 500
        self.use_new_head = use_new_head

        # Freeze pretrained parts.
        for param in self.base_model.encoder.parameters():
            param.requires_grad = False
        for param in self.base_model.backbone.parameters():
            param.requires_grad = False
        for param in self.base_model.decoder.parameters():
            param.requires_grad = False

        # AuroraSmall expects 4 surf_vars channels ("2t", "10u", "10v", "msl").
        self.expected_input_channels = 4
        self.input_adapter = LoRAAdapter(
            new_input_channels, self.expected_input_channels, rank=4
        )

        if self.use_new_head:
            target_size = target_size
            self.new_head = NewVariableHead(
                latent_dim, out_channels=new_input_channels, target_size=target_size
            )

    def forward(self, batch: Batch):
        p = next(self.parameters())
        batch = batch.type(p.dtype)
        batch = batch.to(p.device)

        # Adapt the new input.
        if "species_distribution" not in batch.surf_vars:
            raise ValueError(
                "Finetuning input must include 'species_distribution' in batch.surf_vars."
            )
        new_input = batch.surf_vars["species_distribution"]
        # Allow for optional time dimension.
        if new_input.dim() == 4:
            new_input = new_input.unsqueeze(1)  # (B, 1, C, H, W)
        B, T, C_new, H, W = new_input.shape
        expected_T = 2  # If needed, replicate the time dimension.
        if T < expected_T:
            new_input = new_input.repeat(1, expected_T, 1, 1, 1)
            T = expected_T

        # Merge batch and time dimensions, apply adapter, then reshape back.
        new_input_reshaped = new_input.view(B * T, C_new, H, W)
        adapted = self.input_adapter(
            new_input_reshaped
        )  # (B*T, expected_input_channels, H, W)
        adapted = adapted.view(B, T, self.expected_input_channels, H, W)

        # Split into expected surf_vars keys.
        new_surf_vars = {}
        var_names = list(self.base_model.surf_vars)  # e.g., ("2t", "10u", "10v", "msl")
        for i, name in enumerate(var_names):
            new_surf_vars[name] = adapted[:, :, i : i + 1, :, :].squeeze(2)
        batch = dataclasses.replace(batch, surf_vars=new_surf_vars)

        # Continue with normalization and cropping.
        batch = batch.normalise(surf_stats=self.base_model.surf_stats)
        batch = batch.crop(patch_size=self.base_model.patch_size)

        # Expand static_vars properly.
        B, T = next(iter(batch.surf_vars.values())).shape[:2]
        batch = dataclasses.replace(
            batch,
            static_vars={
                k: v.unsqueeze(0).unsqueeze(0).expand(B, T, *v.shape)
                for k, v in batch.static_vars.items()
            },
        )

        H, W = batch.spatial_shape
        patch_res = (
            self.base_model.encoder.latent_levels,
            H // self.base_model.encoder.patch_size,
            W // self.base_model.encoder.patch_size,
        )

        x = self.base_model.encoder(batch, lead_time=self.base_model.timestep)
        with (
            torch.autocast(device_type="cuda")
            if self.base_model.autocast
            else contextlib.nullcontext()
        ):
            x = self.base_model.backbone(
                x,
                lead_time=self.base_model.timestep,
                patch_res=patch_res,
                rollout_step=batch.metadata.rollout_step,
            )
        if self.use_new_head:
            new_output = self.new_head(x)  # New head expects input of 128 channels.
            return new_output
        else:
            original_output = self.base_model.decoder(
                x,
                batch,
                lead_time=self.base_model.timestep,
                patch_res=patch_res,
            )
            return original_output


class AuroraExtend(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        latent_dim: int,
        in_channels: int = 1000,
        hidden_channels: int = 160,
        out_channels: int = 1000,
        target_size: Tuple[int, int] = [152, 320],
    ):
        """
        Wraps a pretrained Aurora model using a simpler approach with feature extraction.
        """
        super().__init__()
        self.base_model = base_model
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.target_size = target_size

        # Simple CNN-based encoder to map species distribution to 4 channels expected by Aurora
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=1),  # 4 output channels for the 4 surface vars
        )

        # A simple decoder that takes the backbone output and maps directly to the target output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_channels * target_size[0] * target_size[1]),
        )

        # Freeze the base model completely
        for param in base_model.parameters():
            param.requires_grad = False

        print("Initialized Simplified AuroraExtend")

    def forward(self, batch):
        # Extract the species distribution from the batch
        if "species_distribution" not in batch.surf_vars:
            raise ValueError("Input must contain species_distribution in surf_vars")

        species_data = batch.surf_vars["species_distribution"]

        # Handle dimensions - we need (B, C, H, W)
        if species_data.dim() == 5:  # [B, T, C, H, W]
            B, T, C, H, W = species_data.shape
            species_input = species_data[:, 0]  # Use first timestep only
        else:  # [B, C, H, W]
            B, C, H, W = species_data.shape
            species_input = species_data

        # Encode the species data to 4 channels with our simple CNN
        encoded = self.encoder(species_input)  # [B, 4, H, W]

        # Use our feature extraction helper
        backbone_features = extract_features(self.base_model, encoded, self.latent_dim)

        # Decode to target output with our simple MLP
        output = self.decoder(backbone_features)

        # Reshape to desired output format [B, out_channels, H, W]
        output = output.view(
            B, self.out_channels, self.target_size[0], self.target_size[1]
        )

        # Add time dimension to match expected shape [B, 1, out_channels, H, W]
        output = output.unsqueeze(1)

        return output


class AuroraFlex(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        in_channels: int = 1000,
        hidden_channels: int = 160,
        out_channels: int = 1000,
        geo_size: Tuple = (721, 1440),
        atmos_levels: Tuple = (100, 250, 500, 850),
        supersampling: str = False,
    ):
        """
        Wraps a pretrained Aurora model (e.g. AuroraSmall) to adapt a new input with different channels
        and produce a new high-dimensional output.
        """
        super().__init__()
        self.base_model = base_model  # Pre-instantiated AuroraSmall model.
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.encoder = InputMapper(
            in_channels=in_channels,
            timesteps=2,
            base_channels=64,
            geo_size=geo_size,
            atmos_levels=atmos_levels,
            upsampling=supersampling,
        )
        self.decoder = OutputMapper(
            out_channels=out_channels,
            atmos_levels=atmos_levels,
            downsampling=supersampling,
        )

        # Freeze pretrained parts.
        for param in self.base_model.encoder.parameters():
            param.requires_grad = False
        for param in self.base_model.backbone.parameters():
            param.requires_grad = False
        for param in self.base_model.decoder.parameters():
            param.requires_grad = False
        print("Initialized AuroraFlex Mod")

    def forward(self, batch):
        x = batch
        # Encode input
        encoded_input = self.encoder(x)
        # Pass through the Aurora model
        with torch.inference_mode():
            aurora_output = self.base_model(encoded_input)
        # Decode Aurora output
        decoded_aurora = self.decoder(aurora_output)
        return decoded_aurora
