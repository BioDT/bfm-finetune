import contextlib
import dataclasses
from typing import Tuple

import numpy as np
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
            # The backbone returns that
            # latent_dim = 128  # 17x32
            # latent_dim = 12160  # (152, 320)
            # latent_dim = target_size[0] * target_size[1] // self.base_model.patch_size
            # print("latent_dim", latent_dim)
            # Determine the target spatial size from the metadata.
            # target_size = (
            #     int(self.base_model.metadata.lat.shape[0]),
            #     int(self.base_model.metadata.lon.shape[0])
            # )
            target_size = target_size
            self.new_head = NewVariableHead(
                latent_dim, out_channels=new_input_channels, target_size=target_size
            )
            # self.channel_adapter = ChannelAdapter(in_channels=1, out_channels=latent_dim)

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
        # print("new_input.shape", new_input.shape)
        # Allow for optional time dimension.
        if new_input.dim() == 4:
            new_input = new_input.unsqueeze(1)  # (B, 1, C, H, W)
        B, T, C_new, H, W = new_input.shape
        # print("new_input", new_input.shape)
        expected_T = 2  # If needed, replicate the time dimension.
        if T < expected_T:
            new_input = new_input.repeat(1, expected_T, 1, 1, 1)
            T = expected_T
        # print("new_input", new_input.shape)

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
        # print(f"encoder shape {x.shape}")
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
            # print(f"Backbone shape: {x.shape}")
        if self.use_new_head:
            # Use the channel adapter to map from 1 to 128 channels.
            # x = self.channel_adapter(x)  # Now x has shape [B, 128, H_lat, W_lat]
            new_output = self.new_head(x)  # New head expects input of 128 channels.
            # print(f"new_output shape: {new_output.shape}")
            return new_output
        else:
            original_output = self.base_model.decoder(
                x,
                batch,
                lead_time=self.base_model.timestep,
                patch_res=patch_res,
            )
            # print(f"original_output shape: {original_output.surf_vars["2t"].shape}")
            return original_output


class AuroraExtend(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        latent_dim: int,
        in_channels: int = 1000,  # 2 x 500
        hidden_channels: int = 160,
        out_channels: int = 1000,
        target_size: Tuple[int, int] = [152, 320],
    ):
        """
        Wraps a pretrained Aurora model (e.g. AuroraSmall) to adapt a new input with different channels
        and produce a new high-dimensional output.
        """
        super().__init__()
        self.base_model = base_model  # Pre-instantiated AuroraSmall model.
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.target_size = target_size

        self.encoder = NewModalityEncoder(
            self.in_channels, self.hidden_channels, target_size
        )

        # Revert to the actual Aurora backbone output (4 * 17 * 32 = 2176)
        input_size = 2176

        self.shape_adapter = nn.Sequential(
            nn.Flatten(), nn.Linear(input_size, self.latent_dim)
        )

        self.decoder = VectorDecoder(
            self.latent_dim, self.out_channels, self.hidden_channels, target_size
        )

        # Freeze pretrained parts.
        for param in self.base_model.encoder.parameters():
            param.requires_grad = False
        for param in self.base_model.backbone.parameters():
            param.requires_grad = False
        for param in self.base_model.decoder.parameters():
            param.requires_grad = False
        print("Initialized AuroraExtend Mod")

    def _get_feature_shape(self, sample_batch):
        """Get the expected shape of features from the Aurora model using a sample batch"""
        with torch.no_grad():
            try:
                # Try to use encoded input
                encoded_input = self.encoder(sample_batch)
                # Extract features from the first surface variable
                first_key = next(iter(encoded_input.surf_vars.keys()))
                first_var = encoded_input.surf_vars[first_key]
                # Use feature extractor to get backbone output
                features = extract_features(self.base_model, first_var)
                return features.shape[1]  # Return the feature dimension
            except Exception as e:
                print(f"Could not determine feature shape: {e}")
                return 2176  # Fallback to default

    def forward(self, batch):
        # Encode input
        print(f"Input batch type: {type(batch)}")

        # Get the shape of species_distribution for debugging
        if hasattr(batch, "surf_vars") and "species_distribution" in batch.surf_vars:
            print(
                f"Species distribution shape: {batch.surf_vars['species_distribution'].shape}"
            )

        encoded_input = self.encoder(batch)
        print(
            f"Encoder output shape: {[k + ':' + str(v.shape) for k, v in encoded_input.surf_vars.items() if hasattr(v, 'shape')]}"
        )

        # Pass through the Aurora model
        try:
            # Try direct feature extraction for more stable output shape
            first_key = next(iter(encoded_input.surf_vars.keys()))
            first_var = encoded_input.surf_vars[first_key]
            print(
                f"Using feature extraction on {first_key} with shape {first_var.shape}"
            )

            # Extract features directly from the backbone
            with torch.no_grad():
                extracted_features = extract_features(
                    self.base_model, first_var.squeeze(1), self.latent_dim
                )

            print(f"Extracted features shape: {extracted_features.shape}")
            decoded_aurora = self.decoder(extracted_features)

        except Exception as e:
            print(f"Feature extraction failed: {e}, falling back to full model")

            # Traditional approach
            aurora_output = self.base_model(encoded_input)

            # Get diagnostic information about Aurora output
            if isinstance(aurora_output, Batch):
                # If Aurora returns a Batch object
                print(
                    f"Aurora output is Batch with keys: {aurora_output.surf_vars.keys()}"
                )
                # Extract the first surface variable and adapt its shape
                first_key = next(iter(aurora_output.surf_vars.keys()))
                first_var = aurora_output.surf_vars[first_key]
                print(f"Aurora output shape for {first_key}: {first_var.shape}")

                # Get real shape for shape adapter (first time only)
                if not hasattr(self, "_shape_adapter_initialized"):
                    input_shape = first_var.numel() // first_var.shape[0]
                    print(f"Actual shape from Aurora: {input_shape}")

                    # Recreate shape adapter with correct dimensions if needed
                    if self.shape_adapter[1].in_features != input_shape:
                        print(
                            f"Reinitializing shape adapter: {input_shape} -> {self.latent_dim}"
                        )
                        device = first_var.device
                        self.shape_adapter = nn.Sequential(
                            nn.Flatten(), nn.Linear(input_shape, self.latent_dim)
                        ).to(device)

                    self._shape_adapter_initialized = True

                # Flatten and adapt the shape
                adapted_output = self.shape_adapter(first_var)
            else:
                # If Aurora returns a tensor directly
                print(f"Aurora output shape: {aurora_output.shape}")

                # Get real shape for shape adapter (first time only)
                if not hasattr(self, "_shape_adapter_initialized"):
                    input_shape = aurora_output.numel() // aurora_output.shape[0]
                    print(f"Actual shape from Aurora: {input_shape}")

                    # Recreate shape adapter with correct dimensions if needed
                    if self.shape_adapter[1].in_features != input_shape:
                        print(
                            f"Reinitializing shape adapter: {input_shape} -> {self.latent_dim}"
                        )
                        device = aurora_output.device
                        self.shape_adapter = nn.Sequential(
                            nn.Flatten(), nn.Linear(input_shape, self.latent_dim)
                        ).to(device)

                    self._shape_adapter_initialized = True

                adapted_output = self.shape_adapter(aurora_output)

            print(f"Adapted output shape: {adapted_output.shape}")

            # Decode Aurora output with properly shaped input
            decoded_aurora = self.decoder(adapted_output)

        print(f"Decoder output shape: {decoded_aurora.shape}")
        return decoded_aurora


class AuroraFlex(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        lat_lon: Tuple[np.ndarray, np.ndarray],
        in_channels: int = 1000,  # 2 x 500
        hidden_channels: int = 160,
        out_channels: int = 1000,
        atmos_levels: Tuple = (100, 250, 500, 850),
        supersampling_cfg=None,
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
            upsampling=supersampling_cfg,
            atmos_levels=atmos_levels,
        )
        self.decoder = OutputMapper(
            out_channels=out_channels,
            atmos_levels=atmos_levels,
            downsampling=supersampling_cfg,
            lat_lon=lat_lon,
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
        # p = next(self.parameters())
        # batch = batch.type(p.dtype)
        # batch = batch.to(p.device) # Bach here has T=2
        # TODO Adapt the dataset to provide only the new variable
        # x = batch.surface_vars["species_distribution"]
        x = batch
        # Encode input
        # print("batch", x)
        encoded_input = self.encoder(x)
        # print("Encoder output", encoded_input)
        # Pass through the Aurora model
        with torch.inference_mode():
            aurora_output = self.base_model(encoded_input)
        # print("Aurora output", aurora_output)
        # Decode Aurora output
        decoded_aurora = self.decoder(aurora_output)
        # print("Decoder output", decoded_aurora)
        return decoded_aurora
