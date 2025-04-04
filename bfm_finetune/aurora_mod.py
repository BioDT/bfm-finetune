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
        in_channels: int = 1000, # 2 x 500
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

        self.encoder = NewModalityEncoder(self.in_channels, self.hidden_channels, target_size)
        self.decoder = VectorDecoder(self.latent_dim, self.out_channels, self.hidden_channels, target_size)

        # Freeze pretrained parts.
        for param in self.base_model.encoder.parameters():
            param.requires_grad = False
        for param in self.base_model.backbone.parameters():
            param.requires_grad = False
        for param in self.base_model.decoder.parameters():
            param.requires_grad = False
        print("Initialized AuroraExtend Mod")

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
        aurora_output = self.base_model(encoded_input)
        # print("Aurora output", aurora_output)
        # Decode Aurora output
        decoded_aurora = self.decoder(aurora_output)
        # print("Decoder output", decoded_aurora)
        return decoded_aurora
    

class AuroraFlex(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        in_channels: int = 1000, # 2 x 500
        hidden_channels: int = 160,
        out_channels: int = 1000,
        geo_size: Tuple = (721, 1440),
        atmos_levels: Tuple = (100, 250, 500, 850),
        supersampling: str = False
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

        self.encoder = InputMapper(in_channels=in_channels, timesteps=2, base_channels=64, geo_size=geo_size, atmos_levels=atmos_levels, upsampling=supersampling)
        self.decoder = OutputMapper(out_channels=out_channels, atmos_levels=atmos_levels, downsampling=supersampling)

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
        aurora_output = self.base_model(encoded_input)
        # print("Aurora output", aurora_output)
        # Decode Aurora output
        decoded_aurora = self.decoder(aurora_output)
        # print("Decoder output", decoded_aurora)
        return decoded_aurora
