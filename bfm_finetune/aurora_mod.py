import contextlib
import dataclasses
from typing import Tuple

from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.batch import Batch

from bfm_finetune.lora_adapter import LoRAAdapter
from bfm_finetune.new_variable_decoder import (
    InputMapper,
    NewModalityEncoder,
    NewVariableHead,
    OutputMapper,
    VectorDecoder,
)
import torch.utils.checkpoint as cp
from functools import partial

from torch.utils.checkpoint import checkpoint
import types

def _wrap_block(block: nn.Module):
    """Return a new forward that runs under torch.utils.checkpoint."""
    def _ckpt_forward(*args, **kw):
        return checkpoint(block._orig_forward, *args, **kw)
    return _ckpt_forward


def enable_swin3d_checkpointing(backbone: nn.Module):
    """
    Replace each Swin3D block's forward with a checkpointed version.
    Call *once* right after model construction.
    """
    for mod in backbone.modules():
        if mod.__class__.__name__.startswith(("SwinTransformerBlock", "Swin", "PatchMerging")):
            # keep reference to original
            mod._orig_forward = mod.forward
            # monkey-patch
            mod.forward = types.MethodType(_wrap_block(mod), mod)



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
        lat_lon: Tuple[np.ndarray, np.ndarray],
        in_channels: int = 1000, # 2 x 500
        hidden_channels: int = 160,
        out_channels: int = 1000,
        atmos_levels: Tuple = (100, 250, 500, 850),
        supersampling_cfg = None,
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

        self.encoder = InputMapper(in_channels=in_channels, timesteps=2, base_channels=64, upsampling=supersampling_cfg, atmos_levels=atmos_levels)
        self.decoder = OutputMapper(out_channels=out_channels, atmos_levels=atmos_levels, downsampling=supersampling_cfg, lat_lon=lat_lon)

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
        # print("batch", x)
        encoded_input = self.encoder(x)
        # print("Encoder output", encoded_input.shape)
        with torch.inference_mode():
            aurora_output = self.base_model(encoded_input)
        # print("Aurora output", aurora_output.shape)
        decoded_aurora = self.decoder(aurora_output)
        # print("Decoder output", decoded_aurora)
        return decoded_aurora
    

# ───────────────────────────── Encoder ────────────────────────────── #
class TemporalSpatialEncoder(nn.Module):
    """
    Accepts x : (B, T=2, C=500, 152, 320)
    Returns (B, 259 200, 512)

    Strategy: merge time -> channel (simple, no blur), 1×1 projection, flatten.
    """
    def __init__(
        self,
        n_species: int = 500,
        n_timesteps: int = 2,
        embed_dim: int = 512,
        target_hw: Tuple[int, int] = (180, 360),#(360, 720),
    ) -> None:
        super().__init__()
        in_channels = n_species * n_timesteps           # 1000
        self.target_hw = target_hw
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) → (B, T*C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b, t * c, h, w)                      # no copy, just reshape
        x = F.interpolate(x, size=self.target_hw, mode="nearest")
        x = self.proj(x)                                # (B, 512, 360, 720)
        x = x.flatten(2).transpose(1, 2)                # (B, 259 200, 512)
        return x


class TemporalSpatialDecoder(nn.Module):
    """
    Backbone: (B, 259 200, 1024)
    Decoded prediction: (B, 1, 500, 152, 320) -> single future/target map
    """
    def __init__(
        self,
        n_species: int = 500,
        in_dim: int = 1024,
        source_hw: Tuple[int, int] = (180, 360), #(360, 720),
        final_hw:  Tuple[int, int] = (152, 320),
    ) -> None:
        super().__init__()
        self.source_hw = source_hw
        self.final_hw  = final_hw
        self.proj = nn.Conv2d(in_dim, n_species, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, p, d = x.shape                               # (B, 259 200, 1024)
        h, w = self.source_hw
        x = x.transpose(1, 2).reshape(b, d, h, w)       # (B, 1024, 360, 720)
        x = self.proj(x)                                # (B, 500, 360, 720)
        x = F.interpolate(x, size=self.final_hw, mode="nearest")  # (B, 500, 152, 320)
        return x.unsqueeze(1)                           # (B, 1, 500, 152, 320)



class AuroraRaw2(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        lat_lon: Tuple[np.ndarray, np.ndarray],
        in_channels: int = 1000, # 2 x 500
        hidden_channels: int = 160,
        out_channels: int = 1000,
        atmos_levels: Tuple = (100, 250, 500, 850),
        supersampling_cfg = None,
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

        self.encoder = InputMapper(in_channels=in_channels, timesteps=2, base_channels=64, upsampling=supersampling_cfg, atmos_levels=atmos_levels)
        self.decoder = OutputMapper(out_channels=out_channels, atmos_levels=atmos_levels, downsampling=supersampling_cfg, lat_lon=lat_lon)

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

        encoded_input = self.encoder(x)
        # print("Encoder output", encoded_input)
        with torch.inference_mode():
            aurora_output = self.base_model(encoded_input)
        print("Aurora output", aurora_output.shape)
        decoded_aurora = self.decoder(aurora_output)
        # print("Decoder output", decoded_aurora)
        return decoded_aurora


def freeze_except_lora(model):
    """
    sets requires_grad=True  only for tensors whose name contains 'lora_'
    everything else is frozen
    returns list of trainable parameter names for sanity‑check
    """
    trainable = []
    for name, param in model.named_parameters():
        if "lora_" in name:                 # catches lora_matrix_A / B, alpha, etc.
            param.requires_grad = True
            trainable.append(name)
        else:
            param.requires_grad = False
    print(f"[LoRA]  trainable params = {len(trainable)} layers")
    return trainable


class AuroraRaw(nn.Module):
    """
    A slim wrapper around Aurora that discards its own encoder/decoder
    and inserts the custom pair above while optionally continously training the backbone
    with the LoRA adapters
    """
    def __init__(
        self,
        base_model,
        n_species: int = 500,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        
        # enable_swin3d_checkpointing(self.base_model.backbone)

        self.base_model.encoder = nn.Identity()
        self.base_model.decoder = nn.Identity()

        self.encoder = TemporalSpatialEncoder(n_species=n_species)
        self.decoder = TemporalSpatialDecoder(n_species=n_species)

        freeze_except_lora(base_model) # 

        # self.patch_res = (
        #     4,
        #     180,
        #     360,
        # )
        self.patch_res = (
            4,
            90,
            180,
        )
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{trainable/1e6:.2f} M / {total/1e6:.2f} M parameters will update")


    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        `batch` must be shape (B, N_species, 152, 320) with dtype float32/16.
        """
        x = batch["species_distribution"]
        tokens = self.encoder(x)                       # (B, 259 200, 512)
        # with torch.inference_mode():
        feats = self.base_model.backbone(tokens,lead_time=timedelta(hours=6.0), patch_res=self.patch_res, rollout_step=1)          # (B, 259 200, 1024)
        recon = self.decoder(feats)                       # (B, N, 152, 320)
        return recon
