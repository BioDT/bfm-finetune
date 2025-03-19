import contextlib
import dataclasses
from functools import partial

import torch
import torch.nn as nn
from aurora.batch import Batch
from bfm_finetune.lora_adapter import LoRAAdapter
from bfm_finetune.new_variable_decoder import NewVariableHead


class ChannelAdapter(nn.Module):
    def __init__(self, in_channels=1, out_channels=128):
        super(ChannelAdapter, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

class AuroraModified(nn.Module):
    def __init__(self, base_model: nn.Module, new_input_channels: int, use_new_head: bool = True):
        """
        Wraps a pretrained Aurora model (e.g. AuroraSmall) to adapt a new input with different channels
        and produce a new high-dimensional output.
        """
        super().__init__()
        self.base_model = base_model  # Pre-instantiated AuroraSmall model.
        self.new_input_channels = new_input_channels
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
        self.input_adapter = LoRAAdapter(new_input_channels, self.expected_input_channels, rank=4)

        if self.use_new_head:
            # The backbone returns that
            latent_dim = 128 
            # Determine the target spatial size from the metadata.
            # target_size = (
            #     int(self.base_model.metadata.lat.shape[0]),
            #     int(self.base_model.metadata.lon.shape[0])
            # )
            target_size = (17,32)
            self.new_head = NewVariableHead(latent_dim, out_channels=10000, target_size=target_size)
            self.channel_adapter = ChannelAdapter(in_channels=1, out_channels=latent_dim)

    def forward(self, batch: Batch):
        p = next(self.parameters())
        batch = batch.type(p.dtype)
        batch = batch.to(p.device)
        
        # Adapt the new input.
        if "new_input" not in batch.surf_vars:
            raise ValueError("Finetuning input must include 'new_input' in batch.surf_vars.")
        new_input = batch.surf_vars["new_input"]
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
        adapted = self.input_adapter(new_input_reshaped)  # (B*T, expected_input_channels, H, W)
        adapted = adapted.view(B, T, self.expected_input_channels, H, W)
        
        # Split into expected surf_vars keys.
        new_surf_vars = {}
        var_names = list(self.base_model.surf_vars)  # e.g., ("2t", "10u", "10v", "msl")
        for i, name in enumerate(var_names):
            new_surf_vars[name] = adapted[:, :, i:i+1, :, :].squeeze(2)
        batch = dataclasses.replace(batch, surf_vars=new_surf_vars)
        
        # Continue with normalization and cropping.
        batch = batch.normalise(surf_stats=self.base_model.surf_stats)
        batch = batch.crop(patch_size=self.base_model.patch_size)
        
        # Expand static_vars properly.
        B, T = next(iter(batch.surf_vars.values())).shape[:2]
        batch = dataclasses.replace(
            batch,
            static_vars={k: v.unsqueeze(0).unsqueeze(0).expand(B, T, *v.shape) for k, v in batch.static_vars.items()},
        )
        
        H, W = batch.spatial_shape
        patch_res = (
            self.base_model.encoder.latent_levels,
            H // self.base_model.encoder.patch_size,
            W // self.base_model.encoder.patch_size,
        )
        
        x = self.base_model.encoder(batch, lead_time=self.base_model.timestep)
        with torch.autocast(device_type="cuda") if self.base_model.autocast else contextlib.nullcontext():
            x = self.base_model.backbone(
                x,
                lead_time=self.base_model.timestep,
                patch_res=patch_res,
                rollout_step=batch.metadata.rollout_step,
            )
        if self.use_new_head:
            # Use the channel adapter to map from 1 to 128 channels.
            x = self.channel_adapter(x)  # Now x has shape [B, 128, H_lat, W_lat]
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
