import contextlib
import dataclasses
from functools import partial

import torch
import torch.nn as nn
from aurora.batch import Batch
from lora_adapter import LoRAAdapter
from new_variable_decoder import NewVariableHead


class AuroraModified(nn.Module):
    """
    AuroraModified adapts the pretrained AuroraSmall model to a new finetuning task
    in which the input variables differ (e.g. a different number of channels) and the desired output
    is high-dimensional (e.g. 10,000 species occurrences). It uses an input adapter (with LoRA)
    to map the new input to the pretrained space and replaces the decoder with a new head.
    """

    def __init__(
        self,
        base_model: nn.Module,
        new_input_channels: int = 10,
        use_new_head: bool = True,
        **kwargs,
    ):
        """
        Args:
            new_input_channels (int): Number of channels in the new finetuning input.
            use_new_head (bool): If True, use the new decoder head.
            kwargs: Additional arguments for AuroraSmall.
                    For AuroraSmall, defaults are:
                    surf_vars=("2t", "10u", "10v", "msl"), etc.
        """
        super().__init__()
        self.base_model = base_model
        # Remove unsupported keys before passing to AuroraSmall.
        if "expected_input_channels" in kwargs:
            kwargs.pop("expected_input_channels")

        # Freeze the pretrained parts.
        for param in self.base_model.encoder.parameters():
            param.requires_grad = False
        for param in self.base_model.backbone.parameters():
            param.requires_grad = False
        for param in self.base_model.decoder.parameters():
            param.requires_grad = False

        # The original AuroraSmall expects surf_vars with 4 channels (for "2t", "10u", "10v", "msl").
        self.expected_input_channels = len(self.base_model.surf_vars) + len(
            self.base_model.atmos_vars
        )
        print("expected vars dim", self.expected_input_channels)
        # Create the new input adapter using LoRA.
        self.input_adapter = LoRAAdapter(
            new_input_channels, self.expected_input_channels, rank=4
        )

        self.use_new_head = use_new_head
        if self.use_new_head:
            # For AuroraSmall, embed_dim is 256 (see AuroraSmall definition below).
            latent_dim = kwargs.get("embed_dim", 256)
            # Use the patch_size from the base model to determine upsampling factor.
            upsample_factor = self.base_model.patch_size
            self.new_head = NewVariableHead(
                latent_dim=1, out_channels=10000, upsample_factor=upsample_factor
            )

    def forward(self, batch: Batch):
        """
        Forward pass for the modified model. Expects that the finetuning Batch contains a new input under
        batch.surf_vars["new_input"] with shape (B, new_input_channels, H, W).
        The rest of the batch (static_vars, atmos_vars, metadata) is used as in the original model.
        """
        p = next(self.parameters())
        batch = batch.type(p.dtype)
        batch = batch.to(p.device)

        # Extract new input, expected shape: (B, T, new_input_channels, H, W)
        if "new_input" not in batch.surf_vars:
            raise ValueError(
                "Finetuning input must include 'new_input' in batch.surf_vars."
            )
        new_input = batch.surf_vars["new_input"]
        # new_input now should have shape (B, T, new_input_channels, H, W)
        B, T, C_new, H, W = new_input.shape
        # Merge batch and time dimensions
        new_input_reshaped = new_input.view(B * T, C_new, H, W)
        # Apply the input adapter (LoRA) to map new_input_channels -> expected_input_channels
        adapted = self.input_adapter(
            new_input_reshaped
        )  # shape: (B*T, expected_input_channels, H, W)
        # Un-merge to (B, T, expected_input_channels, H, W)
        adapted = adapted.view(B, T, self.expected_input_channels, H, W)
        # Split the adapted tensor into the expected surf variable keys.
        new_surf_vars = {}
        var_names = list(self.base_model.surf_vars)  # e.g., ("2t", "10u", "10v", "msl")
        for i, name in enumerate(var_names):
            # Extract the i-th channel for each time step and remove the channel dimension.
            # Resulting shape: (B, T, H, W)
            new_surf_vars[name] = adapted[:, :, i : i + 1, :, :].squeeze(2)
        # Replace the original surf_vars with our new ones.
        batch = dataclasses.replace(batch, surf_vars=new_surf_vars)

        # Now, normalization expects surf_vars to have shape (B, T, H, W)
        batch = batch.normalise(surf_stats=self.base_model.surf_stats)
        batch = batch.crop(patch_size=self.base_model.patch_size)

        # Expand static_vars appropriately.
        B, T = next(iter(batch.surf_vars.values())).shape[:2]

        batch = dataclasses.replace(
            batch,
            static_vars={
                k: v.unsqueeze(0).unsqueeze(0).expand(B, T, *v.shape)
                for k, v in batch.static_vars.items()
            },
        )

        # expanded_static = {}
        # for k, v in batch.static_vars.items():
        #     # v should originally be of shape (H, W); unsqueeze to (1,1,H,W) and expand to (B, T, H, W)
        #     expanded = v.unsqueeze(0).unsqueeze(0).expand(B, T, *v.shape)
        #     print("origianl shape | expanded shape",v.shape, expanded.shape)
        #     # If the expanded tensor has an extra dimension (i.e., 6D) and dimension 3 equals patch_size, squeeze it.
        #     if expanded.dim() == 6 and expanded.size(3) == self.base_model.patch_size:
        #         expanded = expanded.squeeze(3)
        #     expanded_static[k] = expanded
        # batch = dataclasses.replace(batch, static_vars=expanded_static)

        # batch = dataclasses.replace(
        #     batch,
        #     static_vars={k: v.unsqueeze(0).unsqueeze(0).expand(B, T, *v.shape) for k, v in batch.static_vars.items()},
        # )
        # batch = dataclasses.replace(
        #     batch,
        #     static_vars={k: v[None, None].repeat(B, T, 1, 1) for k, v in batch.static_vars.items()},
        # )

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
            new_output = self.new_head(x)
            return new_output
        else:
            original_output = self.base_model.decoder(
                x,
                batch,
                lead_time=self.base_model.timestep,
                patch_res=patch_res,
            )
            return original_output


# For reference, AuroraSmall is defined as in the original script:
AuroraSmall = partial(  #
    AuroraModified,
    encoder_depths=(2, 6, 2),
    encoder_num_heads=(4, 8, 16),
    decoder_depths=(2, 6, 2),
    decoder_num_heads=(16, 8, 4),
    embed_dim=256,
    num_heads=8,
    use_lora=False,
)
