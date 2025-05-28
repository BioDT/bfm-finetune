import contextlib
import dataclasses
import types
from datetime import timedelta
from functools import partial
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.utils.checkpoint import checkpoint

# TODO: move these general to a separate file
from bfm_finetune.aurora_mod import (
    TemporalSpatialDecoder,
    TemporalSpatialEncoder,
    freeze_except_lora,
)
from bfm_finetune.lora_adapter import LoRAAdapter
from bfm_finetune.new_variable_decoder import (
    InputMapper,
    NewModalityEncoder,
    NewVariableHead,
    OutputMapper,
    VectorDecoder,
)


class BFMRaw(nn.Module):
    """Same as AuroraRaw but for the BFM"""

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

        self.encoder = TemporalSpatialEncoder(n_species=n_species, embed_dim=256)
        self.decoder = TemporalSpatialDecoder(n_species=n_species, in_dim=256)

        freeze_except_lora(base_model)  #
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{trainable/1e6:.2f} M / {total/1e6:.2f} M parameters will update")

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        `batch` must be shape (B, N_species, 152, 320) with dtype float32/16.
        """
        x = batch["species_distribution"]
        encoded = self.encoder(x)
        patch_shape = (
            4,
            90,
            180,
        )
        # print("patch_shape", patch_shape)
        print(encoded.shape)
        feats = self.base_model.backbone(
            encoded, lead_time=2, rollout_step=0, patch_shape=patch_shape
        )
        print(feats.shape)
        recon = self.decoder(feats)
        return recon
