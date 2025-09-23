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

from bfm_finetune.aurora_mod import TemporalSpatialDecoder, TemporalSpatialEncoder


def freeze_except_lora(model):
    """
    sets requires_grad=True  only for tensors whose name contains 'lora_' or "vera_"
    everything else is frozen
    returns list of trainable parameter names for sanity-check
    """
    trainable = []
    for name, param in model.named_parameters():
        if "lora_" or "vera_" in name:  # catches lora_matrix_A / B, alpha, etc.
            param.requires_grad = True
            trainable.append(name)
        else:
            param.requires_grad = False
    print(f"[LoRA] or VeRA trainable params = {len(trainable)} layers")
    return trainable


class TemporalSpatialEncoder_64k(nn.Module):
    """
    Input (x) : (B, 2, 500, 160, 280)
    Output (tokens) : (B, 64400, 256)
    """

    def __init__(
        self,
        n_species: int = 500,
        n_timesteps: int = 2,
        embed_dim: int = 256,
        target_hw: Tuple[int, int] = (161, 400),
    ):
        super().__init__()
        in_c = n_species * n_timesteps
        self.H, self.W = target_hw
        self.proj = nn.Conv2d(in_c, embed_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H0, W0 = x.shape
        x = x.view(B, T * C, H0, W0)
        x = F.interpolate(x, size=(self.H, self.W), mode="nearest")
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TemporalSpatialDecoder_64k(nn.Module):
    """
    Input (tokens) : (B, 64400, 256)
    Output (y) : (B, 1, 500, 160, 280)
    """

    def __init__(
        self,
        n_species: int = 500,
        in_dim: int = 256,
        source_hw: Tuple[int, int] = (161, 400),
        final_hw: Tuple[int, int] = (160, 280),
    ):
        super().__init__()
        self.H, self.W = source_hw
        self.tH, self.tW = final_hw
        self.proj = nn.Conv2d(in_dim, n_species, 1, bias=False)
        self.post = nn.Conv2d(n_species, n_species, 1, bias=False)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        B, L, D = tok.shape
        assert L == self.H * self.W, f"expected {self.H*self.W} tokens, got {L}"
        x = tok.transpose(1, 2).reshape(B, D, self.H, self.W)
        x = self.proj(x)
        x = F.interpolate(
            x, size=(self.tH, self.tW), mode="bilinear", align_corners=False
        )
        x = self.post(x)
        return x.unsqueeze(1)


class LatentPerceiverEncoder(nn.Module):
    """
    Input : (B, T=2, 500, 160, 280)
    Output: (B, 64_400, 256)
    """

    def __init__(
        self,
        n_species: int = 500,
        n_timesteps: int = 2,
        embed_dim: int = 256,
        grid_hw: Tuple[int, int] = (160, 280),
        latent_tokens: int = 512,
        n_heads: int = 8,
    ):
        super().__init__()
        H, W = grid_hw
        in_c = n_species * n_timesteps

        # 1×1 projection to embedding space
        self.stem = nn.Conv2d(in_c, embed_dim, 1, bias=False)

        # fixed 6-D sine–cos positional grid -> 256-D via 1×1
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
        )
        pos = torch.stack(
            (
                x,
                y,
                torch.sin(torch.pi * x),
                torch.cos(torch.pi * x),
                torch.sin(torch.pi * y),
                torch.cos(torch.pi * y),
            ),
            0,
        )  # (6,H,W)
        self.register_buffer("pos_raw", pos.unsqueeze(0))
        self.pos_proj = nn.Conv2d(6, embed_dim, 1, bias=False)

        # latent array + cross-attention
        self.latent = nn.Parameter(torch.randn(1, latent_tokens, embed_dim))
        self.cross_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.cross_kv_grid = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.cross_kv_lat = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns full spatial token grid (B, 64400, 256)
        """
        B, T, C, H0, W0 = x.shape
        x = x.view(B, T * C, H0, W0)
        x = self.stem(x)  # (B,256,160,280)

        # add fixed positional encoding
        pos = self.pos_proj(self.pos_raw)  # (1,256,160,280)
        x = x + pos

        # flatten to sequence  (grid tokens)
        x_tok = x.flatten(2).transpose(1, 2)  # (B, 64 400, 256)

        lat = self.latent.expand(B, -1, -1)  # (B,512,256)

        # grid ->latent
        k_g, v_g = self.cross_kv_grid(x_tok).chunk(2, dim=-1)  # (B,64k,256)
        q_l = self.cross_q(lat)  # (B,512,256)
        lat = self.cross_attn(q_l, k_g, v_g, need_weights=False)[0]

        # latent -> grid (reconstruction tokens)
        k_l, v_l = self.cross_kv_lat(lat).chunk(2, dim=-1)  # (B,512,256)
        rec = self.cross_attn(x_tok, k_l, v_l, need_weights=False)[0]  # (B,64k,256)
        return rec  # (B,64400,256)


class LatentPerceiverDecoder(nn.Module):
    """
    Input : Tokens (B, 64400, 256)
    Output: y_target  (B, 1, 500, 160, 280)
    """

    def __init__(self, n_species: int = 500, embed_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_species),
            nn.ReLU(),  # nn.Softplus() # ensures y_target >= 0
        )

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        B, L, E = tok.shape  # L must be 64400
        assert L == 160 * 280, "Token count mismatch"
        y = self.mlp(tok)  # (B,64400,500)
        y = y.transpose(1, 2).reshape(B, 500, 160, 280)
        return y.unsqueeze(1)  # (B,1,500,160,280)


def GN(ch, g=8):
    g = g if ch % g == 0 else 1
    return nn.GroupNorm(g, ch)


class DWBlock(nn.Module):
    """
    Depth-Wise separable ConvNeXt block with optional down-sampling.

    - in_ch == out_ch  (ConvNeXt style)
    - if stride == 2 the first depth-wise conv does the spatial reduction
    - kernel 3x3 for locality, point-wise 1x1 for channel mixing
    - residual + GELU -> stable with few samples
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expansion: int = 4):
        super().__init__()
        assert out_ch % expansion == 0
        mid = out_ch // expansion

        # depth-wise conv (with stride for down-sampling)
        self.dw = nn.Conv2d(
            in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        # point-wise bottleneck
        self.pw1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.pw2 = nn.Conv2d(mid, out_ch, 1, bias=False)

        self.norm = GN(in_ch)
        self.act = nn.GELU()

        # For stride-2, residual needs a matching down-sample
        self.skip = (
            nn.Identity()
            if stride == 1 and in_ch == out_ch
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), GN(out_ch)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.skip(x)
        x = self.dw(x)
        x = self.act(self.norm(x))
        x = self.pw2(self.act(self.pw1(x)))
        return x + res


class ConvFormerEncoder(nn.Module):
    """
    Hybrid: depth-wise ConvNeXt stem + SegFormer hierarchical tokens.
    Returns 44800 = (160x280)
    """

    def __init__(self, n_sp=500, t=2, emb=256, grid_hw=(160, 280)):
        super().__init__()
        in_c = n_sp * t
        self.stem = nn.Sequential(  # ConvNeXt-style
            nn.Conv2d(in_c, 64, 4, stride=4, padding=0, bias=False),
            nn.GroupNorm(4, 64),
            nn.GELU(),
        )  # 40×70
        self.stage1 = DWBlock(64, 128)  # 40×70
        self.stage2 = DWBlock(128, emb, stride=2)  # 20×35
        self.upsamp = nn.Upsample(size=grid_hw, mode="bilinear")
        self.pos = nn.Parameter(torch.zeros(1, *grid_hw, emb))

    def forward(self, x):
        B, T, C, H0, W0 = x.shape
        x = x.view(B, T * C, H0, W0)
        x = self.upsamp(self.stage2(self.stage1(self.stem(x))))
        x = x + self.pos.permute(0, 3, 1, 2)
        return x.flatten(2).transpose(1, 2)  # (B,64400,256)


class LightSegFormerHead(nn.Module):
    """
    MLP-decoder à la SegFormer - no upsampling needed.
    """

    def __init__(self, n_sp=500, emb=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, n_sp), nn.Softplus()
        )

    def forward(self, tok):
        B, L, E = tok.shape
        out = self.proj(tok).transpose(1, 2).reshape(B, 500, 160, 280)
        return out.unsqueeze(1)


class ConvFormerEncoder1(nn.Module):
    """
    Depth-wise ConvNeXt stem + SegFormer hierarchy
    Output grid now 161 x 400 -> 64 400 tokens x 256-D
    """

    def __init__(self, n_sp=500, t=2, embed_dim=256, grid_hw=(161, 400)):
        super().__init__()
        self.H, self.W = grid_hw
        in_c = n_sp * t

        # ConvNeXt-style stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, 64, 4, stride=4, bias=False), GN(64), nn.GELU()  # 40 × 70
        )
        # Hierarchical DW blocks
        self.stage1 = DWBlock(64, 128, stride=1)  # 40 × 70
        self.stage2 = DWBlock(128, embed_dim, stride=2)  # 20 × 35

        # Learnable up-project to 161 × 400
        self.upsamp = nn.Sequential(
            nn.Upsample(size=grid_hw, mode="bilinear", align_corners=False),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),  # channel adjust
        )
        # Frozen absolute positional tensor in (C,H,W) layout
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W), indexing="ij"
        )
        pos = torch.stack(
            (
                x,
                y,
                torch.sin(torch.pi * x),
                torch.cos(torch.pi * x),
                torch.sin(torch.pi * y),
                torch.cos(torch.pi * y),
            ),
            0,
        )
        self.register_buffer("pos", pos.unsqueeze(0))  # (1,6,H,W)
        self.pos_proj = nn.Conv2d(6, embed_dim, 1, bias=False)

    def forward(self, x):
        B, T, C, H, W = x.shape  # (B,2,500,160,280)
        x = x.reshape(B, T * C, H, W)  # (B,1000,160,280)
        x = self.stem(x)  # 40×70
        x = self.stage2(self.stage1(x))  # 20×35
        x = self.upsamp(x)  # 161×400, 256-C

        x = x + self.pos_proj(self.pos)  # add abs. positions
        return x.flatten(2).transpose(1, 2)  # (B, 64 400, 256)


class LightSegFormerHead1(nn.Module):
    """
    Token-wise MLP head + learnable resize.
    Input : (B, 64 400, 256)
    Output: (B, 1, 500, 160, 280)
    """

    def __init__(
        self,
        n_species: int = 500,
        embed_dim: int = 256,
        grid_hw: Tuple[int, int] = (161, 400),
        target_hw: Tuple[int, int] = (160, 280),
    ):
        super().__init__()
        self.H, self.W = grid_hw
        self.tH, self.tW = target_hw

        # token-wise mixing (≈ 0.19 M params)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_species),
            nn.ReLU(),
            # nn.Softplus()
        )

        # lightweight, learnable resize 161x400 -> 160×280
        #    depth-wise 3x3 with stride=(161//160, 400//280) = (1, ❰≈1.4❱) impossible,
        #    so we:  a) reshape -> feature map,
        #            b) bilinear interpolate (no params, no artefacts),
        #            c) 1x1 conv to let model adjust any interpolation bias.
        self.post_conv = nn.Conv2d(n_species, n_species, 1, bias=False)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        B, L, E = tok.shape
        assert L == self.H * self.W, f"expected {self.H*self.W} tokens, got {L}."

        y = self.mlp(tok)  # (B, 64400, 500)
        y = y.transpose(1, 2).reshape(B, 500, self.H, self.W)  # (B,500,161,400)

        #  learn-nothing interpolation, then 1x1 fuse
        y = F.interpolate(
            y, size=(self.tH, self.tW), mode="bilinear", align_corners=False
        )
        y = self.post_conv(y)  # fine adjust

        return y.unsqueeze(1)  # (B,1,500,160,280)


class SE(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // r, 1),
            nn.GELU(),
            nn.Conv2d(ch // r, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class DWSE(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.se = SE(ch)
        self.act = nn.GELU()
        self.norm = GN(ch)

    def forward(self, x):
        y = self.pw(self.act(self.dw(x)))
        return self.norm(self.se(y) + x)


# ---------- Encoder V3 -------------------------------------------------------
class TemporalSpatialEncoderV3(nn.Module):
    """
    In :  (B, 2, 500, 160, 280)  *or*  (B, 500, 160, 280)
    Out:  (B, 48 800, 256)
    """

    def __init__(
        self,
        n_species: int = 500,
        embed_dim: int = 256,
        grid_hw: Tuple[int, int] = (160, 305),
    ):  # 160*305 = 48 800
        super().__init__()
        self.H, self.W = grid_hw
        self.stem = nn.Conv2d(n_species * 2, 256, 1, bias=False)  # expect T merged
        self.block = nn.Sequential(DWSE(256), DWSE(256))
        self.proj = nn.Conv2d(256, embed_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # merge time dim if present → (B,1000,160,280)
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(
                B, T * C, H, W
            )  # merge time dim (✔)  :contentReference[oaicite:1]{index=1}
        elif x.ndim == 4:
            pass
        else:
            raise ValueError("Input must be 4-D or 5-D tensor.")

        x = self.block(self.stem(x))  # (B,256,160,280)
        x = F.interpolate(x, size=(self.H, self.W), mode="nearest")
        x = self.proj(x)  # (B,256,160,305)
        return x.flatten(2).transpose(1, 2)  # (B,48 800,256)


# ---------- Decoder V3 -------------------------------------------------------
class TemporalSpatialDecoderV3(nn.Module):
    """
    tokens : (B, 48 800, 256) ➜ y : (B,1,500,160,280)
    """

    def __init__(
        self,
        n_species=500,
        embed_dim=256,
        grid_hw: Tuple[int, int] = (160, 305),
        target_hw: Tuple[int, int] = (160, 280),
    ):
        super().__init__()
        self.H, self.W = grid_hw
        self.tH, self.tW = target_hw
        self.path = nn.Sequential(DWSE(embed_dim), DWSE(embed_dim), DWSE(embed_dim))
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, n_species, 1, bias=False), nn.Softplus()
        )

    def forward(self, tok):
        B, L, E = tok.shape
        assert L == self.H * self.W, "token count mismatch"
        x = tok.transpose(1, 2).reshape(B, E, self.H, self.W)  # (B,256,160,305)
        x = self.path(x)
        x = F.interpolate(
            x, size=(self.tH, self.tW), mode="bilinear", align_corners=False
        )  # (B,256,160,280)
        x = self.head(x)  # (B,500,160,280)
        return x.unsqueeze(1)  # (B,1,500,160,280)


class BFMRaw(nn.Module):
    """Same as AuroraRaw but for the BFM"""

    def __init__(
        self,
        base_model,
        n_species: int = 500,
        freeze_backbone: bool = True,
        mode: str = "train",
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.mode = mode

        # enable_swin3d_checkpointing(self.base_model.backbone)

        self.base_model.encoder = nn.Identity()
        self.base_model.decoder = nn.Identity()

        self.encoder = TemporalSpatialEncoder(n_species=n_species, embed_dim=512, target_hw=(115, 140))
        self.decoder = TemporalSpatialDecoder(n_species=n_species, in_dim=512, source_hw=(115, 140))
        # self.encoder = LatentPerceiverEncoder()
        # self.decoder = LatentPerceiverDecoder()
        # self.encoder = ConvFormerEncoder()
        # self.decoder = LightSegFormerHead()
        # self.encoder = ConvFormerEncoder1()
        # self.decoder = LightSegFormerHead1()
        # self.encoder = TemporalSpatialEncoderV3()
        # self.decoder = TemporalSpatialDecoderV3()

        if freeze_backbone:
            freeze_except_lora(base_model)  #
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{trainable/1e6:.2f} M / {total/1e6:.2f} M parameters will update")

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        `batch` must be shape (B, N_species, 160, 280) with dtype float32/16.
        """
        x = batch["species_distribution"]
        encoded = self.encoder(x)
        # print(f"encoded shape: {encoded.shape}")
        # 64400
        # patch_shape = (
        #     23,
        #     70,
        #     40,
        # )

        # 48800 # Works with weights v1
        # encoded shape: torch.Size([1, 44800, 512])
        # patch_shape = (
        #     16,
        #     70,
        #     40,
        # )

        # V2
        # 700 patches
        # 161000 tokens
        # (23, 20, 35)
        patch_shape = (23, 20, 35)
        # print("patch_shape", patch_shape)
        # print(f"Encoded shape: {encoded.shape}")
        feats = self.base_model.backbone(
            encoded, lead_time=1, rollout_step=0, patch_shape=patch_shape
        )
        # print(f"latents shape {feats.shape}")
        recon = self.decoder(feats)
        # print(f"decoded shape {recon.shape}")
        if self.mode == "train":
            return recon
        else:  # you are doing eval and maybe need the latents
            return recon, encoded, feats
