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
from aurora.batch import Batch
from torch.utils.checkpoint import checkpoint

from bfm_finetune.lora_adapter import LoRAAdapter
from bfm_finetune.new_variable_decoder import (
    InputMapper,
    NewModalityEncoder,
    NewVariableHead,
    OutputMapper,
    VectorDecoder,
)


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
        if mod.__class__.__name__.startswith(
            ("SwinTransformerBlock", "Swin", "PatchMerging")
        ):
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
        # print("batch", x)
        encoded_input = self.encoder(x)
        # print("Encoder output", encoded_input.shape)
        with torch.inference_mode():
            aurora_output = self.base_model(encoded_input)
        # print("Aurora output", aurora_output.shape)
        decoded_aurora = self.decoder(aurora_output)
        # print("Decoder output", decoded_aurora)
        return decoded_aurora

class DWDeconvUp(nn.Module):
    """Depth-wise ConvTranspose2d that maps 152x320 -> 180x360."""
    def __init__(self, channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            channels, channels, kernel_size=(29, 41),
            stride=1, padding=0, output_padding=0,
            groups=channels, bias=False
        )
    def forward(self, x): return self.deconv(x)

class TemporalSpatialEncoder3(nn.Module):
    """
    (B,T,C,152,320) → (B,64800,512); works for T=1 or 2.
    """
    def __init__(self, n_species=500, n_timesteps=2, embed_dim=512):
        super().__init__()
        self.max_t = n_timesteps
        in_c = n_species * n_timesteps
        self.up = nn.Sequential(
            DWDeconvUp(in_c),
            nn.Conv2d(in_c, 256, 1, bias=False),
            nn.GroupNorm(8, 256), nn.SiLU(),
        )
        self.fuse    = nn.Sequential(
            DWConvBlock(256, 512),
            nn.Conv2d(512, embed_dim, 1, bias=False)
        )

    def forward(self, x):
        if x.size(1) < self.max_t:
            x = F.pad(x, (0,0,0,0,0,0,0,self.max_t-x.size(1)))
        B,T,C,H,W = x.shape
        x = x.view(B, T*C, H, W)                     # (B,1000,152,320)
        x = self.fuse(self.up(x))                    # (B,512,180,360)
        return x.flatten(2).transpose(1,2)           # (B,64 800,512)
    
def GN(ch, g=8):
    g = g if ch % g == 0 else 1 # fall back to Layer‑norm style
    return nn.GroupNorm(g, ch)

class DWDownProjectLite(nn.Module):
    """
    Light-weight learnable down-sample 180x360 ➜ 152->320
    ≈ 0.2 M params
    """
    def __init__(self, ch: int):
        super().__init__()
        self.stage = nn.Sequential(
            nn.Conv2d(ch, ch, 3, stride=2, padding=1, groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, bias=False),      # point‑wise fuse
            GN(ch), nn.SiLU(),

            nn.Upsample(size=(152, 320), mode='bilinear', align_corners=False),

            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, bias=False),
            GN(ch), nn.SiLU(),                     # keeps gradients alive
        )

    def forward(self, x): 
        return self.stage(x)

# VERY EXPENSIVE
class DWDownProject(nn.Module):
    """
    Learnable down-sample 180x360 -> 152x320 using a single depth-wise conv.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels,
            kernel_size=(29, 41),
            stride=1, padding=0,
            groups=channels, bias=False
        )
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)  # fuse species
        self.norm = GN(channels)
        self.act  = nn.SiLU()

    def forward(self, x):
        x = self.dw(x)
        return self.act(self.norm(self.pw(x)))
 
class TemporalSpatialDecoder3(nn.Module):
    """
    Input : (B, 64800, 1024)
    Output: (B, 1, 500, 152, 320)
    """
    def __init__(self, n_species: int = 500, in_dim: int = 1024):
        super().__init__()
        self.H, self.W = 180, 360
        self.pre = nn.Conv2d(in_dim, 512, 1, bias=False)

        self.up = nn.Sequential(
            DWConvBlock(512, 256),
            nn.PixelShuffle(2),     # (B,64,360,720)
            DWConvBlock(64, 64),
            nn.Conv2d(64, n_species, 1, bias=False)
        )

        self.down = nn.Sequential(
            # halve first so kernel stays small
            nn.Conv2d(n_species, n_species, 3, stride=2, padding=1,
                      groups=n_species, bias=False),      # 180×360
            DWDownProject(n_species),                    # 152×320
        )

    def forward(self, z):
        b, tok, d = z.shape
        z = z.transpose(1, 2).reshape(b, d, self.H, self.W)  # 180×360
        z = self.up(self.pre(z))                              # 360×720
        z = self.down(z)                                      # 152×320
        return z.unsqueeze(1)                                 # (B,1,500,152,320)


# class UpBlock(nn.Module):              # PixelShuffle + DW conv
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.conv = DWConvBlock(in_c, out_c * 4)
#         self.shuf = nn.PixelShuffle(2)           # rearrange, no params
#     def forward(self, x): return self.shuf(self.conv(x))

# class TemporalSpatialDecoder3(nn.Module):
#     """
#     (B,64800,1024) → (B,1,500,152,320)  learnable all the way.
#     """
#     def __init__(self, n_species=500, in_dim=1024):
#         super().__init__()
#         self.H, self.W = 180, 360
#         self.net = nn.Sequential(
#             nn.Conv2d(in_dim, 512, 1, bias=False),
#             DWConvBlock(512, 256),
#             UpBlock(256, 128),                     # 180×360 -> 360×720
#             DWConvBlock(128, 64),
#             nn.Conv2d(64, n_species, 1, bias=False)
#         )

#     def forward(self, z):
#         b,tok,d = z.shape
#         z = z.transpose(1,2).reshape(b,d,self.H,self.W)  # (B,1024,180,360)
#         z = self.net(z)                                  # (B,500,360,720)
#         z = F.adaptive_max_pool2d(z, (152, 320)) # exact downsample to 152×320
#         return z.unsqueeze(1)      



class DWConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.depth = nn.Conv2d(in_c, in_c, 3, padding=1,
                               groups=in_c, bias=False)
        self.point = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.norm  = nn.GroupNorm(8, out_c)
        self.act   = nn.SiLU()
    def forward(self, x): return self.act(self.norm(self.point(self.depth(x))))

class UpProject(nn.Module):
    """Learnable resize 152×320 → 180×360 via ConvTranspose2d."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(
            in_c, out_c,
            kernel_size=(29, 41),  # solved above
            stride=1, padding=0, output_padding=0, bias=False
        )
        self.norm  = nn.GroupNorm(8, out_c)
        self.act   = nn.SiLU()
    def forward(self, x): return self.act(self.norm(self.tconv(x)))

class TemporalSpatialEncoder2(nn.Module):
    """
    Input : (B, T∈{1,2}, 500, 152, 320)
    Output: (B, 64 800, 512)  # 180 × 360 tokens
    """
    def __init__(
        self,
        n_species:   int = 500,
        n_timesteps:       int = 2,     # works for T=1 or 2
        embed_dim:   int = 512
    ):
        super().__init__()
        in_c = n_species * n_timesteps             # 1000 if T=2 else 500
        self.max_t = n_timesteps
        self.up    = UpProject(in_c, 256)    # (B,256,180,360)
        self.fuse  = nn.Sequential(
            DWConvBlock(256, 512),
            nn.Conv2d(512, embed_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, C, 152, 320)
        if x.size(1) < self.max_t:                     # T=1 case
            x = F.pad(x, (0,0,0,0,0,0,0,self.max_t-x.size(1)))
        B,T,C,H,W = x.shape
        x = x.reshape(B, T*C, H, W)                    # (B,TxC,152,320)
        x = self.fuse(self.up(x))                      # (B,512,180,360)
        x = x.flatten(2).transpose(1, 2)               # (B,64 800,512)
        return x

class UpBlock(nn.Module):
    """PixelShuffle‑based learnable upsample ×2."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = DWConvBlock(in_c, out_c * 4)
        self.pix  = nn.PixelShuffle(2)
    def forward(self, x): return self.pix(self.conv(x))

class TemporalSpatialDecoder2(nn.Module):
    """
    Input : (B, 64 800, 1024)
    Output: (B,1,500,152,320)
    """
    def __init__(self, n_species: int = 500, in_dim: int = 1024):
        super().__init__()
        self.H, self.W = 180, 360
        self.decode = nn.Sequential(
            nn.Conv2d(in_dim, 512, 1, bias=False),   # (B,512,180,360)
            DWConvBlock(512, 256),
            UpBlock(256, 128),                       # (B,128,360,720)
            DWConvBlock(128, 64),
            nn.Conv2d(64, n_species, 1, bias=False)  # (B,500,360,720)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, tok, d = z.shape
        z = z.transpose(1, 2).reshape(b, d, self.H, self.W)
        z = self.decode(z)                           # (B,500,360,720)
        z = F.adaptive_max_pool2d(z, (152, 320))                        # exact downsample to 152×320
        return z.unsqueeze(1)                        # (B,1,500,152,320)


######---------- WORKING---------------
class TemporalSpatialEncoder(nn.Module):
    """
    Accepts x : (B, T=2, C=500, 152, 320)
    Returns (B, 259 200, 512)

    Strategy: merge time -> channel (simple, no blur), 1x1 projection, flatten.
    """

    def __init__(
        self,
        n_species: int = 500,
        n_timesteps: int = 2,
        embed_dim: int = 512,
        target_hw: Tuple[int, int] = (180, 360),  # (360, 720),
    ) -> None:
        super().__init__()
        in_channels = n_species * n_timesteps  # 1000
        self.target_hw = target_hw
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) → (B, T*C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b, t * c, h, w)  # no copy, just reshape
        x = F.interpolate(x, size=self.target_hw, mode="nearest")
        x = self.proj(x)  # (B, 512, 360, 720)
        # x = self.act(x)
        x = x.flatten(2).transpose(1, 2)  # (B, 259 200, 512)
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
        source_hw: Tuple[int, int] = (180, 360),  # (360, 720)
        final_hw: Tuple[int, int] = (152, 320),
    ) -> None:
        super().__init__()
        self.source_hw = source_hw
        self.final_hw = final_hw
        self.proj = nn.Conv2d(in_dim, n_species, kernel_size=1, bias=False)
        # self.act  = nn.Softplus(beta=1.)   # ensures output > 0
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, p, d = x.shape  # (B, 259 200, 1024)
        h, w = self.source_hw
        x = x.transpose(1, 2).reshape(b, d, h, w)  # (B, 1024, 360, 720)
        x = self.proj(x)  # (B, 500, 360, 720)
        # x = self.act(x)
        x = F.interpolate(x, size=self.final_hw, mode="nearest")  # (B, 500, 152, 320)
        return x.unsqueeze(1)  # (B, 1, 500, 152, 320)

######---------- WORKING---------------


def freeze_except_lora(model):
    """
    sets requires_grad=True  only for tensors whose name contains 'lora_'
    everything else is frozen
    returns list of trainable parameter names for sanity‑check
    """
    trainable = []
    for name, param in model.named_parameters():
        if "lora_" in name:  # catches lora_matrix_A / B, alpha, etc.
            param.requires_grad = True
            trainable.append(name)
        else:
            param.requires_grad = False
    print(f"[LoRA]  trainable params = {len(trainable)} layers")
    return trainable


###### MASKED TRYOUT #####
class MaskedSpeciesEncoder(nn.Module):
    """
    Spatial-only encoder with Prithvi-style masking.
    Input : (B, 500, 152, 320)
    Output: tokens (B, 64800, 512), idx_masked, idx_visible
    """
    def __init__(self, n_sp=500, embed=512,
                 grid_hw=(180,360),
                 mask_mode='both',
                 mask_ratio=0.9):
        super().__init__()
        self.H, self.W = grid_hw
        self.embed = nn.Conv2d(n_sp, embed, 1, bias=False)
        self.pos = nn.Parameter(torch.randn(1, embed, self.H, self.W))
        self.mask_token = nn.Parameter(torch.zeros(1,1,embed))
        self.mask_mode = mask_mode.lower()
        self.mask_ratio = mask_ratio
        # build indices once to avoid realloc
        g = torch.arange(self.H*self.W) # 0 … 64 799
        self.register_buffer('grid_idx', g)

    def _sample_mask(self, B):
        n_total = self.H*self.W
        n_mask  = int(n_total*self.mask_ratio)
        shuffle = self.grid_idx.repeat(B,1)
        idx = shuffle[ torch.arange(B).unsqueeze(1),
                       torch.randperm(n_total) ]
        return idx[:,:n_mask], idx[:,n_mask:]         # masked, visible

    def forward(self, x):
        B, C, H, W = x.shape                         # 500,152,320
        x = F.interpolate(x, size=(self.H,self.W), mode='nearest')
        x = self.embed(x) + self.pos                # (B,512,H,W)
        x = x.flatten(2).transpose(1,2)             # (B,T=64800,512)
        idx_mask, idx_vis = self._sample_mask(B)
        mask_expand = idx_mask.unsqueeze(-1).expand(-1,-1,x.size(-1))
        vis_expand  = idx_vis .unsqueeze(-1).expand(-1,-1,x.size(-1))
        x_visible   = torch.gather(x, 1, vis_expand)     # (B,T_vis,512)
        # Replace masked positions with learned mask token for decoder
        x_masktok   = self.mask_token.expand(B, idx_mask.size(1), -1)
        x_tokens = torch.cat([x_visible, x_masktok], dim=1)  # order doesn’t matter for backbone
        return x_tokens, idx_mask, idx_vis


class MaskedSpeciesDecoder(nn.Module):
    """
    Reconstruct masked tokens -> full raster.
    Input : feats (B, 64800, 1024), idx_masked/visible
    Output: (B, 1, 500, 152, 320)
    """
    def __init__(self, n_sp=500, in_dim=1024, hid=512, grid_hw=(180,360)):
        super().__init__()
        self.n_sp = n_sp
        self.H,self.W = grid_hw
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Linear(hid, n_sp),   
            # nn.Softplus()    # >=0 predictions
        )
        self.projector = DWDownProjectLite(self.n_sp)

    def forward(self, feats, idx_mask, idx_vis):
        B = feats.size(0)
        # feats are in same order we fed backbone (visibles first)
        n_vis = idx_vis.size(1)
        vis_pred = feats[:, :n_vis]
        mask_pred= feats[:, n_vis:]
        y_pred = torch.empty(B, self.H*self.W, feats.size(-1), device=feats.device)
        y_pred.scatter_(1, idx_vis.unsqueeze(-1).expand_as(vis_pred), vis_pred)
        y_pred.scatter_(1, idx_mask.unsqueeze(-1).expand_as(mask_pred), mask_pred)
        y_img = self.mlp(y_pred).transpose(1,2).reshape(B, -1, self.H, self.W) # (B,500,H,W)
        # y_img = y_img[..., :152, :320] # crop
        y_img = self.projector(y_img)     # 180×360 -> 152×320
        return y_img.unsqueeze(1) # (B,1,500,152,320)
    


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
        self.base_model.encoder = nn.Identity()
        self.base_model.decoder = nn.Identity()
        # V1: Spatiotemporal
        # self.encoder = TemporalSpatialEncoder(n_species=n_species, n_timesteps=2)
        # self.decoder = TemporalSpatialDecoder(n_species=n_species)
        # V2: Spatial => Masking
        self.encoder = MaskedSpeciesEncoder(n_sp=n_species)
        self.decoder = MaskedSpeciesDecoder(n_sp=n_species)

        freeze_except_lora(base_model)  #

        # self.patch_res = (
        #     4,
        #     180,
        #     360,
        # ) = 259200
        self.patch_res = (
            4,
            90,
            180,
        ) # = 64800
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{trainable/1e6:.2f} M / {total/1e6:.2f} M parameters will update")

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        `batch` must be shape (B, T, N_species, 152, 320) with dtype float32/16.
        """
        # x = batch["species_distribution"][:, 0, :, :, :].unsqueeze(0) # use only t0, t1 is the target
        # x = batch["species_distribution"] # for spatiotemporal
        x = batch["species_distribution"][:, 0, :, :, :] # for spatial
        # print("Forward x shape", x.shape)
        tokens, idx_mask, idx_vis = self.encoder(x)  # (B, 259200, 512) | For spatial
        # tokens = self.encoder(x) # For spatiotemporal
        # print("Tokens shape", tokens.shape)
        feats = self.base_model.backbone(
            tokens,
            lead_time=timedelta(hours=6.0),
            patch_res=self.patch_res,
            rollout_step=1,
        )  # (B, 259200, 1024)
        recon = self.decoder(feats, idx_mask, idx_vis)  # (B, C, 152, 320) | For spatial
        # recon = self.decoder(feats) # For spatiotemporal
        return recon


class MaskRatioScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linearly ramps encoder.mask_ratio from start_ratio (warm-up) to
    final_ratio between epoch start_epoch and the last epoch.
    """
    def __init__(self, encoder, start_ratio=0.5, final_ratio=0.9,
                 start_epoch=5, total_epochs=100, last_epoch=-1):
        self.encoder = encoder
        self.start_ratio = start_ratio
        self.final_ratio = final_ratio
        self.start_epoch = start_epoch
        self.total_epochs = total_epochs
        self.last_epoch = last_epoch
        # super().__init__(optimizer=None, last_epoch=last_epoch)

    def get_lr(self):
        return []

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch < self.start_epoch:
            ratio = self.start_ratio
        else:
            frac  = (epoch - self.start_epoch) / max(1, self.total_epochs - self.start_epoch)
            ratio = self.start_ratio + frac * (self.final_ratio - self.start_ratio)
        ratio = max(0.0, min(1.0, ratio))
        self.encoder.mask_ratio = ratio