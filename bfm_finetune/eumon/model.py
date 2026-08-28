"""Build BioAnalyst from config, load the released checkpoint, expose latents.

The released weights differ from the checked-in config in ways that are load-bearing, so
they are asserted rather than trusted: ``patch_size`` is 8 (not the config's 4), the
Lightning module is ``BFM`` built by ``model_helpers.setup_bfm_model``, and Hugging Face
ships a bare state dict, loaded into an explicitly constructed model. The single checkpoint
behind every reported number is ``bfm-pretrain-large``.
"""

import contextlib
import io
from pathlib import Path
from typing import Any

import torch

from .common.runner import project_root, sha256_file

CONFIG_PATH = "bfm-model/bfm_model/bfm/configs/train_config.yaml"
CHECKPOINT = "weights/bfm-pretrain-large.ckpt"
SAFETENSORS = "weights/bfm-pretrain-large.safetensors"

CHECKPOINT_FACTS = {
    "patch_size": 8,
    "embed_dim": 512,
    "species_num": 28,
    "H": 160,
    "W": 280,
    "backbone": "swin",
    "n_tensors": 561,
    "n_elements": 710_202_116,
    "sha256_safetensors": "8bbe0db77575fe2a2f84ab78120a6c945da9842a75538fe5d868c4a4c5a8e466",
    "spatial_patches": 700,
    "expected_encoder_tokens": 16100,
    "note": "spatial patches = (160/8)*(280/8) = 700; the encoder emits 23 token groups per "
            "patch (10 single-value groups + 13 atmospheric levels), so the latent is "
            "[B, 16100, 512]. Verified by strict load: at patch size 8 the checkpoint loads "
            "with zero missing and zero unexpected keys; at 4 it fails on "
            "encoder.surface_latents.",
}


class CheckpointMismatch(RuntimeError):
    pass


def load_config(patch_size: int | None = None, root: Path | None = None):
    from omegaconf import OmegaConf

    root = root or project_root()
    cfg = OmegaConf.load(root / CONFIG_PATH)
    cfg.model.patch_size = patch_size if patch_size is not None else CHECKPOINT_FACTS["patch_size"]
    return cfg


def load_state_dict(checkpoint: str | Path | None = None, root: Path | None = None) -> dict[str, torch.Tensor]:
    root = root or project_root()
    path = Path(checkpoint) if checkpoint else root / CHECKPOINT
    blob = torch.load(path, weights_only=True, map_location="cpu")
    return blob["state_dict"] if "state_dict" in blob else blob


def build(checkpoint: str | Path | None = None, *, device: str = "cpu", cfg: Any = None,
          strict: bool = True, root: Path | None = None, verify_facts: bool = True):
    """Construct BFM and load the released weights. Returns ``(model, info)``."""
    from bfm_model.bfm.model_helpers import setup_bfm_model

    root = root or project_root()
    cfg = cfg if cfg is not None else load_config(root=root)
    sd = load_state_dict(checkpoint, root=root)

    with contextlib.redirect_stdout(io.StringIO()):
        model = setup_bfm_model(cfg, mode="test")
    result = model.load_state_dict(sd, strict=strict)

    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    if strict and (missing or unexpected):
        raise CheckpointMismatch(f"missing={missing[:5]} unexpected={unexpected[:5]}")

    if verify_facts:
        if int(cfg.model.patch_size) != CHECKPOINT_FACTS["patch_size"]:
            raise CheckpointMismatch(
                f"patch_size {cfg.model.patch_size} contradicts the verified checkpoint "
                f"geometry ({CHECKPOINT_FACTS['patch_size']}); a wrong patch size changes "
                "the token grid and silently misplaces every gathered cell")
        n = sum(v.numel() for v in sd.values())
        if n != CHECKPOINT_FACTS["n_elements"]:
            raise CheckpointMismatch(f"checkpoint has {n:,} elements, expected "
                                     f"{CHECKPOINT_FACTS['n_elements']:,}")

    model = model.to(device).eval()
    info = {
        "checkpoint": str(Path(checkpoint) if checkpoint else root / CHECKPOINT),
        "device": device, "strict": strict,
        "n_tensors_in_checkpoint": len(sd),
        "n_elements_in_checkpoint": int(sum(v.numel() for v in sd.values())),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "n_buffers": int(sum(b.numel() for b in model.buffers())),
        "missing_keys": missing, "unexpected_keys": unexpected,
        "patch_size": int(cfg.model.patch_size), "embed_dim": int(cfg.model.embed_dim),
        "species_num": int(cfg.data.species_number),
        "H": int(cfg.model.H), "W": int(cfg.model.W),
        "backbone": str(cfg.model.backbone),
        "species_vars": list(cfg.model.species_vars),
    }
    return model, info


def forward_with_latents(model, batch, lead_time: int | None = None, batch_size: int = 1):
    """One forward pass, returning encoder latent, backbone latent and decoded fields.

    Deliberately not wrapped in ``torch.no_grad``: L3 backpropagates through this call to
    the adapters, and a decorator here silently severs that path while the head still
    trains. Inference callers wrap it themselves.
    """
    lead_time = lead_time if lead_time is not None else getattr(model, "lead_time", 2)
    encoded = model.encoder(batch, lead_time, batch_size)
    nh = model.H // model.encoder.patch_size
    nw = model.W // model.encoder.patch_size
    depth = encoded.shape[1] // (nh * nw)
    backbone_output = model.backbone(encoded, lead_time=lead_time, rollout_step=0,
                                     patch_shape=(depth, nh, nw))
    decoded = model.decoder(backbone_output, batch, lead_time)
    return {"encoded": encoded, "backbone_output": backbone_output, "decoded": decoded,
            "patch_grid": (nh, nw), "tokens_per_patch": depth}


def checkpoint_provenance(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    out: dict[str, Any] = {"facts": CHECKPOINT_FACTS}
    for key, rel in (("safetensors", SAFETENSORS), ("ckpt", CHECKPOINT)):
        path = root / rel
        if path.exists():
            out[key] = {"path": str(path), "bytes": path.stat().st_size,
                        "sha256": sha256_file(path)}
    return out
