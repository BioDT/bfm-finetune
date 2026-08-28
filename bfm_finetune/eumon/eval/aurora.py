"""Aurora's decoded fields sampled at grid cells — the L2 counterpart of BioAnalyst's.

Aurora's decoder does not speak the tasks' species vocabulary, so it cannot run L0 or L1;
L2 is the only setting where the two models are on equal footing. Two correctness constraints:
Aurora's ``Batch.normalise`` expects raw ERA5 physical units, so BioCube's scaling must be
off (``aurora_dataset``, asserted by ``assert_physical_units``); and Aurora requires
latitude descending with longitude in [0, 360), so ``_grid_orientation`` computes the two
permutations once and every tensor, input and output alike, is carried through them.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .. import batch as B
from ..model import load_config
from ..panel import GRID, GridSpec

CHECKPOINT_REPO = "microsoft/aurora"
CHECKPOINT_NAME = "aurora-0.25-pretrained.ckpt"

SURF_VAR_MAP: dict[str, str] = {"2t": "t2m", "10u": "u10", "10v": "v10", "msl": "msl"}
STATIC_VAR_MAP: dict[str, str] = {"lsm": "lsm", "z": "z", "slt": "slt"}
ATMOS_VARS: tuple[str, ...] = ("z", "u", "v", "t", "q")

AURORA_FACTS: dict[str, Any] = {
    "variant": "AuroraPretrained",
    "repo": CHECKPOINT_REPO,
    "checkpoint": CHECKPOINT_NAME,
    "why_this_variant": "standard 0.25 degree pretrained Aurora; loads strictly from the "
                        "local cache and BioCube's grid is a multiple of its patch size",
    "resolution_deg": 0.25,
    "patch_size": 4,
    "n_parameters": 1_256_300_176,
    "surf_vars": tuple(SURF_VAR_MAP),
    "static_vars": tuple(STATIC_VAR_MAP),
    "atmos_vars": ATMOS_VARS,
    "biocube_surf_map": SURF_VAR_MAP,
    "biocube_static_map": STATIC_VAR_MAP,
    "biocube_atmos_map": "identity -- BioCube's atmospheric_variables group already uses "
                         "Aurora's own z/u/v/t/q names",
    "n_atmos_levels": 13,
    "atmos_levels_hpa_note": "read from each batch's batch_metadata.pressure_levels, not "
                             "hard-coded; 1000..50 hPa, a superset of Aurora's default 4",
    "n_output_fields": len(SURF_VAR_MAP) + len(ATMOS_VARS) * 13,
    "substitutions": "none -- every variable Aurora needs has an identically-named BioCube "
                     "counterpart; nothing is zero-filled",
    "checkpoint_revision_note": "default_checkpoint_revision does not match this cache's "
                                "refs/main; resolved offline via try_to_load_from_cache",
}


def _local_checkpoint_path() -> Path:
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo_id=CHECKPOINT_REPO, filename=CHECKPOINT_NAME)
    if not isinstance(path, str):
        raise FileNotFoundError(
            f"{CHECKPOINT_NAME} not found in the local HuggingFace cache for {CHECKPOINT_REPO}; "
            "this module never downloads weights")
    return Path(path)


def build(device: str = "cuda:2", variant: str = "pretrained") -> tuple[torch.nn.Module, dict[str, Any]]:
    """Construct Aurora and load the pretrained checkpoint strictly. Returns ``(model, info)``."""
    import aurora as aurora_pkg

    if variant != "pretrained":
        raise ValueError(f"unsupported variant {variant!r}; only 'pretrained' is implemented "
                         "(see AURORA_FACTS['why_this_variant'])")

    model = aurora_pkg.AuroraPretrained(use_lora=False)
    checkpoint_path = _local_checkpoint_path()
    model.load_checkpoint_local(str(checkpoint_path), strict=True)
    model = model.to(device).eval()

    info = {
        "variant": variant, "device": device, "checkpoint": str(checkpoint_path),
        "patch_size": model.patch_size,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "surf_vars": model.surf_vars, "static_vars": model.static_vars,
        "atmos_vars": model.atmos_vars, "strict_load": True,
    }
    return model, info


def _grid_orientation(grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Index permutations mapping our (lat ascending, lon ascending) grid to Aurora's
    (lat descending, lon in [0, 360) ascending) contract. See module docstring."""
    lat_perm = (np.arange(grid.H)[::-1] if grid.lat_ascending else np.arange(grid.H)).copy()
    lon_perm = np.argsort(np.mod(grid.lons, 360.0))
    return lat_perm, lon_perm


def _invert_perm(perm: np.ndarray) -> np.ndarray:
    return np.argsort(perm)


def aurora_dataset(cfg: Any = None, biocube_dir: str | Path | None = None):
    """BioCube loader in raw ERA5 physical units. Use for every Aurora call site.

    ``batch.make_dataset`` leaves min-max scaling on, which is right for BioAnalyst but wrong
    here: Aurora's ``Batch.normalise`` applies ERA5 statistics, so a scaled ``t2m`` of 0.37
    gets normalised a second time against a ~278 K prior. The config is deep-copied because
    callers share it with the BioAnalyst steps, which need scaling on.
    """
    import copy

    cfg = copy.deepcopy(cfg) if cfg is not None else load_config()
    cfg.data.scaling.enabled = False
    kwargs = {"biocube_dir": biocube_dir} if biocube_dir is not None else {}
    return B.make_dataset(cfg, **kwargs)


# Plausible ranges for the field MEAN over the European window: wide enough that no real
# month trips them, tight enough that a min-max scaled field cannot pass. A unit check, not
# a physics check.
PHYSICAL_MEAN_BOUNDS: dict[str, tuple[float, float]] = {
    "2t": (200.0, 350.0),          # K
    "msl": (5.0e4, 1.5e5),         # Pa
}


def assert_physical_units(surf_vars: dict[str, torch.Tensor]) -> None:
    """Fail loudly when Aurora is handed scaled inputs instead of physical units.

    Scaled inputs do not crash: Aurora's decoder denormalises the output back into a
    plausible numeric range, so the only visible symptom is that the fields lose their
    geography.
    """
    for name, (lo, hi) in PHYSICAL_MEAN_BOUNDS.items():
        if name not in surf_vars:
            continue
        value = float(np.nanmean(surf_vars[name].detach().cpu().numpy()))
        if not (lo <= value <= hi):
            raise ValueError(
                f"Aurora input {name!r} has mean {value:.4g}, outside the physical range "
                f"[{lo:g}, {hi:g}]. BioCube's min-max scaling is almost certainly still "
                f"enabled: Aurora's Batch.normalise expects raw ERA5 units and would "
                f"double-normalise these. Build the loader with "
                f"`aurora.aurora_dataset(cfg)` instead of `batch.make_dataset(cfg)`.")


def to_aurora_batch(biocube_sample: Any, grid: GridSpec = GRID) -> "aurora.Batch":  # noqa: F821
    """Build an ``aurora.Batch`` from one loaded BioCube ``(t-1, t)`` sample.

    ``biocube_sample`` is the object returned by ``batch.load_input`` (optionally passed
    through ``batch.sanitise`` first) -- a ``bfm_model.bfm.dataloader_monthly.Batch``
    namedtuple, not the model-collated form ``batch.collate_for_model`` produces (Aurora
    reads BioCube's raw group tensors directly and never touches BFM's encoder/metadata).
    """
    import aurora as aurora_pkg

    lat_perm, lon_perm = _grid_orientation(grid)
    lat_idx = torch.as_tensor(lat_perm, dtype=torch.long)
    lon_idx = torch.as_tensor(lon_perm, dtype=torch.long)

    def reorder(t: torch.Tensor) -> torch.Tensor:
        return t.index_select(-2, lat_idx).index_select(-1, lon_idx)

    sv = biocube_sample.surface_variables
    av = biocube_sample.atmospheric_variables
    meta = biocube_sample.batch_metadata

    surf_vars = {aurora_name: reorder(sv[bc_name].float())[None]
                for aurora_name, bc_name in SURF_VAR_MAP.items()}
    assert_physical_units(surf_vars)
    # Static fields are time-invariant in BioCube (verified: identical across the month
    # pair); Aurora's Batch contract wants them with no batch/time axis at all.
    static_vars = {aurora_name: reorder(sv[bc_name][-1].float())
                   for aurora_name, bc_name in STATIC_VAR_MAP.items()}
    atmos_vars = {name: reorder(av[name].float())[None] for name in ATMOS_VARS}

    levels = tuple(float(l) for l in meta.pressure_levels)
    lats = torch.as_tensor(grid.lats[lat_perm], dtype=torch.float32)
    lons = torch.as_tensor(np.mod(grid.lons[lon_perm], 360.0), dtype=torch.float32)
    ref_time = datetime.strptime(meta.timestamp[-1], "%Y-%m-%d %H:%M:%S")

    metadata = aurora_pkg.Metadata(lat=lats, lon=lons, time=(ref_time,), atmos_levels=levels)
    return aurora_pkg.Batch(surf_vars=surf_vars, static_vars=static_vars,
                            atmos_vars=atmos_vars, metadata=metadata)


@torch.no_grad()
def forward_fields(model: torch.nn.Module, aurora_batch: "aurora.Batch"  # noqa: F821
                   ) -> dict[str, np.ndarray]:
    """Run Aurora and return every decoded field as ``[H, W]`` on our grid (cell (0, 0) is
    latitude 32.0, longitude -25.0), keyed by a stable name (surf: ``2t``/``10u``/``10v``/
    ``msl``; atmos: ``{var}_{level}``, e.g. ``t_850``). Insertion order is deterministic:
    surf vars in the model's own ``surf_vars`` order, then atmos vars in the model's own
    ``atmos_vars`` order crossed with the batch's own level order.
    """
    from aurora.normalisation import level_to_str

    pred = model(aurora_batch)
    lat_perm, lon_perm = _grid_orientation(GRID)
    inv_lat, inv_lon = _invert_perm(lat_perm), _invert_perm(lon_perm)

    def restore(t: torch.Tensor) -> np.ndarray:
        arr = t.detach().to(torch.float32).cpu().numpy()
        return arr[inv_lat, :][:, inv_lon]

    fields: dict[str, np.ndarray] = {}
    for name in model.surf_vars:
        fields[name] = restore(pred.surf_vars[name][0, 0])
    levels = pred.metadata.atmos_levels
    for name in model.atmos_vars:
        for k, level in enumerate(levels):
            fields[f"{name}_{level_to_str(level)}"] = restore(pred.atmos_vars[name][0, 0, k])
    return fields


def cell_features(fields: dict[str, np.ndarray], cell_i: Sequence[int], cell_j: Sequence[int]
                  ) -> tuple[np.ndarray, list[str]]:
    """Sample every decoded field at the given cells. Column order is ``fields``'s own
    insertion order, which ``forward_fields`` always produces deterministically."""
    names = list(fields)
    if not names:
        raise ValueError("no decoded fields to sample")
    cols = [B.gather_cells(fields[name], cell_i, cell_j) for name in names]
    return np.stack(cols, axis=1), names


def decode_year(task: str, year: int, device: str, biocube_dir: str | Path | None = None
                ) -> tuple[np.ndarray, list[str]]:
    """Aurora's decoded-field feature matrix for every cell of ``panel.GRID``, for the same
    short-lead BioCube input ``batch.forecast_input`` selects for BioAnalyst on this
    (task, year). Row order is C order over (cell_i, cell_j), i.e. row ``i * GRID.W + j`` is
    cell ``(i, j)``. Rebuilds the model and reloads the BioCube sample on every call.
    """
    biocube_dir = Path(biocube_dir) if biocube_dir is not None else B.BIOCUBE_DIR
    model, _ = build(device=device)

    cfg = load_config()
    cfg.data.scaling.enabled = False
    dataset = B.make_dataset(cfg, biocube_dir=biocube_dir)

    info = B.forecast_input(task, year, biocube_dir)
    sample = B.load_input(info["path"], dataset)
    sample, _ = B.sanitise(sample)

    aurora_batch = to_aurora_batch(sample)
    fields = forward_fields(model, aurora_batch)

    ci = np.repeat(np.arange(GRID.H), GRID.W)
    cj = np.tile(np.arange(GRID.W), GRID.H)
    return cell_features(fields, ci, cj)
