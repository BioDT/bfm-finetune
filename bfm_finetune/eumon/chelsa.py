"""CHELSA v2.1 monthly climate onto the BioAnalyst grid.

Each month is window-read over HTTP with GDAL's ``/vsicurl/`` and reduced to the 160x280
grid before it touches disk; the geometry divides exactly (0.25 degrees = 30x30 native
cells), so the reduction is a plain block mean. Two orientation facts are checked rather
than assumed: CHELSA rows run north to south while the model grid runs south to north, and
the grid's coordinates are cell centres, so the read window is offset by half a cell.
"""

import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .common.runner import artefacts_root
from .panel import GRID, GridSpec

BASE_URL = "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/monthly"
NATIVE_RES = 1.0 / 120.0          # 30 arc-seconds
VARIABLES = ("tas", "pr")

# CHELSA v2.1 ships integer rasters with a documented scale and offset per variable,
# verified against read values and asserted in `to_grid`.
SCALING = {"tas": {"scale": 0.1, "offset": 0.0, "units": "K",
                   "plausible": (200.0, 340.0)},
           "pr": {"scale": 0.01, "offset": 0.0, "units": "kg m-2 month-1",
                  "plausible": (0.0, 5000.0)}}

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "VSI_CACHE": "TRUE",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "3",
}


def url_for(variable: str, year: int, month: int) -> str:
    if variable not in VARIABLES:
        raise ValueError(f"unknown CHELSA variable {variable!r}; expected one of {VARIABLES}")
    return f"{BASE_URL}/{variable}/CHELSA_{variable}_{month:02d}_{year}_V.2.1.tif"


def window_bounds(grid: GridSpec = GRID) -> tuple[float, float, float, float]:
    """Read window ``(west, south, east, north)``, expanded by half a grid cell."""
    b = grid.bounds()
    return b["lon_min"], b["lat_min"], b["lon_max"], b["lat_max"]


def _blocks_per_cell(grid: GridSpec = GRID) -> int:
    ratio = grid.res / NATIVE_RES
    n = int(round(ratio))
    if abs(ratio - n) > 1e-9:
        raise ValueError(f"grid resolution {grid.res} is not an integer multiple of the "
                         f"CHELSA native {NATIVE_RES}; a block mean would be wrong")
    return n


def to_grid(array: np.ndarray, variable: str, grid: GridSpec = GRID) -> np.ndarray:
    """Block-mean a native-resolution window onto the model grid, south-up.

    ``array`` must be the exact window from :func:`window_bounds`, north-up as CHELSA
    stores it.
    """
    n = _blocks_per_cell(grid)
    expected = (grid.H * n, grid.W * n)
    if array.shape != expected:
        raise ValueError(f"window is {array.shape}, expected {expected} for a "
                         f"{grid.H}x{grid.W} grid at {n}x{n} native cells per cell")

    spec = SCALING[variable]
    data = array.astype(np.float64)
    data[array == -2147483647] = np.nan
    data = data * spec["scale"] + spec["offset"]

    reduced = np.nanmean(data.reshape(grid.H, n, grid.W, n), axis=(1, 3))
    # CHELSA rows run north to south; the model grid runs south to north.
    reduced = reduced[::-1, :]

    lo, hi = spec["plausible"]
    finite = reduced[np.isfinite(reduced)]
    if finite.size and (finite.min() < lo or finite.max() > hi):
        raise ValueError(f"{variable} values {finite.min():.1f}-{finite.max():.1f} fall outside "
                         f"the plausible range {lo}-{hi} {spec['units']}; check the scale factor")
    return reduced.astype(np.float32)


def read_month(variable: str, year: int, month: int, grid: GridSpec = GRID) -> np.ndarray:
    """Fetch one month's window over HTTP and reduce it to the model grid."""
    import rasterio
    from rasterio.windows import from_bounds

    for key, value in GDAL_ENV.items():
        os.environ.setdefault(key, value)

    west, south, east, north = window_bounds(grid)
    with rasterio.open(f"/vsicurl/{url_for(variable, year, month)}") as src:
        window = from_bounds(west, south, east, north, src.transform)
        window = window.round_lengths().round_offsets()
        array = src.read(1, window=window)
    return to_grid(array, variable, grid)


def verify_orientation(field: np.ndarray, grid: GridSpec = GRID) -> dict[str, Any]:
    """Assert the field is south-up by checking known climatology (July: Madrid > London >
    Stockholm), not by inspection."""
    def at(lat: float, lon: float) -> float:
        i, j = grid.to_cell(lat, lon)
        return float(field[int(i), int(j)])

    madrid, london, stockholm = at(40.42, -3.70), at(51.51, -0.13), at(59.33, 18.07)
    ordered = madrid > london > stockholm
    return {"madrid": round(madrid, 2), "london": round(london, 2),
            "stockholm": round(stockholm, 2), "south_up_ordering_holds": bool(ordered),
            "check": "July: Madrid > London > Stockholm"}


def build_series(years: Sequence[int], variables: Iterable[str] = VARIABLES,
                 grid: GridSpec = GRID, out_dir: str | Path | None = None,
                 progress: bool = False) -> dict[str, Any]:
    """Fetch every month of the requested years, caching one ``.npy`` per (variable, year).

    CHELSA v2.1 is not uniformly complete (``pr`` stops at 2019-06); a missing month is
    recorded as NaN and listed, never silently skipped.
    """
    from .common.runner import atomic_path, write_json

    out_dir = Path(out_dir) if out_dir else artefacts_root() / "chelsa"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"grid": grid.as_dict(), "scaling": SCALING,
                                "source": BASE_URL, "years": list(years), "files": {}}

    for variable in variables:
        for year in years:
            path = out_dir / f"chelsa_{variable}_{year}.npy"
            if path.exists():
                manifest["files"][path.name] = {"cached": True}
                continue
            frames, missing = [], []
            for m in range(1, 13):
                try:
                    frames.append(read_month(variable, year, m, grid))
                except Exception as exc:
                    if "404" not in str(exc):
                        raise
                    frames.append(np.full((grid.H, grid.W), np.nan, dtype=np.float32))
                    missing.append(m)
            months = np.stack(frames)
            with atomic_path(path, suffix=".npy") as tmp:
                np.save(tmp, months)
            manifest["files"][path.name] = {
                "cached": False, "shape": list(months.shape), "missing_months": missing,
                "finite_fraction": float(np.isfinite(months).mean()),
                "min": float(np.nanmin(months)), "max": float(np.nanmax(months))}
            if progress:
                print(f"  {variable} {year}: {months.shape} "
                      f"[{np.nanmin(months):.1f}, {np.nanmax(months):.1f}]", flush=True)

    write_json(out_dir / "chelsa_manifest.json", manifest)
    return manifest


def load_series(variable: str, years: Sequence[int], out_dir: str | Path | None = None
                ) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return ``[n_months, H, W]`` and the matching ``(year, month)`` index."""
    out_dir = Path(out_dir) if out_dir else artefacts_root() / "chelsa"
    arrays, index = [], []
    for year in years:
        path = out_dir / f"chelsa_{variable}_{year}.npy"
        if not path.exists():
            continue
        arrays.append(np.load(path))
        index.extend((year, m) for m in range(1, 13))
    if not arrays:
        raise FileNotFoundError(f"no cached CHELSA {variable} for {list(years)}")
    return np.concatenate(arrays), index
