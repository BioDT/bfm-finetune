from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rioxarray
import xarray as xr
from rasterio.plot import show
from rasterio.warp import transform
from tifffile import TiffFile

from bfm_finetune.dataloaders.biovars import utils

# with 01 as realisation number, 1971–2000 as period and 5 as temporal percentile within the period.
tiff_path = utils.historical_biovars_location / "biovars_met01_1971-2000_5.tif"

# 71 realizations, each one with 3 temporal percentiles --> 213 files
# TODO: average across realisations for ground truth?

# Time dimension
# 1971-2000 should be 30 years, where is 30 (time) in the dimensions?
# yearly / monthly / daily: where is it?

# 1 file = 65MB (x 213 files = 13.8GB)
# (1432, 1563, 26) -> 58M. Every value takes 1 byte + some metadata = 65MB

# with TiffFile(tiff_path) as tif:
#     print("pages", len(tif.pages))  # 1 page
#     for page in tif.pages:
#         print("shape", page.shape)  # (1432, 1563, 26)
#         print("axes", page.axes)  # XYS
#         series = tif.series[0]
#         print("series shape", series.shape)  # (1432, 1563, 26)
#         print("series axes", series.axes)  # XYS
#         for tag in page.tags:
#             tag_name, tag_value = tag.name, tag.value
#             print(tag_name, tag_value)

# ImageWidth 1563
# ImageLength 1432
# 26 variables (GDAL_METADATA sample BioVarX)


# dem = xr.open_rasterio('https://download.osgeo.org/geotiff/samples/pci_eg/latlong.tif')


## plotting


# dem = xr.open_dataset(tiff_path)
# dem = dem[0]
# print(dem)
# <xarray.Dataset> Size: 233MB
# Dimensions:      (band: 26, x: 1563, y: 1432)
# Coordinates:
#   * band         (band) int64 208B 1 2 3 4 5 6 7 8 9 ... 19 20 21 22 23 24 25 26
#   * x            (x) float64 13kB 2.411e+06 2.414e+06 ... 7.094e+06 7.097e+06
#   * y            (y) float64 11kB 5.619e+06 5.616e+06 ... 1.329e+06 1.326e+06
#     spatial_ref  int64 8B ...
# Data variables:
#     band_data    (band, y, x) float32 233MB ...


# img = rasterio.open(tiff_path)
# show(img)
# with rasterio.open(tiff_path) as src:
#     fig, ax = plt.subplots(figsize=(10, 10))
#     show(src, ax=ax, title="Raster Image")
#     # Save the figure to a file
#     plt.savefig("raster_plot.png", dpi=300, bbox_inches="tight")
#     plt.close()

N_VARIABLES = 26
N_X = 1432
N_Y = 1563
N_X_EPSG = 1073
N_Y_EPSG = 2704
# TODO: understand resampling between source (ETRS89-extended) and dest (EPSG:4326)


def get_lat_lon_from_tiff(tiff_file_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    # with rasterio.open(tiff_file_path) as src:
    #     band1 = src.read(1)
    #     height = band1.shape[0]
    #     width = band1.shape[1]
    #     cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    #     xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    #     lon, lat = transform(da.crs, {'init': 'EPSG:4326'},
    #                  x.flatten(), y.flatten())
    #     # EPSG:102002
    #     lons= np.array(xs)
    #     lats = np.array(ys)
    #     return lats, lons
    # data = xr.open_dataset(tiff_file_path, decode_coords="all")
    ds = rioxarray.open_rasterio(tiff_file_path, masked=True)
    ds_proj = ds.rio.reproject("EPSG:4326")
    print(ds.spatial_ref)  # ETRS89-extended
    print(ds_proj.spatial_ref)  # EPSG:4326
    lon = ds_proj.x.to_numpy()
    lat = ds_proj.y.to_numpy()
    return lat, lon


def tiff_to_matrix(tiff_file_path: str | Path) -> np.ndarray:
    data = xr.open_dataset(tiff_file_path)
    res = data.to_array().to_numpy()
    data.close()
    assert len(res.shape) == 4
    assert res.shape[0] == 1
    assert res.shape[1] == N_VARIABLES
    assert res.shape[2] == N_X
    assert res.shape[3] == N_Y
    return res.squeeze(axis=0)  # (26, 1432, 1563)
