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

with TiffFile(tiff_path) as tif:
    print("pages", len(tif.pages))  # 1 page
    for page in tif.pages:
        print("shape", page.shape)  # (1432, 1563, 26)
        print("axes", page.axes)  # XYS
        series = tif.series[0]
        print("series shape", series.shape)  # (1432, 1563, 26)
        print("series axes", series.axes)  # XYS
        for tag in page.tags:
            tag_name, tag_value = tag.name, tag.value
            print(tag_name, tag_value)

# ImageWidth 1563
# ImageLength 1432
# 26 variables (GDAL_METADATA sample BioVarX)


# dem = xr.open_rasterio('https://download.osgeo.org/geotiff/samples/pci_eg/latlong.tif')
import rioxarray

## plotting
import xarray as xr

dem = xr.open_dataset(tiff_path)
dem = dem[0]
print(dem)
# <xarray.Dataset> Size: 233MB
# Dimensions:      (band: 26, x: 1563, y: 1432)
# Coordinates:
#   * band         (band) int64 208B 1 2 3 4 5 6 7 8 9 ... 19 20 21 22 23 24 25 26
#   * x            (x) float64 13kB 2.411e+06 2.414e+06 ... 7.094e+06 7.097e+06
#   * y            (y) float64 11kB 5.619e+06 5.616e+06 ... 1.329e+06 1.326e+06
#     spatial_ref  int64 8B ...
# Data variables:
#     band_data    (band, y, x) float32 233MB ...

import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

# img = rasterio.open(tiff_path)
# show(img)
with rasterio.open(tiff_path) as src:
    fig, ax = plt.subplots(figsize=(10, 10))
    show(src, ax=ax, title="Raster Image")
    # Save the figure to a file
    plt.savefig("raster_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
