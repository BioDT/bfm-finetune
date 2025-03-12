from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bfm_finetune.utils import get_lat_lon_ranges

# finetune_location = Path("/projects/prjs1134/data/projects/biodt/storage/finetune") # snellius
finetune_location = Path("data/finetune")  # local

geolifeclef_location = finetune_location / "geolifeclef24"

# monthly_train_bioclimatic_path = (
#     geolifeclef_location
#     / "TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-train-bioclimatic_monthly/"
# )

# all_pt_files = glob(str(monthly_train_bioclimatic_path / "*.pt"))
# all_pt_files = sorted(all_pt_files)
# first_pt_file_path = all_pt_files[0]

# tensor = torch.load(first_pt_file_path)
# print(tensor.shape)
# # torch.Size([4, 19, 12])


# https://lab.plantnet.org/seafile/d/bdb829337aa44a9489f6/files/?p=%2FPresenceAbsenceSurveys%2FReadMe.txt
# presence-absence: in europe
pa_path = geolifeclef_location / "GLC24_PA_metadata_train.csv"

df = pd.read_csv(pa_path)
# 1483637
print(len(df))
# ['lon', 'lat', 'year', 'geoUncertaintyInM', 'areaInM2', 'region', 'country', 'speciesId', 'surveyId']
print(df.columns)

df["speciesId"].unique()  # 5016 species
df["surveyId"].unique()  # 88987 different surveys

# one survey:
df[df["surveyId"] == 212]  # 15 rows (15 different species?)

# one species?
df[df["speciesId"] == 6874.0]  # 924 rows


# https://lab.plantnet.org/seafile/d/bdb829337aa44a9489f6/files/?p=%2FPresenceOnlyOccurrences%2FReadMe.txt
po_path = geolifeclef_location / "GLC24_PA_metadata_train.csv"

po_df = pd.read_csv(po_path)

# https://github.com/plantnet/malpolon/blob/80e1084a8d575d9c3e4fca1bd409125dfe9863fa/malpolon/data/datasets/geolifeclef2024_pre_extracted.py#L291C35-L291C44
# the label is the speciesId from metadata


# TODO: how to get a distribution matrix (lat-lon) for a specific species in a specific point in time?
# - let's try with year 2021
# - speciesId 6874
# - what does the value represent? animal is there or not / how many animals of this species?

step = 0.25
lat_range, lon_range = get_lat_lon_ranges(lat_step=step, lon_step=step)

df_selected = df[df["speciesId"] == 6874.0]
df_selected = df_selected[df_selected["year"] == 2021]
# 183 rows
matrix = np.zeros()
for lat_i, lat_val in enumerate(lat_range):
    for lon_i, lon_val in enumerate(lon_range):
        df_here = df_selected[
            df_selected["lat"] >= lat_val - step / 2
            and df_selected["lat"] < lat_val + step / 2
        ]  # TODO filter pandas multiple filters
