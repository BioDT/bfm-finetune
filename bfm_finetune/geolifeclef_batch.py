import importlib
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from bfm_finetune import paths, plots
from bfm_finetune.utils import (
    aggregate_into_latlon_grid,
    get_lat_lon_ranges,
    unroll_matrix_into_df,
)


def load_pa_csv() -> pd.DataFrame:
    finetune_location = paths.STORAGE_DIR / "finetune"
    geolifeclef_location = finetune_location / "geolifeclef24"

    # https://lab.plantnet.org/seafile/d/bdb829337aa44a9489f6/files/?p=%2FPresenceAbsenceSurveys%2FReadMe.txt
    # presence-absence: in europe
    pa_path = geolifeclef_location / "GLC24_PA_metadata_train.csv"
    df = pd.read_csv(pa_path)
    return df


def get_matrix_for_species(
    df: pd.DataFrame,
    species_ids: List[float],
    years: List[int],
    lat_range: np.ndarray,
    lon_range: np.ndarray,
    step: float,
) -> np.ndarray:
    """Returns a matrix with shape [species, years, latitudes, longitudes]"""
    result = np.zeros((len(species_ids), len(years), len(lat_range), len(lon_range)))
    for species_i, species_id in enumerate(tqdm(species_ids, desc="species")):
        df_species = df[df["speciesId"] == species_id]
        for year_i, year in enumerate(years):
            matrix_species_year = aggregate_into_latlon_grid(
                df_species[df_species["year"] == year],
                lat_range=lat_range,
                lon_range=lon_range,
                step=step,
            )
            result[species_i, year_i, :, :] = matrix_species_year
    return result


if __name__ == "__main__":
    df = load_pa_csv()
    # select the most frequent species
    occurrences = (
        df.groupby(["speciesId"])["speciesId"].count().sort_values(ascending=False)
    )
    how_many_species = 500
    species_ids = occurrences.index[:how_many_species].tolist()

    # or if we want all species
    # species_ids = df["speciesId"].unique().tolist()

    years = sorted(int(el) for el in df["year"].unique().tolist())

    step = 0.25
    lat_range, lon_range = get_lat_lon_ranges(lat_step=step, lon_step=step)

    species_matrix = get_matrix_for_species(
        df=df,
        species_ids=species_ids,
        years=years,
        lat_range=lat_range,
        lon_range=lon_range,
        step=step,
    )
    print("shape", species_matrix.shape)
