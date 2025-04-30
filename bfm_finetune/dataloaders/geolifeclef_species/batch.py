import importlib
import json
import os
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import typer
from tqdm import tqdm

from bfm_finetune.dataloaders.geolifeclef_species import utils
from bfm_finetune.utils import (
    aggregate_into_latlon_grid,
    get_lat_lon_ranges,
)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def load_pa_csv() -> pd.DataFrame:
    # https://lab.plantnet.org/seafile/d/bdb829337aa44a9489f6/files/?p=%2FPresenceAbsenceSurveys%2FReadMe.txt
    # presence-absence: in europe
    pa_path = utils.geolifeclef_location / "GLC24_PA_metadata_train.csv"
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
    """Returns a matrix with shape [years, species, latitudes, longitudes]"""
    result = np.zeros((len(years), len(species_ids), len(lat_range), len(lon_range)))
    for species_i, species_id in enumerate(tqdm(species_ids, desc="species")):
        df_species = df[df["speciesId"] == species_id]
        for year_i, year in enumerate(years):
            matrix_species_year = aggregate_into_latlon_grid(
                df_species[df_species["year"] == year],
                lat_range=lat_range,
                lon_range=lon_range,
                step=step,
            )
            result[year_i, species_i, :, :] = matrix_species_year
    return result


def compute_and_write_stats(species_matrix: np.ndarray, output_path: str | Path):
    # shape: [T, S, H, W]
    stats = []
    for species_i in range(species_matrix.shape[1]):
        single_species_matrix = species_matrix[:, species_i, :, :]
        min = single_species_matrix.min()
        max = single_species_matrix.max()
        mean = single_species_matrix.mean()
        std = single_species_matrix.std()
        count = np.count_nonzero(single_species_matrix)
        stats.append(
            {
                "species_i": species_i,
                "min": min,
                "max": max,
                "mean": mean,
                "std": std,
                "count": count,
            }
        )
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
        f.write("\n") # newline at the end


@app.command()
def main(only_positive_lon: bool = False, roll_negative_lon: bool = False):
    if only_positive_lon or roll_negative_lon:
        print(
            "Better to use only_positive_lon or roll_negative_lon only in the DataLoader"
        )
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
    if only_positive_lon:
        lon_range = lon_range[lon_range >= 0.0]
        print("only_positive_lon", len(lon_range))

    species_matrix = get_matrix_for_species(
        df=df,
        species_ids=species_ids,
        years=years,
        lat_range=lat_range,
        lon_range=lon_range,
        step=step,
    )
    # now put lon+360
    lon_neg_loc = np.where(lon_range < 0)[0]
    if only_positive_lon:
        assert len(lon_neg_loc) == 0
        print("not doing roll with only_positive_lon")
    else:
        if roll_negative_lon:
            max_neg = lon_neg_loc.max()
            print("max_neg", max_neg)
            lon_range[: max_neg + 1] += 360
            lon_range = np.roll(lon_range, shift=-(max_neg + 1))
            species_matrix = np.roll(species_matrix, shift=-(max_neg + 1), axis=3)
    # stats for each species
    os.makedirs(utils.aurorashape_species_location, exist_ok=True)
    train_folder = utils.aurorashape_species_location / "train"
    val_folder = utils.aurorashape_species_location / "val"
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    compute_and_write_stats(
        species_matrix, utils.aurorashape_species_location / "stats.json"
    )
    # finished
    print("shape", species_matrix.shape)  # [T, S, H, W]
    paired_years_indices = [
        (i, i + 1)
        # we want all the transitions: [0,1], [1,2], [2,3] ...
        for i in range(len(years) - 1)
    ]
    print("year pairs", paired_years_indices)
    count = 0
    for year1_index, year2_index in tqdm(paired_years_indices, desc="Saving batches"):
        year_1 = years[year1_index]
        year_2 = years[year2_index]
        # if no split
        folder = utils.aurorashape_species_location
        if year_2 == years[-1]:
            # the last year goes to val
            folder = val_folder
        else:
            # all the other go to train
            folder = train_folder
        file_path = folder / f"yearly_species_{year_1}-{year_2}.pt"
        filtered_matrix = species_matrix[[year1_index, year2_index], :, :, :]
        batch_structure = {
            "species_distribution": torch.Tensor(filtered_matrix),
            "metadata": {
                "lat": lat_range.tolist(),
                "lon": lon_range.tolist(),
                "time": [year_1, year_2],  # TODO: this is only year now
                "species_ids": species_ids,
            },
        }
        torch.save(batch_structure, file_path)


if __name__ == "__main__":
    app()
