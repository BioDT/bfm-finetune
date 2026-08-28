"""Dataset and DataLoader pairing BioCube inputs with panel targets.

One sample is one target year: the BioCube ``(t-1, t)`` month pair that closes before that
task's survey window, plus every observed panel row for that year. Targets stay in long
form; gridding first would reintroduce the empty cells the panel exists to exclude.
Collation keeps the model input un-batched — BFM consumes its own ``Batch`` namedtuple, so
this loader yields one year at a time.
"""

from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from . import batch as B
from .eval.nulls import observed


class EumonYearDataset(Dataset):
    """(BioCube input, observed panel rows) for each requested target year.

    Years whose BioCube input is unavailable are dropped at construction with a reason,
    so a missing month surfaces before any GPU time.
    """

    def __init__(self, panel: pd.DataFrame, years: Sequence[int], cfg: Any,
                 biocube_dir: str | Path = B.BIOCUBE_DIR, species: Iterable[str] | None = None,
                 lazy: bool = True):
        self.task = str(panel["task"].iloc[0])
        self.cfg = cfg
        self.lazy = lazy
        self.panel = observed(panel)
        if species is not None:
            self.panel = self.panel.loc[self.panel["species"].isin(list(species))]

        self.years: list[int] = []
        self.inputs: dict[int, dict[str, Any]] = {}
        self.skipped: list[dict[str, Any]] = []
        for year in sorted(set(int(y) for y in years)):
            if not (self.panel["year"] == year).any():
                self.skipped.append({"year": year, "reason": "no observed panel rows"})
                continue
            try:
                self.inputs[year] = B.forecast_input(self.task, year, biocube_dir)
            except (B.WindowUnavailable, ValueError) as exc:
                self.skipped.append({"year": year, "reason": str(exc)})
                continue
            self.years.append(year)

        self._dataset = None if lazy else B.make_dataset(cfg, biocube_dir)
        self._biocube_dir = biocube_dir
        self._cache: dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.years)

    def _backend(self):
        if self._dataset is None:
            self._dataset = B.make_dataset(self.cfg, self._biocube_dir)
        return self._dataset

    def targets_for(self, year: int) -> pd.DataFrame:
        cols = ["unit_id", "species", "year", "cell_i", "cell_j", "lat", "lon", "effort", "value"]
        return self.panel.loc[self.panel["year"] == year, cols].reset_index(drop=True)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        year = self.years[idx]
        if year not in self._cache:
            self._cache[year] = B.load_input(self.inputs[year]["path"], self._backend())
        return {"year": year, "task": self.task, "x": self._cache[year],
                "targets": self.targets_for(year), "input_meta": self.inputs[year]}

    def describe(self) -> dict[str, Any]:
        return {"task": self.task, "n_years": len(self.years), "years": self.years,
                "skipped": self.skipped,
                "n_units": int(self.panel["unit_id"].nunique()),
                "n_species": int(self.panel["species"].nunique()),
                "rows_per_year": {int(y): int((self.panel["year"] == y).sum())
                                  for y in self.years}}


def identity_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """One year per step; BFM's Batch cannot be stacked along a new leading axis."""
    if len(items) != 1:
        raise ValueError("EumonYearDataset must be loaded with batch_size=1")
    return items[0]


def make_loader(dataset: EumonYearDataset, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                      collate_fn=identity_collate)


def split_loaders(panel: pd.DataFrame, cfg: Any, *, train_years: Sequence[int],
                  val_years: Sequence[int] = (), test_years: Sequence[int] = (),
                  **kwargs: Any) -> dict[str, EumonYearDataset]:
    overlap = set(train_years) & set(test_years)
    if overlap:
        raise ValueError(f"train and test years overlap: {sorted(overlap)}")
    return {name: EumonYearDataset(panel, years, cfg, **kwargs)
            for name, years in (("train", train_years), ("val", val_years), ("test", test_years))
            if len(years)}
