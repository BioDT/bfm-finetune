from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoLifeCLEFSpeciesDataset,
)
from tests.utils import requires_data


@requires_data()
def test_find_files():
    dataset = GeoLifeCLEFSpeciesDataset()
    assert len(dataset) >= 3


@requires_data()
def test_load_one():
    dataset = GeoLifeCLEFSpeciesDataset()
    element = dataset[0]
    batch = element["batch"]
    target = element["target"]
    assert "species_distribution" in batch, f"keys available in surf_vars: {batch.keys()}"
    species_distribution = batch["species_distribution"]
    assert species_distribution.shape[0] == 2, species_distribution.shape
    assert species_distribution.shape[1] == 500, species_distribution.shape
    assert species_distribution.shape[2] == 152, species_distribution.shape
    assert species_distribution.shape[3] == 320, species_distribution.shape
    assert target.shape[0] == 1, target.shape
    assert target.shape[1] == 500, target.shape
    assert target.shape[2] == 152, target.shape
    assert target.shape[3] == 320, target.shape


if __name__ == "__main__":
    test_find_files()
    test_load_one()
