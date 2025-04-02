from bfm_finetune.dataloaders.geolifeclef_species.dataloader import (
    GeoLifeCLEFSpeciesDataset,
)
from tests.utils import requires_data


@requires_data()
def test_find_files():
    dataset = GeoLifeCLEFSpeciesDataset()
    assert len(dataset) >= 4


@requires_data()
def test_load_one():
    dataset = GeoLifeCLEFSpeciesDataset()
    element = dataset[0]
    batch = element["batch"]
    target = element["target"]
    assert "species_distribution" in batch.surf_vars
    species_distribution = batch.surf_vars["species_distribution"]
    assert species_distribution.shape[0] == 2
    assert species_distribution.shape[1] == 500
    assert species_distribution.shape[2] == 152
    assert species_distribution.shape[3] == 320
    assert target.shape[0] == 500
    assert target.shape[1] == 152
    assert target.shape[2] == 320


if __name__ == "__main__":
    test_find_files()
    test_load_one()
