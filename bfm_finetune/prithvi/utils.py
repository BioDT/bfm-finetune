from bfm_finetune.dataloaders.geolifeclef_species.utils import geolifeclef_location
from bfm_finetune.paths import REPO_FOLDER, STORAGE_DIR

checkpoints_path = STORAGE_DIR / "checkpoints_prithvi"
prithvi_species_patches_location = geolifeclef_location / "prithvi_species_patches"
prithvi_output_checkpoint_path = (
    checkpoints_path / "prithvi_species_geolifeclef" / "species_distributions_best.pt"
)

prithvi_inference_results_path = (
    REPO_FOLDER / "results" / "prithvi_inference_species_distributions"
)
