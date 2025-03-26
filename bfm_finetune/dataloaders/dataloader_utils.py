import torch
from aurora.batch import Batch, Metadata
from torch.utils.data import DataLoader, Dataset, default_collate


# Custom collate function to merge a list of Batch objects.
def collate_batches(batch_list):
    # Merge surf_vars, static_vars, and atmos_vars by stacking their tensor values.
    surf_vars = {
        k: torch.stack([b.surf_vars[k] for b in batch_list], dim=0)
        for k in batch_list[0].surf_vars.keys()
    }
    # static_vars = {
    #     k: torch.stack([b.static_vars[k] for b in batch_list], dim=0)
    #     for k in batch_list[0].static_vars.keys()
    # }
    atmos_vars = {
        k: torch.stack([b.atmos_vars[k] for b in batch_list], dim=0)
        for k in batch_list[0].atmos_vars.keys()
    }
    # For static_vars, we simply take the one from the first sample.
    static_vars = batch_list[0].static_vars
    # For metadata, we assume they are the same across samples.
    metadata = batch_list[0].metadata
    return Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=metadata,
    )


def custom_collate_fn(samples):
    # Each sample is a dict with keys "batch" and "target".
    batch_list = [s["batch"] for s in samples]
    collated_batch = collate_batches(batch_list)
    targets = default_collate([s["target"] for s in samples])
    return {"batch": collated_batch, "target": targets}
