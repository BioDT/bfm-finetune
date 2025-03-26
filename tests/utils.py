import os
import platform

import pytest
import torch

from bfm_finetune.paths import STORAGE_DIR


def requires_gpu():
    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU is not available"
    )


def requires_snellius():
    return pytest.mark.skipif(
        not platform.node().endswith("snellius.surf.nl"),
        reason="You are not running on snellius",
    )


def requires_data():
    return pytest.mark.skipif(not STORAGE_DIR.exists(), reason="STORAGE_DIR not found")
