import logging
import os
import platform
from pathlib import Path

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
platform_node = platform.node()
if platform_node.endswith("snellius.surf.nl"):
    STORAGE_DIR = Path("/projects/prjs1134/data/projects/biodt/storage")
else:
    logger.warning(
        f"Unknown platform {platform_node}. Assuming data is in the current directory"
    )
    REPO_FOLDER = Path(os.path.dirname(os.path.abspath(__file__))) / ".."
    STORAGE_DIR = (REPO_FOLDER / "data").resolve()

logger.warning(f"STORAGE_DIR: {STORAGE_DIR}")
if not os.path.exists(STORAGE_DIR):
    logger.warning(
        f"STORAGE_DIR={STORAGE_DIR} does not exist!!!! Have you run ./initialize.sh?"
    )
