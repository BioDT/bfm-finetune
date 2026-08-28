#!/usr/bin/env bash
# Idempotent environment setup: submodules, in-project venv, dependencies, git hooks.
#
#   ./initialize.sh                 # core install (eumon benchmark and BioAnalyst tasks)
#   ./initialize.sh --with-prithvi  # add the Prithvi-WxC dependency group
#   ./initialize.sh --with-assets   # additionally fetch the legacy task assets (large)
set -euo pipefail
cd "$(dirname "$0")"

WITH_PRITHVI=0
WITH_ASSETS=0
for arg in "$@"; do
    case $arg in
        --with-prithvi) WITH_PRITHVI=1 ;;
        --with-assets)  WITH_PRITHVI=1; WITH_ASSETS=1 ;;
        *) echo "unknown option: $arg (use --with-prithvi / --with-assets)"; exit 2 ;;
    esac
done

if [[ ${HOSTNAME:-} =~ snellius ]]; then
    module purge
    module load 2024 Python/3.12.3-GCCcore-13.3.0
fi

PY=${PYTHON:-python3.12}
command -v "$PY" >/dev/null 2>&1 || PY=python3

git submodule update --init --recursive

if [ ! -d .venv ]; then
    echo "creating .venv with $("$PY" --version)"
    "$PY" -m venv .venv
fi
.venv/bin/pip install --quiet --upgrade pip poetry

if [ $WITH_PRITHVI -eq 1 ]; then
    .venv/bin/poetry install --with prithvi
else
    .venv/bin/poetry install
fi

.venv/bin/pre-commit install
echo "environment ready: $(.venv/bin/python --version) at .venv/bin/python"
echo
echo "benchmark inputs and weights (pinned SHA-256s):"
echo "  .venv/bin/python -m bfm_finetune.eumon.download fetch-all"

[ $WITH_ASSETS -eq 1 ] || exit 0

##############################################################################
# Legacy task assets: Prithvi-WxC checkpoint, GeoLifeCLEF-24, bioVars
##############################################################################
STORAGE_DIR=${STORAGE_DIR:-data}
[[ ${HOSTNAME:-} =~ snellius ]] && STORAGE_DIR=/projects/prjs1134/data/projects/biodt/storage
echo "STORAGE_DIR=$STORAGE_DIR"

PRITHVI_DIR=$STORAGE_DIR/checkpoints_prithvi
PRITHVI_CKPT=$PRITHVI_DIR/prithvi.wxc.rollout.2300m.v1.pt
if [ ! -f "$PRITHVI_CKPT" ]; then
    mkdir -p "$PRITHVI_DIR"
    wget "https://huggingface.co/ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M-rollout/resolve/main/prithvi.wxc.rollout.2300m.v1.pt?download=true" \
        -O "$PRITHVI_CKPT"
fi

GLC_DIR=$STORAGE_DIR/finetune/geolifeclef24
GLC_CSV=$GLC_DIR/GLC24_PA_metadata_train.csv
if [ ! -f "$GLC_CSV" ]; then
    mkdir -p "$GLC_DIR"
    .venv/bin/python bfm_finetune/plantnet_downloader.py \
        "https://lab.plantnet.org/seafile/d/bdb829337aa44a9489f6/files/?p=%2FPresenceAbsenceSurveys%2FGLC24-PA-metadata-train.csv" \
        "$GLC_CSV"
fi
[ -d "$GLC_DIR/aurorashape_species/train" ] || .venv/bin/python bfm_finetune/dataloaders/geolifeclef_species/batch.py
[ -d "$GLC_DIR/prithvi_species_patches/train" ] || .venv/bin/python bfm_finetune/prithvi/create_patches.py

BIOVARS_DIR=$STORAGE_DIR/finetune/biovars
BIOVARS_TAR=$BIOVARS_DIR/bioVars_1971-2000_met.tar.gz
BIOVARS_OUT=$BIOVARS_DIR/bioVars_1971-2000_met
mkdir -p "$BIOVARS_OUT"
[ -f "$BIOVARS_TAR" ] || wget "https://zenodo.org/records/14624171/files/bioVars_1971-2000_met.tar.gz?download=1" -O "$BIOVARS_TAR"
[ -n "$(ls -A "$BIOVARS_OUT" 2>/dev/null)" ] || tar -xzf "$BIOVARS_TAR" -C "$BIOVARS_OUT"

echo "DONE"
