#!/bin/bash
set -e


# poetry to venv
VENV_PATH=venv
if test -d $VENV_PATH; then
    echo "venv: $VENV_PATH already exists, using it"
else
    # create venv
    echo "creating venv at: $VENV_PATH"
    python3 -m venv $VENV_PATH
fi

source $VENV_PATH/bin/activate
pip install poetry


# install python dependencies
poetry install

# install pre-commit git hooks
pre-commit install


######################################################################################
# prithvi gravity wave finetuning
######################################################################################
git submodule update --init --recursive
PRITHVI_CHECKPOINT_URL=https://huggingface.co/ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M-rollout/resolve/main/prithvi.wxc.rollout.2300m.v1.pt?download=true
PRITHVI_CHECKPOINT_PATH=checkpoints/prithvi.wxc.rollout.2300m.v1.pt
if test -f $PRITHVI_CHECKPOINT_PATH; then
    echo "PRITHVI_CHECKPOINT_PATH: $PRITHVI_CHECKPOINT_PATH already exists, using it"
else
    echo "PRITHVI_CHECKPOINT_PATH: $PRITHVI_CHECKPOINT_PATH downloading..."
    mkdir -p checkpoints
    wget $PRITHVI_CHECKPOINT_URL -O $PRITHVI_CHECKPOINT_PATH
    echo "PRITHVI_CHECKPOINT_PATH downloaded to $PRITHVI_CHECKPOINT_PATH"
fi


######################################################################################
# geolifeclef24
######################################################################################

# download source csv
echo "downloading source csv.."
PA_CSV_URL=https://lab.plantnet.org/seafile/seafhttp/files/710990c5-411a-4512-b846-676c94f94034/GLC24-PA-metadata-train.csv
GEOLIFECLEF_PATH=data/finetune/geolifeclef24
mkdir -p $GEOLIFECLEF_PATH
wget $PA_CSV_URL -O $GEOLIFECLEF_PATH/GLC24_PA_metadata_train.csv

echo "creating batches for geolifeclef..."
python bfm_finetune/dataloaders/geolifeclef_species/batch.py
echo "creating batches for geolifeclef+prithvi..."
python bfm_finetune/prithvi/create_patches.py

echo "DONE!"
