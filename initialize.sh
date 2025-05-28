#!/bin/bash
set -e

# poetry to venv
VENV_PATH=.venv
if test -d $VENV_PATH; then
    echo "venv: $VENV_PATH already exists, using it"
else
    # create venv
    echo "creating venv at: $VENV_PATH"
    python3 -m venv $VENV_PATH
fi

# install poetry
source $VENV_PATH/bin/activate
pip install -U pip setuptools wheel
pip install poetry

git submodule update --init --recursive

# install python dependencies
poetry install

# install pre-commit git hooks (formats code)
pre-commit install

######################################################################################
# prithvi gravity wave finetuning
######################################################################################
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
# BFM finetuning
######################################################################################
# checkpoint (snellius): /projects/prjs1134/data/projects/biodt/storage/weights/epoch=268-val_loss=0.00493.ckpt
# TODO: update script to download from huggingface when will be published there
# checkpoint (non-snellius): data/weights/epoch=268-val_loss=0.00493.ckpt


######################################################################################
# geolifeclef24 batches
######################################################################################

# download source csv
echo "downloading geolifeclef source csv.."
PA_CSV_URL=https://lab.plantnet.org/seafile/d/bdb829337aa44a9489f6/files/?p=%2FPresenceAbsenceSurveys%2FGLC24-PA-metadata-train.csv
GEOLIFECLEF_PATH=data/finetune/geolifeclef24
GEOLIFECLEF_FILE=$GEOLIFECLEF_PATH/GLC24_PA_metadata_train.csv
if test -f $GEOLIFECLEF_FILE; then
    echo "GEOLIFECLEF_FILE: $GEOLIFECLEF_FILE already exists, using it"
else
    mkdir -p $GEOLIFECLEF_PATH
    python bfm_finetune/plantnet_downloader.py $PA_CSV_URL $GEOLIFECLEF_FILE
fi

echo "creating batches for geolifeclef..."
python bfm_finetune/dataloaders/geolifeclef_species/batch.py
echo "creating batches for geolifeclef+prithvi..."
python bfm_finetune/prithvi/create_patches.py



######################################################################################
# biovars batches
######################################################################################

echo "preparing biovars files.."
BIOVARS_FILE_NAME_WITHOUT_EXTENSION=bioVars_1971-2000_met
BIOVARS_FILE_NAME_WITH_EXTENSION=$BIOVARS_FILE_NAME_WITHOUT_EXTENSION.tar.gz
BIOVARS_URL=https://zenodo.org/records/14624171/files/$BIOVARS_FILE_NAME_WITH_EXTENSION?download=1
BIOVARS_PATH=data/finetune/biovars
BIOVARS_EXTRACTED_PATH=$BIOVARS_PATH/$BIOVARS_FILE_NAME_WITHOUT_EXTENSION
BIOVARS_FILE_PATH=$BIOVARS_PATH/$BIOVARS_FILE_NAME_WITH_EXTENSION
mkdir -p $BIOVARS_PATH
if test -f $BIOVARS_FILE_PATH; then
    echo "BIOVARS_FILE_PATH: $BIOVARS_FILE_PATH already exists, using it"
else
    wget $BIOVARS_URL -O $BIOVARS_FILE_PATH
fi
mkdir -p $BIOVARS_EXTRACTED_PATH
tar -xvzf $BIOVARS_FILE_PATH -C $BIOVARS_EXTRACTED_PATH

echo "creating batches for geolifeclef..."

echo "DONE!"
