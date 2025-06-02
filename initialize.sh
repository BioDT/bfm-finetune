#!/bin/bash
set -e

# STORAGE_DIR is the root of all the data
if [[ $HOSTNAME =~ "snellius" ]]; then
    export STORAGE_DIR=/projects/prjs1134/data/projects/biodt/storage # snellius
else
    export STORAGE_DIR=data # local folder
fi
echo "STORAGE_DIR=$STORAGE_DIR"

# load proper modules to be able to install
if [[ $HOSTNAME =~ "snellius" ]]; then
    module purge
    module load 2024 Python/3.12.3-GCCcore-13.3.0
fi

# poetry to venv
VENV_PATH=.venv
if test -d $VENV_PATH; then
    echo "venv: $VENV_PATH already exists, using it"
else
    # create venv
    echo "creating venv at: $VENV_PATH"
    python3 -m venv $VENV_PATH
fi

# init all submodules
git submodule update --init --recursive

# install poetry
source $VENV_PATH/bin/activate
if ! [ -x "$(command -v poetry)" ]; then
    echo 'INFO: poetry is not installed. Installing'
    pip install poetry
fi
# pip install poetry


# install python dependencies
poetry install

# install pre-commit git hooks (formats code)
pre-commit install

echo Python path: $(which python)

######################################################################################
# prithvi gravity wave finetuning
######################################################################################
PRITHVI_CHECKPOINT_URL=https://huggingface.co/ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M-rollout/resolve/main/prithvi.wxc.rollout.2300m.v1.pt?download=true
PRITHVI_CHECKPOINT_DIR=$STORAGE_DIR/checkpoints_prithvi
PRITHVI_CHECKPOINT_PATH=$PRITHVI_CHECKPOINT_DIR/prithvi.wxc.rollout.2300m.v1.pt
if test -f $PRITHVI_CHECKPOINT_PATH; then
    echo "PRITHVI_CHECKPOINT_PATH: $PRITHVI_CHECKPOINT_PATH already exists, using it"
else
    echo "PRITHVI_CHECKPOINT_PATH: $PRITHVI_CHECKPOINT_PATH downloading..."
    mkdir -p $PRITHVI_CHECKPOINT_DIR
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
GEOLIFECLEF_PATH=$STORAGE_DIR/finetune/geolifeclef24
GEOLIFECLEF_FILE=$GEOLIFECLEF_PATH/GLC24_PA_metadata_train.csv
if test -f $GEOLIFECLEF_FILE; then
    echo "GEOLIFECLEF_FILE: $GEOLIFECLEF_FILE already exists, using it"
else
    mkdir -p $GEOLIFECLEF_PATH
    python bfm_finetune/plantnet_downloader.py $PA_CSV_URL $GEOLIFECLEF_FILE
fi

echo "creating batches for geolifeclef..."
GEOLIFE_AURORASHAPE_PATH=$GEOLIFECLEF_PATH/aurorashape_species/train
files=$(shopt -s nullglob dotglob; echo $GEOLIFE_AURORASHAPE_PATH)
if (( ${#files} )) ; then
    echo "$GEOLIFE_AURORASHAPE_PATH contains files"
else
    echo "$GEOLIFE_AURORASHAPE_PATH is empty (or does not exist or is a file)"
    python bfm_finetune/dataloaders/geolifeclef_species/batch.py
fi

echo "creating batches for geolifeclef+prithvi..."
GEOLIFE_PRITHVI_PATH=$GEOLIFECLEF_PATH/prithvi_species_patches/train
files=$(shopt -s nullglob dotglob; echo $GEOLIFE_PRITHVI_PATH)
if (( ${#files} )) ; then
    echo "$GEOLIFE_PRITHVI_PATH contains files"
else
    echo "$GEOLIFE_PRITHVI_PATH is empty (or does not exist or is a file)"
    python bfm_finetune/prithvi/create_patches.py
fi



######################################################################################
# biovars batches
######################################################################################

echo "preparing biovars files.."
BIOVARS_FILE_NAME_WITHOUT_EXTENSION=bioVars_1971-2000_met
BIOVARS_FILE_NAME_WITH_EXTENSION=$BIOVARS_FILE_NAME_WITHOUT_EXTENSION.tar.gz
BIOVARS_URL=https://zenodo.org/records/14624171/files/$BIOVARS_FILE_NAME_WITH_EXTENSION?download=1
BIOVARS_PATH=$STORAGE_DIR/finetune/biovars
BIOVARS_EXTRACTED_PATH=$BIOVARS_PATH/$BIOVARS_FILE_NAME_WITHOUT_EXTENSION
BIOVARS_FILE_PATH=$BIOVARS_PATH/$BIOVARS_FILE_NAME_WITH_EXTENSION
mkdir -p $BIOVARS_PATH
if test -f $BIOVARS_FILE_PATH; then
    echo "BIOVARS_FILE_PATH: $BIOVARS_FILE_PATH already exists, using it"
else
    wget $BIOVARS_URL -O $BIOVARS_FILE_PATH
fi
mkdir -p $BIOVARS_EXTRACTED_PATH
files=$(shopt -s nullglob dotglob; echo $BIOVARS_EXTRACTED_PATH)
if (( ${#files} )) ; then
    echo "$BIOVARS_EXTRACTED_PATH contains files"
else
    echo "$BIOVARS_EXTRACTED_PATH is empty (or does not exist or is a file)"
    tar -xvzf $BIOVARS_FILE_PATH -C $BIOVARS_EXTRACTED_PATH
fi

echo "DONE!"
