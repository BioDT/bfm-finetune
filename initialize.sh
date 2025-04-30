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

echo "DONE!"
