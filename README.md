# bfm-finetune

## Getting started

### 1. If you're on Snellius

On Snellius you need to execute the following first in order to use `python3.11` / `python3.12`:

```bash
module purge
# 3.11
module load 2023 Python/3.11.3-GCCcore-12.3.0
# or 3.12
module load 2024 Python/3.12.3-GCCcore-13.3.0
```

### 2. Initialize everything

Run the script `./initialize.sh`.

### 5. Finetuning tasks 


#### Biotic Task - Species Distribution Model
For [geolifeclef24](https://www.kaggle.com/competitions/geolifeclef-2024) dataset, you can recreate the batches with:

```bash
python bfm_finetune/dataloaders/geolifeclef_species/batch.py
```

This will create yearly batches for all the 5016 distinct species. You can then configure in `bfm_finetune/finetune_config.yaml` how many you want to use (e.g. setting to 500, will take the 500 most frequent species).

Keep in mind that the less frequent ones appear only in a few cells of the grid (after 1k). Also the closer you go to 5016, the highest CUDA memory you will need.

- Inside the batches saved to disk, `species_distribution` has shape `[T=2, Species, H, W]`.

- To train the model, you can use the script `bfm_finetune/finetune_bfm_sdm.py`.

```bash
python bfm_finetune/finetune_bfm_sdm.py
```

- To visualise the predictions of the finetuned model, you can use the notebook `notebooks/geolifeclef_species.ipynb`.

#### Abiotic Task - Climate linear probing

For the [CHELSA](https://chelsa-climate.org/) and BFMLatents datasets, you can recreate the batches with:

```bash
python bfm_finetune/dataloaders/chelsa/batch.py
```
This will create yearly batches for all the 19 CHELSA variables. You can then configure in `bfm_finetune/finetune_config.yaml` for the time period you want to use (e.g. setting to 2010-2020, will take the 11 years of data), along with other parameters for the backbone and the decoder outputs.

The latent variables and decoder oututs will be save as a netcdf file in the path defined in `bfm_finetune/dataloaders/chelsa/batch_config.yaml`.

To train the model, you can use the script `bfm_finetune/finetune_chelsa.py`.

```bash
python bfm_finetune/finetune_chelsa.py
```

To visualise the predictions of the finetuned model, you can use the notebook `notebooks/chelsa_2010_tas_pr.ipynb`.


### 6. Manually run code formatting / pre-commit

You can manually run the command on all the files (even if not modified) with:

```bash
pre-commit run --all-files
```

## Run some finetune workflows

First get some resources if you are in the cluster.
*Note: set gpus-per-node=2 or more if you planning to be faster!*
```
salloc -p gpu_h100 --gpus-per-node=1 -t 01:00:00
```

### Aurora Fine-Tune

1) In an activated environment, run `python bfm_finetune/finetune_new_variables.py`.

2) You can select to debug your finetune models using the toy dataset by changing the flag `finetune_new_variables(use_toy=True)`

3) Uncomment either one of the 3 Versions of the models to experiment with

4) You can do parallel training if your hardware supports it, by running the command `finetune_new_variables_multi_gpu.py`. You can edit the `finetune_config.yaml` to support your settings, e.g. fsdp vs ddp or the gpus ids [0,1].

#### Visualise predictions

You can visualise the predictions of the finetuned model by using the notebook `visualise_eval.ipynb`. Just change the **PATH** variable to map the location of your checkpoint.

#### Experimentation - Work in progress

An intro script with a toy example, using the small Aurora model and finetuning with the below logic is `finetune_new_variables.py`.

Concept:
- Spatiotemporal Encoder:
The new input (with, for example, 500 channels/species) is passed through a series of convolutional layers to match the backbone's input shape.

- Frozen Backbone & LoRA Finetune:
The backbone is frozen and LoRA adapters are added to the attention heads.

- Spatiotemporal Decoder:
The backbone's output is reconstructed after a series of convolutional layer back to the coordinate grid.

*NOTE: We are currently using the Aurora small for integration experiments. In the future we will adapt the codebase for using the BFM.*

### Prithvi-WxC Fine-Tune
*Experimental*

In this setting, we finetune keeping frozen the Prithvi-WxC backbone and using the same U-net style encoder-decoder architecture that was used during the gravite-wave finetuning routine.

Start the fine-tune training: `bfm_finetune/prithvi/train.sh`

Inference: `bfm_finetune/prithvi/inference.sh`

## TODOs

* [x] Monitoring & Logging
* [x] Checkpointing & Loading
* [x] Result visualisation
* [ ] Validate new visualisations & metrics
* [ ] Compare with baselines (50%)
* [x] Upsample to (721, 1440) earth grid in the encoder and downsample to (152, 320) in decoder. Edit the coordinates on the dataset
* [x] Normalization on train data
* [x] Validate way of Lat Long (H,W) processed from the model but also from our dataset/plotting functions
