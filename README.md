# bfm-finetune

## Install

### If you're on Snellius

On Snellius you need to execute the following first in order to use `python3.11` / `python3.12`:

```bash
module purge
# 3.11
module load 2023 Python/3.11.3-GCCcore-12.3.0
# or 3.12
module load 2024 Python/3.12.3-GCCcore-13.3.0
```

### If you don't have a recent poetry installed

If you don't have poetry you can install to the local environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install poetry
# and then install
poetry install
```

### Installing python dependencies

You can install with poetry (plain pip does not respect the specific versions):

```bash
poetry install
```

### Installing pre-commit hooks

To have code properly formatted and linted, we use `pre-commit`.
It's a dev dependency that you will find installed after following the previous steps.

To automatically run `pre-commit` before any commit, you need to install the git hooks with:

```bash
pre-commit install
```

### Manually run code formatting / pre-commit

You can manually run the command on all the files (even if not modified) with:

```bash
pre-commit run --all-files
```

## Running the code

## Run some finetune workflows

First get some resources if you are in the cluster.
*Note: set gpus-per-node=2 or more if you planning to be faster!*
```
salloc -p gpu_h100 --gpus-per-node=1 -t 01:00:00
```

1) In an activated environment, run the script `finetune_new_variables.py`. 

2) You can select to debug your finetune models using the toy dataset by changing the flag `finetune_new_variables(use_toy=True)` 

3) Uncomment either one of the 3 Versions of the models to experiment with

4) You can do parallel training if your hardware supports it, by running the command `finetune_new_variables_multi_gpu.py`. You can edit the `finetune_config.yaml` to support your settings, e.g. fsdp vs ddp or the gpus ids [0,1].

## Visualise predictions

You can visualise the predictions of the finetuned model by using the notebook `visualise_eval.ipynb`. Just change the **PATH** variable to map the location of your checkpoint.

## Experimentation - Work in progress

An intro script with a toy example, using the small Aurora model and finetuning with the below logic is `finetune_new_variables.py`.

Concept:
- Input Adapter:
The new input (with, for example, 10 channels) is passed through a LoRA-based input adapter. This adapter maps the 10 channels to the 4 channels that the pretrained Aurora model expects. This is crucial for ensuring compatibility without retraining the whole encoder.

- New Decoder Head:
The modified model uses a new output head to generate high-dimensional outputs (like 10,000 species occurrences). This head is trained while the rest of the model remains frozen.

- Custom Collate Function:
The collate function just merges multiple samples into one batch. It doesn’t change how the new input is handled—it merely stacks the custom Batch objects so that the model receives them correctly.

*NOTE: We are currently using the Aurora small for integration experiments. In the future we will adapt the codebase for using the BFM.*

## TODOs

* [x] Monitoring & Logging
* [x] Checkpointing & Loading
* [x] Result visualisation
* [ ] Compare with baselines
* [x] Upsample to (721, 1440) earth grid in the encoder and downsample to (152, 320) in decoder. Edit the coordinates on the dataset
* [x] Normalization on train data 
* [ ] Validate way of Lat Long (H,W) processed from the model but also from our dataset/plotting functions 