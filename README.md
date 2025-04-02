# bfm-finetune

## Install

You can install with poetry (plain pip does not respect the specific versions):

```bash
poetry install
```

If you don't have poetry you can install to the local environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install poetry
# and then install
poetry install
```

On snellius you need to execute the following first in order to use `python3.11` / `python3.12`:

```bash
module purge
# 3.11
module load 2023 Python/3.11.3-GCCcore-12.3.0
# or 3.12
module load 2024 Python/3.12.3-GCCcore-13.3.0
# then create venv and install poetry and deps (as above)
```

## File formatting - pre-commit

To keep the code clean, `pre-commit` is installed as dev dependency. In this way, every time before a commit, the files modified will be formatted.
You can manually run the command on all the files (even if not modified) with:

```bash
pre-commit run --all-files
```

## Run some finetune workflows

1) In an activated environment, run the script `finetune_new_variables.py`. 

2) You can select to debug your finetune models using the real dataset by changing the flag `finetune_new_variables(use_toy=True)` 

3) Uncomment either one of the 3 Versions of the models to experiment with


## Experimentation - Work in progress

An intro script with a toy example, using the small Aurora model and finetuning with the below logic is `finetune_new_variables.py`.

Concept:
- Input Adapter:
The new input (with, for example, 10 channels) is passed through a LoRA-based input adapter. This adapter maps the 10 channels to the 4 channels that the pretrained Aurora model expects. This is crucial for ensuring compatibility without retraining the whole encoder.

- New Decoder Head:
The modified model uses a new output head to generate high-dimensional outputs (like 10,000 species occurrences). This head is trained while the rest of the model remains frozen.

- Custom Collate Function:
The collate function just merges multiple samples into one batch. It doesn’t change how the new input is handled—it merely stacks the custom Batch objects so that the model receives them correctly.

*Limitations: We are currently using the Aurora small for integration experiments. In the future we will adapt the codebase for using the BFM.*

## TODOs

* [ ] Monitoring & Logging
* [ ] Checkpointing & Loading
* [ ] Result visualisation & comparing with baselines