# bfm-finetune

## Install

You can install with poetry or with pip:

```bash
# works better
poetry install
pip install -e ".[all]"
```

If you don't have poetry you can install to the local environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install poetry
# and then install
poetry install
```

## Experimentation - Work in progress

An intro script with a toy example, using the small Aurora model and finetuning with the below logic is `finetune_new_variables.py`.

Concept:
- Input Adapter:
The new input (with, for example, 10 channels) is passed through a LoRA-based input adapter. This adapter maps the 10 channels to the 4 channels that the pretrained Aurora model expects. This is crucial for ensuring compatibility without retraining the whole encoder.

- New Decoder Head:
The modified model uses a new output head to generate high-dimensional outputs (like 10,000 species occurrences). This head is trained while the rest of the model remains frozen.

- Custom Collate Function:
The collate function just merges multiple samples into one batch. It doesn’t change how the new input is handled—it merely stacks the custom Batch objects so that the model receives them correctly.