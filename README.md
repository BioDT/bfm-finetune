# bfm-finetune

Evaluation and fine-tuning benchmarks for [BioAnalyst](https://github.com/BioDT/bfm-model)
(BFM), the Biodiversity Foundation Model. The core of the repository is the **eumon
benchmark**: three biotic forecasting tasks and one abiotic (CHELSA) task, scored against
pre-registered null models and classical baselines on a four-setting evaluation ladder for
BioAnalyst and Aurora. Earlier fine-tuning experiments are retained as legacy tasks.

## Repository layout

```
bfm-finetune/
├── bfm_finetune/
│   └── eumon/            # the benchmark package (panels, nulls, baselines, ladder)
├── scripts/              # run.py, audit.py, preflight.py, l3_smoke.py, campaign.sh
├── tests/                # unit tests (tests/test_eumon.py is CPU-only)
├── bfm-model/            # submodule: model code, config, scaling statistics
├── gravity-wave-finetuning/  # submodule: Prithvi-WxC experiments (optional)
├── data/                 # raw archives (data/raw/…) and BioCube batches — not tracked
├── weights/              # bfm-pretrain-large.ckpt — not tracked
└── artefacts/            # everything the pipeline writes — not tracked
```

## Installation

| | |
|---|---|
| Python | `>=3.12,<3.14` |
| Poetry | `>=2.0` — installed into the venv by `initialize.sh` |
| Git | submodules resolve over HTTPS; no SSH access required |
| GPU | CUDA for the fine-tuning arms; nulls and classical baselines are CPU-only |

### Quick start

```bash
git clone --recurse-submodules https://github.com/BioDT/bfm-finetune.git
cd bfm-finetune
./initialize.sh
```

`initialize.sh` is idempotent: it initialises the submodules, creates the in-project
`.venv`, installs Poetry into it, runs `poetry install` and installs the pre-commit hooks.
`--with-prithvi` adds the Prithvi-WxC dependency group; `--with-assets` also fetches the
legacy task assets (large downloads — not needed for the benchmark).

### Manual setup

```bash
git submodule update --init --recursive
python3.12 -m venv .venv                 # create the venv first; Poetry then uses it
.venv/bin/pip install --upgrade pip poetry
.venv/bin/poetry install                 # add --with prithvi for the Prithvi-WxC experiments
```

Poetry is configured for an in-project virtual environment (`poetry.toml`). `bfm-model`
is installed editable from the submodule; to update it, pull inside `bfm-model/` and
re-run `poetry install`.

## Data and weights

Every benchmark input — the three survey archives, the published indices and the
BioAnalyst weights — is pinned by URL and SHA-256 in `bfm_finetune/eumon/download.py`:

```bash
.venv/bin/python -m bfm_finetune.eumon.download fetch-all   # download whatever is missing
.venv/bin/python -m bfm_finetune.eumon.download verify      # check hashes, download nothing
```

The BioCube monthly batches are expected under `data/batches_28species/`
(override with `EUMON_BIOCUBE`). CHELSA months are window-read over HTTP on demand and
cached under the artefacts root.

## Benchmark (`eumon`)

An effort-controlled biodiversity benchmark: null models and learned baselines are
scored before any foundation model, on a four-setting ladder — L0 zero-shot, L1 calibration,
L2 frozen probe, L3 PEFT / full fine-tune — for BioAnalyst and Aurora.

Run from the repository root:

```bash
.venv/bin/python scripts/run.py --list           # stages and what each one writes
.venv/bin/python scripts/run.py all              # everything, in dependency order
.venv/bin/python scripts/run.py l2 l3 --gpu 1    # named stages on a chosen device
.venv/bin/python scripts/audit.py                # read-only checks; expect 0 FAIL
.venv/bin/python scripts/preflight.py --gpu 1    # tiny end-to-end pass over every setting
```

| variable | effect |
|---|---|
| `EUMON_ROOT` | project root for data, weights, `bfm-model` and outputs (default: this checkout) |
| `EUMON_ARTEFACTS` | output root (default `artefacts/`, relative to `EUMON_ROOT`); redirects a whole run |
| `EUMON_BIOCUBE` | BioCube batch directory (default `$EUMON_ROOT/data/batches_28species`) |
| `EUMON_THREADS` | CPU thread cap applied before torch is imported (default 8) |
| `EUMON_NO_ENERGY` | `1` skips per-card power sampling and records wall-clock only |

Stages are idempotent: each declares its outputs, is skipped when they exist with a
matching SHA-256, and is safe to re-run after a crash. GPU stages refuse to start on an
occupied card unless `--allow-shared` is passed, because per-device energy measurement is
otherwise silently corrupted.

## Development

```bash
.venv/bin/pytest tests/test_eumon.py             # CPU-only; no data or weights needed
.venv/bin/ruff check bfm_finetune/eumon scripts  # lint
.venv/bin/pre-commit run --all-files             # formatting hooks
```

## Legacy experiments

Superseded by the eumon benchmark and kept for reference. They read their data root from
`bfm_finetune/paths.py` (`STORAGE_DIR`: a Snellius project path on that cluster, otherwise
`./data`); fetch their assets with `./initialize.sh --with-assets`.

### GeoLifeCLEF-24 species distribution
Recreate the yearly batches with `python bfm_finetune/dataloaders/geolifeclef_species/batch.py`,
train with `python bfm_finetune/finetune_bfm_sdm.py`, and visualise predictions in
`notebooks/geolifeclef_species.ipynb`.

### CHELSA climate probing
Recreate the batches with `python bfm_finetune/dataloaders/chelsa/batch.py` and train with
`python bfm_finetune/finetune_chelsa.py`. The eumon `abiotic*` stages re-examine this
experiment with a proper split and null battery.

### Aurora new-variable fine-tuning
`python bfm_finetune/finetune_new_variables.py` (toy dataset via `use_toy=True`);
multi-GPU variant in `finetune_new_variables_multi_gpu.py`, configured through
`bfm_finetune/finetune_config.yaml`.

### Prithvi-WxC gravity-wave fine-tuning
Requires `poetry install --with prithvi`. Train with `bfm_finetune/prithvi/train.sh`,
run inference with `bfm_finetune/prithvi/inference.sh`.

### Resources

- [BioAnalyst code](https://github.com/BioDT/bfm-model)
- [BioAnalyst weights](https://huggingface.co/BioDT/bfm-pretrained)
- [BioCube data](https://huggingface.co/datasets/BioDT/BioCube)
- [BioCube code](https://github.com/BioDT/bfm-data)

## Citation

If you like our work and used it in any context, please consider citing us as follows:

**BioAnalyst**
```
@misc{trantas2025bioanalystfoundationmodelbiodiversity,
      title={BioAnalyst: A Foundation Model for Biodiversity}, 
      author={Athanasios Trantas and Martino Mensio and Stylianos Stasinos and Sebastian Gribincea and Taimur Khan and Damian Podareanu and Aliene van der Veen},
      year={2025},
      eprint={2507.09080},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.09080}, 
}
```

**BioCube**
```
@article{stasinos2025biocube,
  title={Biocube: A multimodal dataset for biodiversity research},
  author={Stasinos, Stylianos and Mensio, Martino and Lazovik, Elena and Trantas, Athanasios},
  journal={arXiv preprint arXiv:2505.11568},
  year={2025}
}
```