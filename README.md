# melt-curve-classifier

A personal learning project: training **classical** (XGBoost) and **PyTorch/CUDA**
models to classify bacterial species from qPCR **melt curves**, on a Windows box
with an RTX 4060 Ti (16 GB).

This repo is standalone — it has no dependency on any organization's code. It reads
raw curves from S3 with plain `boto3`, reimplements the preprocessing from scratch,
and owns its own models. Bucket names and credentials live only in a gitignored
`.env`, never in source.

> The data itself is not part of this repo (see `.gitignore`). The code is written
> to run against org data on an org machine, but the code is mine.

## What a melt curve is here

Each well gives a fluorescence-vs-temperature curve. Preprocessing turns it into a
peak-centered, smoothed **negative derivative** (−dF/dT) of length **61** — the melt
"peak". A curve is a *positive* sample if it has a real peak in the expected frame
window. The label is the **bacterial species**. See `src/mcc/data/preprocess.py`
for the exact recipe (it's faithful to the pipeline that produced the training data).

## Setup

```bash
uv sync
cp .env.example .env    # then fill in MCC_S3_BUCKET / MCC_S3_EVA_DB
```

Install PyTorch with CUDA separately (large, custom index — Ada = sm_89):

```bash
uv add torch --index-url https://download.pytorch.org/whl/cu126
```

## Workflow

```bash
# 0. Build a dataset from a chip manifest (chip_id,species). Fetches + caches from S3.
python scripts/build_dataset.py configs/manifest.csv

# 1. Classical baseline (the bar to beat).
python scripts/run_baseline.py data/processed/dataset.npz

# 2+. PyTorch CNN / ResNet — see notebooks/ and src/mcc/train/.

# Tests (preprocessing only; no S3 or torch needed):
uv run pytest
```

## Layout

| Path | What |
|---|---|
| `src/mcc/data/s3_source.py` | generic boto3 fetch of `.bson.gz` blobs |
| `src/mcc/data/preprocess.py` | the 5-step recipe + positive-well mining |
| `src/mcc/data/dataset.py` | manifest → cached raw → mined/preprocessed `Dataset`, chip-grouped split |
| `src/mcc/features.py` | hand-crafted features for classical models |
| `src/mcc/models/classical.py` | XGBoost baseline |
| `src/mcc/models/cnn1d.py` | from-scratch 1D-CNN |
| `src/mcc/train/loop.py` | whole-dataset-on-GPU training loop with AMP |
| `src/mcc/train/cuda_utils.py` | device/memory/timing helpers |

See [`PLAN.md`](PLAN.md) for the learning roadmap.
