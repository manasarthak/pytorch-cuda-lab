# pytorch-cuda-lab

A hands-on lab for learning **PyTorch internals and GPU/CUDA mechanisms**, using
real time-series classification as the workload. Built for a Windows box with an
RTX 4060 Ti 16 GB (Ada, `sm_89`) and a Ryzen 5 5600X.

The goal is not just to train a model — it's to understand *what the GPU is
actually doing*: memory hierarchy and coalescing, kernel launch overhead,
occupancy, bandwidth vs compute limits, streams and asynchrony, mixed precision,
fusion, and profiling. The classification task is the vehicle that makes those
measurable on real data.

This repo is standalone — no dependency on any organization's code. It reads raw
curves from S3 with plain `boto3`, reimplements the preprocessing from scratch, and
owns its own models. Bucket names and credentials live only in a gitignored `.env`,
never in source.

> The data itself is not part of this repo (see `.gitignore`). The code is written
> to run against org data on an org machine, but the code is mine.

## The workload

Each well of a qPCR run gives a fluorescence-vs-temperature curve. Preprocessing
turns it into a smoothed **negative derivative** (-dF/dT) and crops a fixed,
**unaligned** window (frames 350-470, length **120**) — the melt "peak" left at its
true position so the melt temperature (Tm) stays a feature. The label is the
bacterial **species**. See `src/gpulab/data/preprocess.py` for the exact recipe.

Why it's a good CUDA teaching workload: there are thousands of curves per organism,
but each is only 120 floats — so **the entire dataset fits in VRAM** (1M curves is
~480 MB). That removes the input pipeline as a variable and puts the interesting
GPU behaviour (launch overhead, occupancy, fusion) directly under the microscope.

## Setup

```bash
uv sync
cp .env.example .env     # then fill in MCC_S3_BUCKET / MCC_S3_EVA_DB
uv add torch --index-url https://download.pytorch.org/whl/cu126   # Ada = sm_89
```

Full instructions and troubleshooting: [`docs/00-setup.md`](docs/00-setup.md).

## Workflow

```bash
# Build a dataset from a chip manifest (chip_id,species). Fetches + caches from S3.
uv run python scripts/build_dataset.py configs/manifest.csv

# Classical baseline (the bar to beat).
uv run python scripts/run_baseline.py data/processed/dataset.npz

# Tests (preprocessing only; no S3 or torch needed).
uv run pytest
```

## Documentation

| Doc | What |
|---|---|
| [`GUIDE.md`](GUIDE.md) | 12-step see-it-yourself PyTorch lab — you write the code, helpers reveal the internals |
| [`docs/04-cuda-deep-dive.md`](docs/04-cuda-deep-dive.md) | **12 GPU/CUDA experiment modules** — launch overhead, bandwidth, Tensor Cores, roofline, occupancy, allocator, streams, fusion, custom kernels |
| [`docs/06-gpu-concepts.md`](docs/06-gpu-concepts.md) | Reference: execution model, memory hierarchy, occupancy, precision formats |
| [`docs/05-profiling.md`](docs/05-profiling.md) | `torch.profiler`, Nsight Systems/Compute, `nvidia-smi` workflows |
| [`docs/`](docs/README.md) | Full index: setup, data pipeline, baselines, training, experiment log |
| [`PLAN.md`](PLAN.md) | Overall roadmap (rungs 0-3 + CUDA deep-dive) |

## Layout

| Path | What |
|---|---|
| `src/gpulab/data/s3_source.py` | generic boto3 fetch of `.bson.gz` blobs |
| `src/gpulab/data/preprocess.py` | the 5-step recipe + positive-well mining |
| `src/gpulab/data/dataset.py` | manifest -> cached raw -> mined `Dataset`, chip-grouped split |
| `src/gpulab/features.py` | hand-crafted features for classical models |
| `src/gpulab/models/classical.py` | XGBoost baseline |
| `src/gpulab/models/cnn1d.py` | from-scratch 1D-CNN |
| `src/gpulab/train/loop.py` | whole-dataset-on-GPU training loop with AMP |
| `src/gpulab/train/cuda_utils.py` | device/memory/timing helpers |
| `src/gpulab/learn/` | inspection + visualization helpers (tensor internals, autograd, CUDA) |
