# Documentation

Runbooks for this lab. `GUIDE.md` (repo root) teaches PyTorch concepts step by
step; these documents tell you **how to actually run things** and how to run the
GPU/CUDA experiments rigorously.

## Read in this order

| Doc | What it covers |
|---|---|
| [00-setup.md](00-setup.md) | Windows environment, CUDA-enabled torch, verifying the GPU, troubleshooting |
| [01-data-pipeline.md](01-data-pipeline.md) | Configure S3, build and verify the dataset, EDA |
| [02-baselines.md](02-baselines.md) | Run the classical baselines, record the accuracy bar |
| [03-neural-training.md](03-neural-training.md) | Train the neural models; experiment protocol |
| [04-cuda-deep-dive.md](04-cuda-deep-dive.md) | **The GPU/CUDA mechanism experiments** — 12 modules |
| [05-profiling.md](05-profiling.md) | `torch.profiler`, Nsight Systems/Compute, `nvidia-smi` workflows |
| [06-gpu-concepts.md](06-gpu-concepts.md) | Reference: hardware model, memory hierarchy, terminology |
| [07-experiment-log.md](07-experiment-log.md) | Template for recording every run |

## How to use these

Each experiment is written as **Question → Run → Measure → Expect → Why it
matters**. The "Expect" numbers are ballparks for an RTX 4060 Ti (Ada, 16 GB) —
treat them as hypotheses to confirm or refute, not facts. Measuring something that
contradicts the expectation is the most useful outcome; write down why.

Record every run in `07-experiment-log.md`. A number without its configuration
(batch size, dtype, model, driver) is not a result.
