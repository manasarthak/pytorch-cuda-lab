# 00 — Environment setup

Target: Windows 11, AMD Ryzen 5 5600X, NVIDIA RTX 4060 Ti 16 GB (Ada, `sm_89`).

## 1. Verify the driver and GPU

```bash
nvidia-smi
```

Record: driver version, CUDA version reported (this is the *driver's* max supported
CUDA, not the toolkit you install), total memory, and current utilization. If this
command fails, nothing else here will work — fix the driver first.

## 2. Base environment

```bash
uv sync
```

This installs everything except PyTorch (numpy, scipy, xgboost, boto3, matplotlib).
Run the tests to confirm the base is healthy:

```bash
uv run pytest
```

## 3. PyTorch with CUDA

The CUDA wheels come from a custom index (`cu126`), which is registered in
`pyproject.toml` and scoped to torch only. So installing is just the extra:

```bash
uv sync --extra torch
```

> **Do not run `uv add torch --index-url https://download.pytorch.org/whl/cu126`.**
> `--index-url` *replaces* PyPI for the entire resolution, so `uv` can no longer
> find `boto3` and the other PyPI deps and fails with "boto3 was not found in the
> package registry." The index is already configured in `pyproject.toml` under
> `[[tool.uv.index]]` + `[tool.uv.sources]`; `uv sync --extra torch` is all you need.

Verify — **this is the gate for every CUDA experiment**:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You want `True` and your GPU name. If it prints `False`:

- You likely installed the CPU wheel. Check `torch.version.cuda` — `None` means CPU-only.
- Reinstall: `uv sync --reinstall-package torch --extra torch`.
- Confirm the driver supports your chosen CUDA version (`nvidia-smi` top-right).

## 4. Confirm the device properties

```bash
uv run python -c "
import torch
p = torch.cuda.get_device_properties(0)
print(p.name, '| SMs:', p.multi_processor_count, '| VRAM GB:', round(p.total_memory/1e9,1), '| cc:', f'sm_{p.major}{p.minor}')
"
```

Expect roughly: `NVIDIA GeForce RTX 4060 Ti | SMs: 34 | VRAM GB: 17.2 | cc: sm_89`.
Write these down — module M0 of the deep dive builds on them.

## 5. Secrets and data config

Keep the secrets file **outside the repo** so it can never be staged. Copy the
template somewhere private and point `MCC_ENV_FILE` at it (set once, per user):

```powershell
copy .env.example C:\Users\you\secrets\gpulab.env   # then fill it in
setx MCC_ENV_FILE "C:\Users\you\secrets\gpulab.env" # open a NEW shell afterward
```

Fill in `MCC_S3_BUCKET` and `MCC_S3_EVA_DB`. Resolution order: a variable already
set in your real environment wins; otherwise the file at `MCC_ENV_FILE` is loaded;
otherwise a `.env` in the current directory (gitignored) is used if present. You can
also skip the file entirely and just `setx` the two variables.

AWS credentials are resolved by boto3's normal chain (env vars, `~/.aws/credentials`,
`AWS_PROFILE`, or an instance role) — this repo neither reads nor stores them.

## 6. Optional tooling for the deep dive

| Tool | Purpose | Notes |
|---|---|---|
| **Nsight Systems** (`nsys`) | timeline profiling, CPU/GPU overlap | Free NVIDIA download; works on Windows |
| **Nsight Compute** (`ncu`) | per-kernel counters, occupancy, memory throughput | Windows OK; may need admin for counters |
| `nvidia-smi dmon` | live utilization sampling | ships with the driver |

## Windows caveats worth knowing up front

- **`torch.compile` / Triton:** Triton's Windows support is unofficial and has been
  patchy. If `torch.compile` errors or silently falls back, that's expected — run
  module M10 under **WSL2** instead, where the Linux toolchain is first-class.
- **DataLoader workers:** Windows uses `spawn`, so any script with
  `num_workers > 0` must guard its entry point with `if __name__ == "__main__":`.
  Most experiments here keep data resident on the GPU and use `num_workers=0`.
- **Console encoding:** the terminal is cp1252. All helper output in this repo is
  ASCII deliberately; if you add prints, keep them ASCII or set
  `PYTHONIOENCODING=utf-8`.

## Where to go next

[01-data-pipeline.md](01-data-pipeline.md) to get real data flowing, or jump
straight to [04-cuda-deep-dive.md](04-cuda-deep-dive.md) — the CUDA modules only
need synthetic tensors, so they run before any data exists.
