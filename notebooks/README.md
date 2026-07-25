# notebooks/ — the learning playground

This is where you **actually learn the concepts** — run cells, tweak them, break
them. The `docs/` files are the reference you read alongside; these notebooks are the
executable version where you do the work.

## How to run

```bash
uv sync --extra torch     # torch only needed for Phase 2
uv run jupyter lab        # or: open the repo in VS Code and pick the .venv kernel
```

Always run in the project's `.venv` (where `gpulab` lives), not your base conda env.

## How each notebook is built

Every concept appears as: **demo** (run it, watch) → **your turn** (a `# TODO` you
write) → **check** (often `np.allclose` against the repo's version). Don't skip the
TODOs — running finished code is not the same as learning it. Each notebook ends with
a **"Concepts to note"** block; copy those into your own theory notebook.

## Sequence

### Phase 1 — data foundations (built, do these first)

| Notebook | Learn | Paired doc |
|---|---|---|
| `00_orientation.ipynb` | environment check, how to use these | — |
| `01_numpy_foundations.ipynb` | arrays, axes, views, broadcasting, masks, gradient/trapezoid | `docs/08` A-B |
| `02_pandas_dataframes.ipynb` | DataFrames, manifests, `groupby`, grouped splits, labels | `docs/08` C-D |
| `03_preprocessing_pipeline.ipynb` | rebuild the melt-curve transform + mining, diff vs repo | `docs/01`, `docs/08` |
| `04_visualization.ipynb` | plotting curves, heatmaps, EDA, the `viz` helpers | `docs/01` step 5 |

### Phase 2 — PyTorch & CUDA (built; needs `uv sync --extra torch`)

CUDA-graceful: they run on CPU (printing a note where a GPU is required), but the
timing/memory numbers are only meaningful on your RTX 3060 — run them there.

| Notebook | Learn | Source material |
|---|---|---|
| `05_tensors_autograd.ipynb` | tensors, storage/views, dtype/device, autograd graph | GUIDE 1-4 |
| `06_module_training_loop.ipynb` | `nn.Module`, grad flow, the loop, overfit-check | GUIDE 5-8 |
| `07_cuda_utilization.ipynb` | whole-dataset-on-GPU, launch vs compute bound, batch sweep, AMP | GUIDE 9-10, M0-M6 |
| `08_profiling_memory.ipynb` | allocator (alloc vs reserved), sync traps, profiler | M7-M9, `docs/05` |
| `09_fusion_kernels_precision.ipynb` | `torch.compile`, Savitzky-Golay as a conv kernel, precision | M10-M12 |

The M11 cell (`09`) validates the Savitzky-Golay-as-`conv1d` idea against scipy and
runs on CPU — a real, checkable result even without a GPU.

Track concept coverage across everything in [`docs/09-concept-map.md`](../docs/09-concept-map.md).
