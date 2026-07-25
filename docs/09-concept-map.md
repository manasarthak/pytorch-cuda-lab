# 09 — Concept map (so nothing gets skipped)

Every non-trivial concept the repo touches, where it lives in the code, and where
you actually *learn* it. Because the code was written for you, a concept can be
"present in the repo" but "not yet learned" — this table separates the two. Tick the
last column as you go.

Legend for **Learn in**: `G`=GUIDE.md step, `M`=04-cuda-deep-dive module,
`D`=08-data-layer-lab part, `—`=gap (see "Gaps to fill" below).

## NumPy

| Concept | In repo | Learn in | Done |
|---|---|---|---|
| Finite differences (`np.gradient`) | preprocess.py | D-A1 | ☐ |
| FIR filtering / Savitzky-Golay | preprocess.py | D-A2, M11 | ☐ |
| `axis=` reductions | everywhere | D-A3, D-B2 | ☐ |
| Slicing as a view (vs copy) | center_peaks | D-B1, G1 | ☐ |
| `argmax`, `pad(mode="edge")`, clamping | center_peaks | D-B1 | ☐ |
| Boolean masks, `&`, `flatnonzero`/`where` | positive_mask | D-B2 | ☐ |
| Broadcasting + `np.newaxis` | AUC norm | D-B3, G-broadcast | ☐ |
| Trapezoidal integration (`trapezoid`) | preprocess/features | D-B4 | ☐ |
| dtype, `asarray`, float32 vs 64 | throughout | D-B5 | ☐ |
| `concatenate`, stacking, channel axis | dataset.py | D-B5 | ☐ |
| RNG (`default_rng`, seeds) | splitting | D-C3 | ☐ |
| `savez_compressed` / `load` | dataset.py | D-D2 | ☐ |

## pandas

| Concept | In repo | Learn in | Done |
|---|---|---|---|
| DataFrame from records / `read_csv` | read_manifest | D-C1 | ☐ |
| `rename`, `dropna`, column set algebra | read_manifest | D-C1 | ☐ |
| `itertuples` vs `iterrows` vs vectorized | build_dataset | D-C2 | ☐ |
| `groupby` + `unique` | grouped split | D-C3 | ☐ |
| `Counter` for class balance | (EDA) | D-C4 | ☐ |

## Statistics / features / classical ML

| Concept | In repo | Learn in | Done |
|---|---|---|---|
| Distribution moments (skew/kurtosis) | features.py | D-E1 | ☐ |
| Peak detection (`find_peaks`, `peak_widths`) | features.py | D-E2 | ☐ |
| Feature matrix from dicts | features.py | D-E3 | ☐ |
| Integer vs one-hot labels | dataset.py | D-D1 | ☐ |
| Gradient-boosted trees (XGBoost) | models/classical | 02-baselines | ☐ |
| Chip-grouped split / leakage | dataset.py | D-C3, 01 | ☐ |
| Seed sweeps, mean±std reporting | — | 02, 03 | ☐ |
| MiniROCKET (random conv kernels) | — (to add) | 02-baselines | ☐ |

## PyTorch — core

| Concept | In repo | Learn in | Done |
|---|---|---|---|
| Tensor = storage + view; strides | — | G1 | ☐ |
| dtype / device / `.to()` | loop.py | G2, G3 | ☐ |
| Autograd graph, `backward`, `.grad` | loop.py | G4 | ☐ |
| `no_grad` / `inference_mode` | — | G4 | ☐ |
| `nn.Module`, parameter registration | cnn1d.py | G5 | ☐ |
| `state_dict` (inspect) | — | G5 | ☐ |
| CrossEntropyLoss (integer targets) | loop.py | G6 | ☐ |
| Optimizer step / `zero_grad` | loop.py | G7 | ☐ |
| Conv1d, BatchNorm, pooling | cnn1d.py | G-layers (gap G-L) | ☐ |
| Flatten vs global-pool head (Tm) | cnn1d.py | 03-neural-training | ☐ |
| Broadcasting in tensors | — | G-broadcast (gap) | ☐ |

## PyTorch — training & CUDA

| Concept | In repo | Learn in | Done |
|---|---|---|---|
| Whole-dataset-on-GPU loop | loop.py | G8, G9 | ☐ |
| AMP / autocast (bf16) | loop.py | G10, M4, M12 | ☐ |
| `torch.compile` | — | G11, M10 | ☐ |
| `torch.profiler` / traces | learn/inspect | G12, M-05 | ☐ |
| Device ceilings / roofline | — | M0, M5 | ☐ |
| PCIe transfer, pinned memory | — | M1 | ☐ |
| Kernel launch overhead, CUDA Graphs | — | M2 | ☐ |
| Memory bandwidth, coalescing | — | M3 | ☐ |
| Tensor Cores, precision formats | — | M4, M12, 06 | ☐ |
| Occupancy, batch sweep | — | M6 | ☐ |
| Caching allocator, fragmentation | — | M7 | ☐ |
| Streams, async, sync traps | learn/inspect | M8, M9 | ☐ |
| Writing a kernel (conv1d/Triton) | — | M11 | ☐ |

## Gaps to fill (genuinely not covered yet — small exercises)

These are real holes. Each is a short exercise; do them where marked.

- **G-L — Layers you use but didn't derive** (Conv1d, BatchNorm1d, ReLU, pooling).
  Exercise: on a `(2, 1, 120)` input, apply a `Conv1d(1, 4, 7, padding=3)` by hand
  with `F.conv1d` and matching weights; confirm output shape and that BatchNorm
  normalizes per-channel over the batch. Explain why BN behaves differently in
  `train()` vs `eval()`.
- **G-broadcast — tensor broadcasting/einsum.** Exercise: reproduce a batched dot
  product three ways — a loop, broadcasting, and `torch.einsum("bi,bi->b", a, b)`.
  Same rules as NumPy D-B3.
- **DataLoader / Dataset.** This repo *deliberately* skips it (data lives on the
  GPU), but it's core PyTorch. Exercise: wrap `ds.X`/`y` in a `TensorDataset` +
  `DataLoader(batch_size=512, shuffle=True, num_workers=0)`, iterate one epoch, and
  compare wall-time to the on-GPU loop. (On Windows, `num_workers>0` needs the
  `if __name__=="__main__":` guard — see 00-setup.)
- **LR schedulers.** Exercise: add `torch.optim.lr_scheduler.CosineAnnealingLR`,
  print the LR each epoch, and plot it against the loss curve.
- **Checkpointing (`state_dict` save/load).** Exercise: `torch.save(model.state_dict(),
  "ck.pt")`, build a fresh model, `load_state_dict`, and confirm identical logits.
  Note why you save `state_dict` (weights) not the pickled model object.
- **Weight init & reproducibility.** Exercise: seed everything, init two models the
  same way, confirm identical initial params; then read what `nn.Linear` uses by
  default (Kaiming uniform) and why init matters for deep nets.
- **Custom `autograd.Function` (advanced, optional).** Exercise: implement a ReLU
  as a `torch.autograd.Function` with explicit `forward`/`backward`, and check its
  gradient with `torch.autograd.gradcheck`.

## How to use this page

Work the numbered docs in the [pathway](README.md); tick concepts here as you cover
them. When every box in a domain is ticked, you've genuinely learned that layer —
not just run code that used it. The "Gaps to fill" items are the ones the pre-written
code would otherwise let you skip; don't.
