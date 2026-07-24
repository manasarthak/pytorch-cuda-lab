# 03 — Neural training

Work through `GUIDE.md` Steps 1-8 first — write your own loop before reading
`gpulab/train/loop.py`. This document is the protocol for running the experiments
once the loop works.

## Run a training job

```python
from gpulab.data.dataset import Dataset, grouped_train_test_split
from gpulab.models.cnn1d import CNN1D
from gpulab.train.loop import TrainConfig, train

ds = Dataset.load("data/processed/dataset.npz")
y, classes = ds.y_int()
tr, te = grouped_train_test_split(ds, test_fraction=0.25, seed=0)

model = CNN1D(n_classes=len(classes), roi_len=ds.X.shape[1], head="flatten")
out = train(model, ds.X[tr], y[tr], ds.X[te], y[te], TrainConfig(epochs=50, batch_size=512))
```

The loop keeps the whole dataset resident on the GPU and slices batches on-device.
That is deliberate: it removes the input pipeline as a variable so the CUDA
measurements in [04-cuda-deep-dive.md](04-cuda-deep-dive.md) are about compute.

### The head choice decides whether the model can use Tm

The data uses a **fixed, unaligned window (350-470)** so the melt temperature (Tm =
peak position) is preserved (see [01-data-pipeline.md](01-data-pipeline.md)). Whether
the model can actually *use* Tm depends entirely on its head:

| Head | Sees peak position (Tm)? | Use with |
|---|---|---|
| `head="flatten"` (default) | **yes** — `Flatten -> Linear` keeps position | fixed unaligned ROI |
| `head="avg"` | no — global average pooling is translation-invariant | peak-centered ROI (`roi=None`) |

Pairing the fixed window with an `avg` head silently throws Tm away inside the
network — the exact information the fixed window was meant to keep. A worthwhile
experiment: train `flatten` vs `avg` on the same fixed-window data and measure the
gap. That gap *is* the value of Tm for this task.

## Protocol

1. **Overfit a tiny subset first.** Take 50 curves and train until training accuracy
   hits ~100%. If it can't, the model or loop is broken — debug before scaling.
   This is the single highest-value sanity check in deep learning.
2. **Then train properly**, same chip-grouped split as the baseline.
3. **Seed sweep** (`seed=0..4`), report mean ± std, exactly as in
   [02-baselines.md](02-baselines.md). Compare like for like.
4. **Check gradient health** after the first backward pass:
   ```python
   from gpulab.learn import inspect as I, viz
   I.grad_norms(model); viz.plot_grad_flow(model)
   ```
   Every trainable parameter should have a non-`None`, non-zero gradient.
5. **Plot the history** — `viz.plot_history(out["history"])`. Look for the usual
   diagnoses below.

## Reading training curves

| Pattern | Diagnosis | Action |
|---|---|---|
| Train loss flat from epoch 0 | LR too low, or gradients not reaching layers | check `grad_norms`, raise LR |
| Loss to NaN | LR too high, or fp16 overflow | lower LR; use bf16 not fp16 |
| Train acc high, test acc low | overfitting, or a leaky split | verify chip-grouped split first |
| Both plateau below baseline | model underpowered or data-limited | widen/deepen; revisit features |
| Test acc noisy across epochs | test set too small (few chips) | more chips, or report mean over seeds |

## Hyperparameters worth sweeping

Sweep one at a time and record each. In rough order of impact:

1. Learning rate (`1e-2` → `1e-4`, log scale) — almost always the biggest lever.
2. Batch size (64 → 4096) — interacts with LR; also the axis for module M6.
3. Model width/depth — the honest test of whether capacity is the limit.
4. Weight decay, then LR schedule (cosine, step).

## Beating the baseline

If the CNN does not beat XGBoost, that is a legitimate result, not a failure.
Report it. Likely explanations, in order of probability:

- The task is dominated by peak position and height, which the features already
  capture explicitly.
- Too few chips — deep models need more data diversity than 13 features do.
- The ROI window (350-470) has discarded discriminative signal outside it — widen
  it and rebuild. Or the head is pooling away Tm (use `head="flatten"`).

The third is testable: rebuild the dataset with a wider ROI (a `(start, end)` tuple
instead of peak-centering) and rerun both models.

## Next model up

ResNet1D / InceptionTime — residual connections address exactly the vanishing-
gradient pattern you can see in `plot_grad_flow` on a deeper plain CNN. Building it
is Rung 3 in `PLAN.md`.
