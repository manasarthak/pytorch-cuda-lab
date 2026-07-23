# 02 — Classical baselines

Establish the accuracy bar before writing a single neural network. If a gradient-
boosted tree on 13 hand-crafted features matches your CNN, the CNN is not earning
its complexity.

## Run it

```bash
uv run python scripts/run_baseline.py data/processed/dataset.npz
```

This featurizes, fits XGBoost, and prints accuracy, a per-class report, and feature
importances on a chip-grouped split.

## What the features are

`gpulab/features.py` reduces each 61-point curve to shape descriptors an analyst
would recognize: peak position and height, AUC, distribution moments (mean, std,
skew, kurtosis), FWHM, peak count, minimum, and left/right area split.

## Protocol

1. **Baseline run** — defaults, `seed=0`. Record accuracy and macro-F1.
2. **Seed sweep** — repeat with `seed=1,2,3,4`. Report **mean ± std**, not a single
   number. Chip-grouped splits are high variance when chips are few; a single run
   tells you very little.
3. **Feature ablation** — drop the top feature, refit. A large drop means the model
   leans on one descriptor; that's fragility worth knowing.
4. **Class balance** — check the per-class report. Rare species with low recall are
   the real story that overall accuracy hides.

## Second baseline worth adding

**MiniROCKET + ridge** — random convolutional kernels feeding a linear classifier.
It is near state-of-the-art on time-series classification benchmarks, trains in
seconds on CPU, and has essentially no hyperparameters. It is the strongest
"simple" baseline for this data shape and a fair opponent for the neural track.

```bash
uv add sktime            # provides MiniRocket
```

Fit the transform on train only, then apply to test — fitting on all data leaks.

## Recording

Log every run in [07-experiment-log.md](07-experiment-log.md). At minimum: dataset
size, class counts, split seed, model, accuracy, macro-F1. The neural rungs are
judged against these numbers, so they need to be trustworthy.

## Reading the result

| Observation | What it means |
|---|---|
| High accuracy, low variance across seeds | task is largely solved by peak shape; neural gains will be small |
| High accuracy, high variance | too few chips; get more data before trusting any model |
| One class with poor recall | class imbalance or genuinely overlapping melt signatures |
| Feature importance dominated by `peak_idx` | the model is keying on melt temperature (Tm) — biologically sensible |

Move on to [03-neural-training.md](03-neural-training.md) once you have a bar you
believe.
