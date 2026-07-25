# 08 — Data-layer lab (NumPy + pandas, see-it-yourself)

`GUIDE.md` makes you *write* the PyTorch. This doc does the same for the data layer
you were handed pre-written (`preprocess.py`, `dataset.py`, `features.py`). The
method is identical: **reimplement each piece yourself, then diff against the repo's
version with `np.allclose`.** If your version matches, you understood it; if it
doesn't, the gap is the lesson.

The runnable version of this lab is in `notebooks/01`-`04` — do the exercises there;
this doc is the written reference for them.

You already built a dataset, so use it as the playground:

```python
import numpy as np, pandas as pd
from gpulab.data.dataset import Dataset
ds = Dataset.load("data/processed/dataset.npz")          # (N, 120) curves + labels
raw = np.load("data/raw/<some_chip>.npy")                # (n_wells, n_frames) raw
```

Rule for every exercise: **write it from the concept, not by copying the source.**
Open the repo file only to diff *after* you've tried.

---

## Part A — NumPy signal ops

The core transform is `smoothed_negative_derivative` (`preprocess.py`): smooth →
`-gradient` → smooth.

**A1. Finite differences (`np.gradient`).**
Write it: for a 1-D `curve`, compute the derivative by hand as
`(curve[2:] - curve[:-2]) / 2` for the interior. Compare to `np.gradient(curve)`.
Look for: `np.gradient` uses **central differences** in the interior and one-sided
at the edges, so lengths match the input. Ask yourself why the melt pipeline negates
it (`-gradient`) — what does a *falling* fluorescence curve's negative slope become?

**A2. Savitzky-Golay is a fixed filter (`scipy.signal.savgol_filter`).**
Write it: smooth a curve two ways — a plain moving average (`np.convolve(curve,
np.ones(w)/w, mode="same")`) and `savgol_filter(curve, 13, 2)`. Plot both over the
raw. Look for: SG fits a **degree-2 polynomial** in each sliding window, so it
preserves peak height and width far better than a box average (which flattens
peaks). Key realization for later: SG is a **fixed convolution kernel** — the same
coefficients every window. That's why M11 can reimplement it as a `conv1d` on the
GPU. Print `scipy.signal.savgol_coeffs(13, 2)` and see the kernel.

**A3. `axis=` is the whole game.**
Write it: `savgol_filter(raw, 13, 2, axis=1)` smooths each well across time. Now
try `axis=0` and look at the shape/meaning. Look for: axis=1 = "along frames, per
well"; axis=0 = "across wells, per frame" (wrong here). Almost every bug in array
code is a wrong axis.

**A4. Reassemble the pipeline.**
Write your own `smoothed_negative_derivative(raw)` from A1-A3, then:
```python
from gpulab.data.preprocess import smoothed_negative_derivative, PreprocessConfig
mine = my_version(raw)
theirs = smoothed_negative_derivative(raw, PreprocessConfig())
print(np.abs(mine - theirs).max())    # want ~0
```

---

## Part B — Indexing, slicing, broadcasting, masking

This is the densest concept cluster and the most transferable to PyTorch (tensor
views in `GUIDE.md` Step 1 are the same idea).

**B1. Argmax within a window + edge padding (`center_peaks`).**
Write it: for one row, find the peak *inside frames 380-460* with
`380 + np.argmax(row[380:460])`, take a 61-wide window centered on it, and
edge-pad if it runs off the end (`np.pad(slice, (l, r), mode="edge")`). Handle the
clamp (`max(0, start)`, `min(len, end)`). Diff against `center_peaks`. Look for:
why search a *sub-window* for the peak instead of the global argmax? (Noise spikes
outside the melt region.) This is also where you feel slicing = a **view**, not a
copy — the same lesson as torch strides.

**B2. Boolean masks and `np.where` (positive mining).**
Write it: reproduce `positive_mask` — `has_peak = deriv[:, 380:460].max(axis=1) >
4.0`, combine with `&`, then `np.flatnonzero(mask)` for the kept indices. Look for:
`max(axis=1)` collapses the frame axis → one value per well; `&` is elementwise AND
on boolean arrays (use `&`, never `and`); `np.flatnonzero` = `np.where(mask)[0]`.
This is *exactly* how you'll index batches on the GPU later.

**B3. Broadcasting and `np.newaxis` (AUC normalization).**
Write it: `auc = np.trapezoid(roi, axis=1)` gives shape `(N,)`; now divide each row
by its own AUC. Try `roi / auc` (fails or wrong) then `roi / auc[:, np.newaxis]`.
Look for: broadcasting aligns shapes from the **right**; `(N,120) / (N,)` mismatches,
`(N,120) / (N,1)` broadcasts across columns. `np.newaxis` (= `None`) inserts a
length-1 axis. This rule is identical in PyTorch — learn it once here.

**B4. `np.trapezoid` (integration).**
Write it: integrate a curve by hand as `np.sum((y[:-1]+y[1:])/2)` and compare to
`np.trapezoid(y)`. Look for: it's the trapezoidal rule; dividing by it gives every
curve unit area, removing amplitude while keeping shape and peak position (Tm).

**B5. dtype and copies (`np.asarray`, float32 vs float64).**
Write it: check `raw.dtype`, force `np.asarray(raw, dtype=np.float64)`, and note
where the pipeline casts to `float32` before stacking (memory). Look for: float32
halves memory and is what the GPU wants; float64 is numpy's default and safer for
the intermediate math. When does `np.asarray` copy vs return the same object?

---

## Part C — pandas: the manifest and grouped splits

**C1. Build a DataFrame from records (`read_manifest`).**
Write it: load your `manifest.json` with `json.loads`, build rows, and make a
`pd.DataFrame(rows, columns=["chip_id","species"])`. Diff columns/shape against
`read_manifest`. Look for: `dtype=str` on `read_csv`, `df.rename(columns=...)`,
`df.dropna(subset=...)`, and set algebra on `df.columns` for validation.

**C2. Iteration: `itertuples` vs the alternatives.**
Write it: iterate your manifest three ways — `iterrows()`, `itertuples(index=False)`,
and a vectorized column access `df["chip_id"].to_numpy()`. Time them. Look for: why
`build_dataset` uses `itertuples` (fast, named, C-backed) over `iterrows` (slow,
boxes each row into a Series). Vectorized beats both when you can avoid the loop.

**C3. `groupby` + `unique` (the chip-grouped split).**
Write it: reproduce `grouped_train_test_split`'s core — `df.groupby("species")`,
take `.unique()` chips per species, shuffle with `np.random.default_rng(seed)`, and
assign whole chips to train/test. Look for: **why group by chip** (wells within a
chip leak); how `groupby` yields `(key, subframe)` pairs; `default_rng(seed)` for
reproducible shuffles. Verify no chip appears on both sides (a `set(...) & set(...)`).

**C4. `collections.Counter` (class balance).**
Write it: `from collections import Counter; Counter(ds.species)`. Look for: your 14
species and how uneven they are — the number that will make per-class metrics noisy.

---

## Part D — Labels and serialization

**D1. Integer-encode vs one-hot.**
Write it: build `classes = sorted(set(species))`, a `{name: i}` dict, and map to an
`int64` array (this is `Dataset.y_int`). Then make the one-hot with
`np.eye(len(classes))[y]`. Look for: XGBoost wants integer labels; the Keras models
used one-hot; PyTorch `CrossEntropyLoss` wants **integer** targets (not one-hot) —
a common beginner trap.

**D2. `savez_compressed` / `load`.**
Write it: `np.savez_compressed("tmp.npz", X=ds.X, y=y)` then `np.load("tmp.npz")`.
Look for: `.npz` is a zip of named arrays; `allow_pickle=True` is needed for object
(string) arrays and is a security note worth understanding (never load untrusted
pickles).

---

## Part E — Feature engineering and statistics (`features.py`)

**E1. Distribution moments by hand.**
Write it: for one curve, treat it as a weighted distribution over frame index and
compute mean, variance, skew, kurtosis from the normalized weights (as
`curve_features` does). Diff against the repo. Look for: what skew/kurtosis of a
melt peak actually mean (asymmetry, tailedness) and why moments summarize shape.

**E2. Peak detection (`scipy.signal.find_peaks`, `peak_widths`).**
Write it: run `find_peaks(curve, height=...)` and `peak_widths(...)` and plot the
detected peaks and their FWHM on the curve. Look for: how a multi-modal melt curve
(two peaks) shows up in `n_peaks` — a feature that can separate species.

**E3. Feature matrix from records.**
Write it: turn a list of per-curve feature dicts into an `(N, n_features)` float32
matrix (`feature_matrix`). Look for: the dict-of-features → matrix pattern, and why
you keep the feature-name list aligned with the columns.

---

## When you're done

You should be able to, from a blank file, rebuild the entire `raw -> X` transform
and the manifest/split logic without looking, and explain every `axis=`, every
broadcast, and why each `dtype` is what it is. Then `preprocess_gpu.py` in the CUDA
deep-dive (M11) is just this same math expressed as GPU kernels — you'll already
know exactly what it has to compute.

Track coverage in [09-concept-map.md](09-concept-map.md).
