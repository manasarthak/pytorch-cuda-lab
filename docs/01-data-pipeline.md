# 01 — Data pipeline

Turns raw melt-signal blobs in S3 into an `(N, 61)` float32 matrix with species
labels. This is the workload the GPU experiments run on.

## What the pipeline does

Raw fluorescence curves are `(n_wells, n_frames)` per chip. Each well becomes one
training sample through five steps (`gpulab/data/preprocess.py`):

1. Savitzky-Golay smooth of the raw signal (window 13, polyorder 2).
2. Negative first derivative (`-gradient`) — melt transitions become positive peaks.
3. Second Savitzky-Golay smooth on the derivative.
4. ROI crop — a length-**61** window centered on the peak found in frames 380-460.
5. Optional normalization — divide by trapezoidal area (unit AUC).

**Positive-well mining** keeps a well only if its derivative peak in frames 380-460
exceeds `4.0` **and** its centered derivative never dips below `minimum_value`.

## Steps

### 1. Configure

`.env` must have `MCC_S3_BUCKET` and `MCC_S3_EVA_DB` (see [00-setup.md](00-setup.md)).

### 2. Write a manifest

Create `configs/manifest.csv` (gitignored — chip IDs are org data):

```csv
chip_id,species
<chip-id>,E. coli
<chip-id>,S. aureus
```

Start with 3-5 chips per species to validate the pipeline before scaling up.

### 3. Build

```bash
uv run python scripts/build_dataset.py configs/manifest.csv
```

Raw curves are cached under `data/raw/<chip_id>.npy` on first fetch, so subsequent
builds are offline. Output: `data/processed/dataset.npz`.

### 4. Verify before trusting it

```python
from gpulab.data.dataset import Dataset, grouped_train_test_split
ds = Dataset.load("data/processed/dataset.npz")
print(len(ds), ds.X.shape, ds.X.dtype, ds.classes)
import collections; print(collections.Counter(ds.species))
```

Check these, in order:

| Check | Expect | If it fails |
|---|---|---|
| `ds.X.shape[1]` | `61` | ROI config changed, or raw frames < 460 |
| No NaN/Inf | `np.isfinite(ds.X).all()` | a curve had zero AUC (division blew up) |
| Curves per chip | tens to thousands | mining thresholds may be too strict/loose |
| Class balance | note it — it will be uneven | plan for balancing or class weights |
| Peak position | near index 30 (center) | peak-centering isn't working |

### 5. Look at the data

```python
from gpulab.learn import viz
y, classes = ds.y_int()
viz.plot_curves(ds.X[:200], y[:200], title="preprocessed -dF/dT")
viz.heatmap(ds.X[:100], title="100 curves")
```

Curves of the same species should visibly cluster in peak shape/width. If every
class looks identical, the task is harder than the features suggest — worth knowing
before you blame a model.

### 6. Split correctly

```python
train_idx, test_idx = grouped_train_test_split(ds, test_fraction=0.25, seed=0)
```

**Always split by chip, never by well.** Wells from one chip are highly correlated;
a random per-well split leaks and produces optimistic accuracy that collapses in
production. Verify no chip appears on both sides:

```python
set(ds.chip_id[train_idx]) & set(ds.chip_id[test_idx])   # must be empty
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `MCC_S3_BUCKET is not set` | `.env` missing or not in the working directory |
| `NoCredentialsError` | boto3 found no AWS credentials in its chain |
| `KeyError: 'y'` | the BSON document schema differs — inspect one document |
| Blob won't decode | not concatenated BSON; try `bson.decode_file_iter` on a stream |
| `No positive curves found` | thresholds too strict, or frames 380-460 is the wrong window for this data |

The frame window (380-460), peak threshold (4.0), and slice length (61) are the
assumptions most likely to differ across instruments. They're constants at the top
of `gpulab/data/preprocess.py`.

## Scale note

With thousands of curves per organism the whole dataset is still small: 1M curves x
61 floats x 4 bytes is about **244 MB**. It fits entirely in 16 GB of VRAM, which is
what makes the GPU experiments in [04-cuda-deep-dive.md](04-cuda-deep-dive.md)
possible without an input pipeline in the way.
