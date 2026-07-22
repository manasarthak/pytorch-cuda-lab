# Learning roadmap

Goal: learn **PyTorch and CUDA utilization** while training **robust, simple models**
for melt-curve species classification. Same 61-point dataset at every rung, so
results are directly comparable. Classical models set the accuracy bar; the neural
track is where the CUDA learning happens.

## The data reality that shapes the CUDA lessons

There are **thousands of curves per organism**, but each curve is only 61 floats.
Even 1M curves is ~244 MB, so the **entire dataset fits in the 4060 Ti's 16 GB**.
That flips the usual lessons:

- Hold `X` as one resident CUDA tensor; slice batches on-device (no DataLoader).
- With a tiny model you'll see training is **kernel-launch bound, not compute
  bound** — GPU utilization stays low regardless of batch size. Feeling this is the
  single most useful CUDA lesson here.
- To actually **saturate** the GPU, add compute: wider/deeper nets, big batches +
  AMP, or the 2D **recurrence-plot** representation (61-pt curve → image → 2D CNN).

## Rungs

- [ ] **Rung 0 — Data + EDA.** `s3_source.py` + `preprocess.py` + `dataset.py`.
      Build and cache a dataset; plot curves per species; sanity-check class balance
      and positive-well counts. *(Scaffolded and unit-tested.)*
- [ ] **Rung 1 — Classical baseline.** `features.py` + `models/classical.py`.
      XGBoost on hand-crafted features. Add **MiniROCKET + ridge** as a second
      baseline (random conv kernels — near-SOTA on time series, trains in seconds).
      Record the accuracy bar. *(XGBoost scaffolded.)*
- [ ] **Rung 2 — First PyTorch model (CUDA core).** `models/cnn1d.py` + `train/loop.py`.
      Learn: tensors/devices, the training loop, `torch.autocast` (bf16 AMP),
      `torch.cuda` memory, profiling with `train/cuda_utils.py`. Compare GPU
      utilization at batch 64 vs 4096 and explain what you see. *(Scaffolded.)*
- [ ] **Rung 3 — Scale the net.** `models/resnet1d.py` (ResNet1D / InceptionTime port).
      Learn: residuals + BatchNorm, LR schedules, early stopping, checkpointing,
      TensorBoard, seeding/reproducibility, `torch.compile`.
- [ ] **CUDA deep-dive.** `data/preprocess_gpu.py`: reimplement the *exact*
      preprocessing as `conv1d` ops on GPU (Savitzky-Golay = a fixed FIR filter =
      convolution; `gradient` = a finite-difference conv; AUC = a reduction).
      Validate bit-for-bit against the scipy version. Then push utilization with the
      recurrence-plot 2D CNN and large batches.

## Deliberately out of scope (for now)

- Transformers / HF time-series foundation models. On 61-point curves they're a
  learning stretch, not an accuracy play — revisit only after Rung 3 if curious.
- Open-set / unknown-species rejection.

## Notes to self

- Windows `DataLoader(num_workers>0)` uses `spawn` → guard entry points with
  `if __name__ == "__main__":`. With whole-dataset-on-GPU you'll often use
  `num_workers=0` anyway.
- bf16 AMP on Ada needs no gradient scaler; fp16 would.
- Always split **by chip**, never by well, or accuracy will be optimistically wrong.
