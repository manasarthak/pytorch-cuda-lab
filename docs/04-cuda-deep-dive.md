# 04 — GPU / CUDA deep dive

Twelve experiment modules that expose how the GPU actually behaves. Each one is a
measurement you run and interpret, not a paragraph you read.

Modules M0-M9 need only synthetic tensors, so they run before any real data exists.
M5 and M6 are more meaningful once you have the melt-curve dataset.

**Format:** Question → Run → Measure → Expect → Why it matters.
The "Expect" figures are hypotheses for an RTX 4060 Ti 16 GB (Ada, `sm_89`).
Confirming them is fine; *contradicting* them is more educational — find out why.

Helpers used throughout:

```python
from gpulab.learn import inspect as I
I.cuda_mem(label); I.cuda_time(fn, iters); I.reset_cuda_peak(); I.profile(fn)
```

---

## M0 — Inventory and theoretical ceilings

**Question:** what are this GPU's hard limits, and can I derive them rather than
look them up?

**Run:** query `torch.cuda.get_device_properties(0)` for
`multi_processor_count`, `total_memory`, `major/minor`, and
`max_threads_per_multi_processor`. Then compute by hand:

- **FP32 peak** = CUDA cores x 2 (FMA counts as 2 flops) x boost clock.
  With 4352 cores at ~2.5 GHz: **~22 TFLOPS**.
- **Memory bandwidth** = bus width x memory clock x 2 (DDR).
  128-bit at 18 Gbps: **~288 GB/s**.
- **Arithmetic intensity at balance** = FLOPS ÷ bandwidth = 22e12 / 288e9 ≈
  **~76 FLOP per byte**. Any operation doing less work per byte than this is
  **memory-bound** on this card.

**Measure:** record SM count (expect 34), VRAM (~16 GB usable), `sm_89`, L2 cache
(Ada is generously cached — AD106 has ~32 MB).

**Why it matters:** every later module is a comparison against these ceilings.
"Slow" is meaningless without knowing the maximum. The 76 FLOP/byte balance point
is the single most useful number here — it predicts, before you run anything,
that elementwise work on 61-point curves will be memory- and launch-bound.

---

## M1 — Host-to-device transfer, pageable vs pinned

**Question:** how fast can data cross PCIe, and does pinned memory matter?

**Run:** allocate a large CPU tensor (e.g. 1 GB of float32). Time `.to("cuda",
non_blocking=True)` from (a) normal pageable memory, (b) `.pin_memory()`.
Compute GB/s = bytes ÷ seconds. Time the reverse direction too.

**Measure:** achieved GB/s each way, pageable vs pinned.

**Expect:** the 4060 Ti is **PCIe 4.0 x8** (not x16), so the ceiling is roughly
**~16 GB/s** per direction. Pageable transfers typically reach a fraction of that;
pinned memory should be noticeably faster and is required for genuinely async
copies. `non_blocking=True` only actually overlaps when the source is pinned.

**Why it matters:** this is the wall every "load data from CPU each step" pipeline
hits. It is also why this repo keeps the entire dataset resident in VRAM — 244 MB
crosses PCIe once instead of every epoch.

---

## M2 — Kernel launch overhead and CUDA Graphs

**Question:** what does it cost just to *ask* the GPU to do something?

**Run:** time a loop of a trivially small op (e.g. `x.add_(1)` on a 1-element
tensor) with `I.cuda_time(fn, iters=1000)`. Then time a chain of 100 such ops.
Divide to get per-launch cost. Then capture the same sequence with
`torch.cuda.CUDAGraph` (or `torch.cuda.graphs.make_graphed_callables`) and re-time.

**Measure:** microseconds per kernel launch; speedup from graph capture.

**Expect:** roughly **3-10 microseconds per launch**. A kernel that does 5 us of
work behind a 5 us launch means the GPU is idle half the time. CUDA Graphs replay a
pre-recorded sequence with one submission and can cut this dramatically.

**Why it matters:** this *is* the melt-curve regime. Tiny model + tiny curves =
thousands of cheap kernels, each dominated by launch cost. `I.cuda_time` prints
`LAUNCH-bound` when CPU time exceeds GPU time — M2 explains why.

---

## M3 — The memory bandwidth ceiling

**Question:** what fraction of 288 GB/s can a simple op actually reach?

**Run:** on a large tensor (≥256 MB), time these and convert to GB/s using
**bytes moved**, not element count:

| Op | Bytes moved |
|---|---|
| `y = x.clone()` | 2N (read + write) |
| `y = x + 1` | 2N |
| `z = x + y` | 3N |
| `x.sum()` | 1N |

**Measure:** achieved GB/s for each; percentage of the 288 GB/s ceiling.

**Expect:** well-written elementwise ops should reach **70-90%** of peak. If you see
far less, the tensor may be too small (launch overhead dominates) or non-contiguous
(uncoalesced access). Try a non-contiguous input (`x.t()`) and watch it collapse —
that is memory coalescing made visible.

**Why it matters:** most deep-learning ops that are not matmul or convolution are
memory-bound. Optimizing them means moving fewer bytes (fusion, lower precision),
not doing fewer flops.

---

## M4 — The compute ceiling and Tensor Cores

**Question:** how much faster are Tensor Cores, and when do they engage?

**Run:** square matmuls at increasing size (1024, 2048, 4096, 8192). For each,
compute TFLOPS = `2 * N^3 / seconds`. Run in:

1. FP32 with `torch.backends.cuda.matmul.allow_tf32 = False`
2. FP32 with `allow_tf32 = True` (TF32 Tensor Core path)
3. `bfloat16`
4. `float16`

**Measure:** TFLOPS for each precision at each size.

**Expect:** FP32 approaching ~22 TFLOPS; TF32, bf16 and fp16 substantially higher
because they use Tensor Cores. Small matrices will fall far short of peak — there
isn't enough work to fill 34 SMs.

**Why it matters:** this is where "mixed precision is faster" stops being folklore
and becomes a number you measured. Note that Tensor Cores need the *shapes* to
cooperate — dimensions that are multiples of 8/16 hit the fast paths.

**On the T4 (if you ever run there):** Turing has fp16 Tensor Cores but **no bf16
and no TF32**. Case 2 and 3 will not accelerate. This is exactly why portable code
exposes a precision switch.

---

## M5 — Roofline: place your actual workload

**Question:** is the melt-curve model limited by compute, bandwidth, or launches?

**Run:** for one forward pass of `CNN1D` at a given batch size, estimate FLOPs
(for a Conv1d: `2 * batch * out_channels * in_channels * kernel_size * out_length`)
and bytes moved (activations in + out + weights). Compute arithmetic intensity
(FLOP/byte) and place it against the ~76 FLOP/byte balance point from M0. Then
measure actual time with `I.cuda_time` and compute achieved TFLOPS and GB/s.

**Measure:** arithmetic intensity; achieved vs peak on both axes.

**Expect:** far below both ceilings — the model is neither compute- nor bandwidth-
bound at small batch, it is **launch-bound**. Increasing batch size moves you right
along the roofline toward the bandwidth roof.

**Why it matters:** this is the module that makes every other one pay off. You can
now say precisely *which* resource limits your job, and therefore which
optimization could possibly help.

---

## M6 — Occupancy and the batch-size sweep

**Question:** how much parallel work does it take to fill 34 SMs?

**Run:** sweep batch size 32 → 65536 (powers of 2) through one forward+backward.
For each: time per sample, total throughput (samples/s), and peak memory. In a
second terminal, sample utilization:

```bash
nvidia-smi dmon -s um
```

**Measure:** throughput vs batch size; GPU utilization %; memory vs batch size.

**Expect:** throughput rises steeply, then flattens — the knee is where you finally
have enough parallelism to saturate the SMs. Below the knee you are paying full
launch cost for a partly idle GPU. With 61-point curves you may need very large
batches to find the knee, and may never reach 100% utilization.

**Why it matters:** "the GPU is only at 20%" is a symptom, not a diagnosis. This
experiment tells you whether the cause is insufficient parallelism (fixable with
batch size) or insufficient work per kernel (fixable only with a bigger model or
fused kernels).

---

## M7 — The caching allocator, fragmentation, and memory history

**Question:** why does `nvidia-smi` show more memory used than my tensors need?

**Run:**
```python
I.reset_cuda_peak(); I.cuda_mem("start")
# allocate a few large tensors, delete some, allocate different sizes
I.cuda_mem("after churn")
torch.cuda.empty_cache(); I.cuda_mem("after empty_cache")
```
Then record a full history:
```python
torch.cuda.memory._record_memory_history()
# ... run a training step ...
torch.cuda.memory._dump_snapshot("runs/mem.pickle")
```
View the snapshot at <https://pytorch.org/memory_viz>.

**Measure:** `allocated` (live tensors) vs `reserved` (allocator pool) vs what
`nvidia-smi` reports; peak allocated during a training step.

**Expect:** `reserved` > `allocated` — PyTorch caches freed blocks instead of
returning them to the driver, because `cudaMalloc` is slow. `empty_cache()` returns
them, and usually makes your program *slower*. Allocating many different sizes
causes fragmentation: plenty of free memory, but no contiguous block.

**Why it matters:** almost every "CUDA out of memory" with free memory showing is a
fragmentation or peak-vs-steady-state story. The snapshot viewer turns that from
guesswork into a picture.

---

## M8 — Streams, async execution, and overlap

**Question:** can the GPU compute and transfer at the same time?

**Run:** with **pinned** host memory, issue a large H2D copy and an unrelated
compute kernel. First on the default stream (serialized), then with the copy on a
separate `torch.cuda.Stream`. Time both arrangements. Use `torch.cuda.Event` to
timestamp, and `torch.cuda.current_stream().wait_stream(...)` to synchronize
correctly.

**Measure:** total wall time serialized vs overlapped.

**Expect:** overlapped ≈ max(copy, compute) rather than their sum — the copy engine
and SMs work concurrently. This only happens with pinned memory and separate streams.

**Why it matters:** it is the mechanism behind prefetching data loaders. It also
teaches the discipline of stream synchronization — a classic source of silent race
conditions and wrong results that only appear under load.

---

## M9 — Synchronization traps

**Question:** which innocuous lines secretly stall the CPU?

**Run:** time a training loop, then add `loss.item()` inside the inner loop and
re-time. Repeat with `print(loss)`, `.cpu()`, and `torch.cuda.synchronize()`.
Confirm with a profiler timeline (see [05-profiling.md](05-profiling.md)) that the
CPU is blocked waiting.

**Measure:** wall time per epoch with and without each sync point.

**Expect:** GPU ops are queued asynchronously; anything that reads a value back
forces the CPU to wait for the whole queue to drain. In a launch-bound loop this
can be a large regression.

**Why it matters:** this is the most common accidental performance bug in PyTorch
code — logging a loss every step. The fix is to accumulate on-GPU and sync once per
epoch.

---

## M10 — Fusion with `torch.compile`

**Question:** how much do fewer, bigger kernels help?

**Run:** time the eager model, then `torch.compile(model)` (discard the first call —
it pays compilation cost). Use `I.profile(...)` on both and **count the number of
distinct kernels** in the table.

**Measure:** kernel count and time per step, eager vs compiled.

**Expect:** the compiler fuses elementwise chains into single kernels. The win is
largest exactly where you were launch-bound (M2, M5) — fewer launches, more work
each.

**Windows caveat:** Triton support on Windows is unofficial. If `torch.compile`
errors or silently falls back to eager, run this module under **WSL2**. Note in
your log which environment produced the numbers.

---

## M11 — Write a kernel yourself

**Question:** what does the work look like below the framework?

**Run:** port step 1 of the preprocessing to the GPU. Savitzky-Golay smoothing is a
**fixed FIR filter**, which is exactly a 1-D convolution with precomputed
coefficients; `np.gradient` is a finite-difference stencil. Implement it three ways
and compare:

1. `torch.nn.functional.conv1d` with the SG coefficients as a fixed kernel.
2. A **Triton** kernel (`@triton.jit`) — you choose block size and handle masking.
3. Optionally raw CUDA C via `torch.utils.cpp_extension.load_inline`.

Validate all three against the scipy implementation in `gpulab/data/preprocess.py`
with `np.allclose`. Then benchmark on the full dataset.

**Measure:** correctness (max absolute difference vs scipy), then throughput
(curves/second) for each implementation.

**Expect:** the fused custom kernel should beat a chain of library calls on this
workload because it avoids intermediate round-trips to global memory. You will
directly confront thread/block indexing, bounds masking, and coalesced access.

**Why it matters:** this is the module where "CUDA" stops being an abstraction.
Grid/block/thread decomposition, shared memory, and coalescing are things you
reason about only once you have written a kernel that gets them wrong.

---

## M12 — Precision, numerics, and determinism

**Question:** what do I actually give up in low precision?

**Run:** take one trained model. Evaluate in fp32, bf16, and fp16, comparing logits
and accuracy. Separately: sum a large tensor in fp16 vs fp32 and compare against a
float64 reference. Then enable determinism:

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```
and measure the speed cost of reproducibility.

**Measure:** max logit deviation per dtype; accumulation error; determinism overhead.

**Expect:** bf16 has the same exponent range as fp32 with fewer mantissa bits — it
rarely overflows but is coarse. fp16 has a narrow range and *will* overflow without
loss scaling, which is why fp16 training needs `GradScaler` and bf16 does not.
Large fp16 reductions accumulate visible error. `cudnn.benchmark=True` autotunes
algorithms (fast, nondeterministic); determinism costs throughput.

**Why it matters:** it turns the precision choice into an engineering decision with
known trade-offs, and explains the bf16-vs-fp16 split between your Ada card and
Turing-era hardware.

---

## Suggested order

M0 → M2 → M3 → M4 → M5 are the backbone: limits, launches, bandwidth, compute,
then place your workload. M1 and M8 pair naturally (transfers and overlap). M6, M7,
M9 are the practical debugging trio. M10-M12 are the depth modules.

Record everything in [07-experiment-log.md](07-experiment-log.md). Concepts and
terminology are collected in [06-gpu-concepts.md](06-gpu-concepts.md).
