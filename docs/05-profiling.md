# 05 — Profiling workflow

Three tools, three questions. Reach for them in this order.

| Tool | Answers | Granularity |
|---|---|---|
| `nvidia-smi dmon` | "is the GPU busy at all?" | seconds |
| `torch.profiler` | "which op costs the most?" | per-op / per-kernel |
| Nsight Systems / Compute | "why is *this* kernel slow?" | timeline / hardware counters |

## 1. Is it even busy? — `nvidia-smi`

```bash
nvidia-smi dmon -s um
```

Columns: `sm%` (SM utilization), `mem%` (memory controller activity), `fb`
(framebuffer MB). Run this in a second terminal during training.

Read it as: **`sm%` low + job slow** = starved GPU (launch-bound or input-bound) —
go to M2/M5/M6. **`sm%` high + job slow** = genuinely compute-bound — go to M4.

Caveat: `sm%` reports the fraction of time *at least one* kernel was resident, not
how well the SMs were filled. It can read high while occupancy is poor. Use Nsight
Compute for the real story.

## 2. Which op costs the most? — `torch.profiler`

```python
from gpulab.learn import inspect as I
I.profile(lambda: (model(X[:1024]).sum().backward()), row_limit=20,
          trace_path="runs/step.json")
```

The table ranks operators by total CUDA time. What to look for:

- One op dominating → optimize or fuse that one.
- Many tiny kernels, none dominant → launch-bound; fusion (M10) or bigger batches.
- Large `aten::copy_` or `to` entries → unnecessary host/device traffic (M1).
- `cudaStreamSynchronize` high on the CPU side → a sync trap (M9).

Open `runs/step.json` in `chrome://tracing` or <https://ui.perfetto.dev>. **Gaps on
the GPU row are the whole point** — they show the GPU idle, waiting on the CPU.

For a training loop, use the scheduled profiler so you skip warmup steps:

```python
from torch.profiler import profile, schedule, ProfilerActivity
sched = schedule(wait=1, warmup=1, active=3, repeat=1)
```

Profile *steady state*, never the first iteration — it includes allocation,
autotuning, and compilation.

## 3. Why is this kernel slow? — Nsight

**Nsight Systems** (timeline, whole application):

```bash
nsys profile -o runs/train --trace=cuda,nvtx uv run python your_script.py
```

Annotate regions so the timeline is readable:

```python
with torch.cuda.nvtx.range("forward"):
    ...
```

**Nsight Compute** (single-kernel hardware counters):

```bash
ncu --set full -o runs/kernel uv run python your_script.py
```

This is where you get achieved occupancy, memory throughput vs peak, warp stall
reasons, and cache hit rates — the numbers that explain *why* M3/M4 fell short of
the ceiling. It serializes kernels and is slow; target one kernel with `-k`.

Nsight Compute may require elevated permissions for counter access on Windows.

## Method

1. **Measure before changing anything.** Record a baseline number.
2. **Change one thing.**
3. **Re-measure with the same protocol** — same batch, same warmup, same iteration count.
4. **Log it** in [07-experiment-log.md](07-experiment-log.md), including what did *not* help.

Timing rules that keep results honest:

- Always warm up (first iterations include allocation and autotuning).
- Always `torch.cuda.synchronize()` before stopping a CPU timer, or use CUDA events.
  `I.cuda_time` does both for you.
- Report median or mean over many iterations, never a single run.
- Change one variable at a time.
