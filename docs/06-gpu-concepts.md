# 06 — GPU and CUDA concepts reference

Terminology and the mental model behind the experiments in
[04-cuda-deep-dive.md](04-cuda-deep-dive.md). Each entry names the module that
makes it observable.

## Execution model

| Term | Meaning |
|---|---|
| **Thread** | one lane of execution; runs the kernel body for one data element |
| **Warp** | 32 threads executing in lockstep (SIMT). The real scheduling unit |
| **Block** | a group of threads (up to 1024) that share **shared memory** and can synchronize with each other |
| **Grid** | all blocks launched by one kernel |
| **SM** (Streaming Multiprocessor) | the physical core cluster that executes blocks. The 4060 Ti has **34** |
| **Kernel** | a function compiled to run on the GPU across the grid |

A block is assigned to exactly one SM; an SM runs many blocks concurrently if
registers and shared memory allow. You size blocks; the hardware schedules warps.

**Warp divergence:** if threads in a warp take different branches, both paths run
serially with inactive lanes masked. Branchy per-thread logic is expensive.

## Latency hiding and occupancy

GPUs do not avoid memory latency, they **hide** it: when a warp stalls on a memory
read, the SM instantly switches to another ready warp. This only works if enough
warps are resident.

- **Occupancy** = resident warps ÷ maximum possible warps per SM.
- Limited by registers per thread, shared memory per block, and block size.
- Higher occupancy is not automatically better, but *very low* occupancy means
  nothing is left to switch to, and latency becomes exposed.

*Observed in M6 (batch sweep), measured properly with Nsight Compute.*

## Memory hierarchy

From fastest/smallest to slowest/largest:

| Level | Scope | Notes |
|---|---|---|
| **Registers** | per thread | fastest; spilling to "local" memory is a silent performance cliff |
| **Shared memory** | per block | programmer-managed cache; the main tool for data reuse in a kernel |
| **L1 / texture** | per SM | |
| **L2** | whole device | Ada is generously provisioned (~32 MB on AD106), which matters a lot for small working sets |
| **Global (VRAM)** | whole device | 16 GB at ~288 GB/s |
| **Host RAM** | CPU | reached over PCIe 4.0 x8, ~16 GB/s |

**Memory coalescing:** when the 32 threads of a warp read consecutive addresses,
the hardware merges them into a few wide transactions. Strided or scattered access
issues many transactions and wastes bandwidth. This is why a non-contiguous tensor
(`x.t()`) collapses in M3, and why `.contiguous()` sometimes makes code faster
despite copying.

## What limits a kernel

Three regimes — identifying which one you are in is the point of M5:

1. **Compute-bound** — saturating the arithmetic units. Fix: lower precision,
   Tensor Cores, better algorithms.
2. **Memory-bound** — saturating bandwidth. Fix: move fewer bytes — fusion, lower
   precision, better access patterns, data reuse in shared memory.
3. **Launch/latency-bound** — the GPU is idle waiting for work. Fix: bigger batches,
   kernel fusion, CUDA Graphs.

**Arithmetic intensity** = FLOPs ÷ bytes moved. Compare it to the device balance
point (~76 FLOP/byte here, from M0) to predict the regime before measuring.
The **roofline model** plots achievable performance against arithmetic intensity:
a bandwidth slope on the left, a flat compute ceiling on the right.

## Asynchrony

CUDA calls are **queued**, not executed synchronously. The CPU races ahead
launching kernels while the GPU works through the stream.

- **Stream** — an ordered queue. Work in different streams may overlap.
- **Event** — a marker in a stream; used for timing and cross-stream dependencies.
- **Synchronization** — `.item()`, `.cpu()`, `print()`, or an explicit
  `torch.cuda.synchronize()` blocks the CPU until the queue drains (*M9*).
- **Pinned (page-locked) host memory** — cannot be swapped out, so the DMA engine
  can copy it without CPU involvement. Required for truly async transfers (*M1, M8*).
- **CUDA Graphs** — record a sequence of launches once, replay with a single
  submission. Amortizes per-launch overhead (*M2*).

Because of asynchrony, **any CPU-side timer without a synchronize is wrong**.

## Precision

| Format | Bits (E/M) | Range | Notes |
|---|---|---|---|
| FP32 | 8/23 | wide | the baseline |
| TF32 | 8/10 | FP32 range | Ampere+; FP32-transparent Tensor Core path for matmul |
| BF16 | 8/7 | FP32 range | coarse but hard to overflow; **no loss scaling needed** |
| FP16 | 5/10 | narrow | more mantissa than bf16, but overflows; needs `GradScaler` |
| FP8 | 4/3 or 5/2 | very narrow | Ada supports it; specialist use |

**Tensor Cores** are dedicated matrix-multiply-accumulate units. They need
cooperative shapes (multiples of 8/16) and a supported dtype. Ada supports
TF32/BF16/FP16/FP8; **Turing (T4) supports FP16 and INT8 but not BF16 or TF32** —
the reason portable training code exposes a precision switch.

**Mixed precision** (`torch.autocast`) keeps a master copy of weights in fp32 while
running heavy ops in low precision; the exact op policy is device-dependent
(`I.autocast_probe` prints what actually happened). *M4, M12.*

## PyTorch specifics

- **Caching allocator** — PyTorch calls `cudaMalloc` rarely and reuses freed blocks,
  because allocation synchronizes and is slow. Hence `reserved` > `allocated`;
  `empty_cache()` returns memory to the driver and usually costs performance (*M7*).
- **Fragmentation** — many differently-sized allocations leave free memory that is
  not contiguous; you can OOM with plenty free.
- **`torch.compile`** — traces the graph and generates fused kernels (via Triton),
  reducing launches and intermediate memory traffic (*M10*).
- **cuDNN benchmark** — `torch.backends.cudnn.benchmark = True` autotunes conv
  algorithms per input shape: faster in steady state, nondeterministic, and it
  re-tunes whenever shapes change.

## Questions worth being able to answer

Use these to check your own understanding after running the modules:

- Why does doubling batch size sometimes not change time per epoch at all?
- Why is `empty_cache()` usually a mistake?
- Why does bf16 avoid the loss-scaling that fp16 requires?
- Why does a transposed tensor make an elementwise kernel slower?
- Why is a CPU-side `time.time()` around a GPU op meaningless?
- Given FLOPs and bytes for an op, which resource limits it on this card?
- Why can `sm%` read high while the GPU is doing little useful work?
