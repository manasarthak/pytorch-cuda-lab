"""Under-the-hood inspectors for PyTorch tensors, autograd, modules, and CUDA.

Every function here PRINTS what's happening -- the point is for you to run your own
one-liners and watch the internals. Nothing here trains a model or hides logic.

Import: ``from gpulab.learn import inspect as I`` (torch required).
"""

from __future__ import annotations

from typing import Any

import torch


def _mb(n_bytes: int) -> str:
    return f"{n_bytes / 1e6:.3f} MB"


# ----------------------------------------------------------------------------- #
# Tensors: storage + view                                                        #
# ----------------------------------------------------------------------------- #

def describe_tensor(t: torch.Tensor, name: str = "tensor") -> None:
    """Full anatomy of a tensor: what it is, where it lives, how it's laid out."""
    storage = t.untyped_storage()
    is_view = t._base is not None
    print(f"--- {name} " + "-" * 40)
    print(f"  shape          {tuple(t.shape)}")
    print(f"  dtype          {t.dtype}  (element_size={t.element_size()} B)")
    print(f"  device         {t.device}")
    print(f"  numel          {t.numel()}  ->  {_mb(t.numel() * t.element_size())} of live data")
    print(f"  strides        {t.stride()}   (how many elements to step per dim)")
    print(f"  contiguous     {t.is_contiguous()}")
    print(f"  storage_offset {t.storage_offset()}")
    print(f"  is_view        {is_view}  (shares memory with another tensor: {is_view})")
    print(f"  data_ptr       {hex(t.data_ptr())}")
    print(f"  storage bytes  {_mb(storage.nbytes())}  @ {hex(storage.data_ptr())}")
    print(f"  requires_grad  {t.requires_grad}   grad_fn={type(t.grad_fn).__name__ if t.grad_fn else None}")


def show_storage(t: torch.Tensor, name: str = "tensor") -> None:
    """Reveal the difference between the flat memory and the strided view.

    Run this on a tensor and its ``.t()`` / slice to *see* that a transpose or slice
    is just a new set of strides over the SAME storage -- no data copied.
    """
    flat = t.untyped_storage()
    print(f"{name}: view shape {tuple(t.shape)}, strides {t.stride()}, offset {t.storage_offset()}")
    print(f"  underlying storage holds {flat.nbytes() // t.element_size()} elements (1-D, contiguous)")
    print(f"  contiguous copy needed?  {'yes' if not t.is_contiguous() else 'no'}")


def shares_memory(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True if two tensors are views over the same storage (no copy between them)."""
    same = a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    print(f"share storage: {same}  ({hex(a.data_ptr())} vs {hex(b.data_ptr())})")
    return same


# ----------------------------------------------------------------------------- #
# Autograd: the graph                                                            #
# ----------------------------------------------------------------------------- #

def trace_graph(t: torch.Tensor, max_depth: int = 12) -> None:
    """Walk the autograd graph backward from a tensor and print it as a tree.

    Build a small expression with ``requires_grad=True`` leaves, then run this on
    the result to see the chain of grad_fn nodes PyTorch recorded for backprop.
    """
    if t.grad_fn is None:
        print(f"leaf tensor (grad_fn=None, requires_grad={t.requires_grad}) -- nothing to trace")
        return

    def walk(fn: Any, depth: int) -> None:
        if fn is None or depth > max_depth:
            return
        print("    " * depth + f"|_ {type(fn).__name__}")
        for nxt, _ in getattr(fn, "next_functions", ()):
            walk(nxt, depth + 1)

    print(f"backward graph of the result (top = last op, leaves = your inputs):")
    walk(t.grad_fn, 0)


def grad_norms(module: "torch.nn.Module") -> None:
    """Per-parameter gradient norms -- run AFTER ``loss.backward()``.

    Shows which layers actually receive gradient, and flags vanishing/exploding
    grads (norm ~0 or huge). ``None`` means that parameter got no gradient.
    """
    print(f"{'parameter':32s} {'param_norm':>12s} {'grad_norm':>12s} {'ratio':>8s}")
    for pname, p in module.named_parameters():
        pn = p.detach().norm().item()
        if p.grad is None:
            print(f"{pname:32s} {pn:12.4e} {'None':>12s} {'-':>8s}")
        else:
            gn = p.grad.detach().norm().item()
            ratio = gn / pn if pn > 0 else float("nan")
            print(f"{pname:32s} {pn:12.4e} {gn:12.4e} {ratio:8.3f}")


# ----------------------------------------------------------------------------- #
# nn.Module: parameters                                                          #
# ----------------------------------------------------------------------------- #

def param_table(module: "torch.nn.Module") -> None:
    """List every parameter: shape, count, dtype, device, trainability."""
    total = trainable = 0
    print(f"{'parameter':32s} {'shape':>18s} {'numel':>10s} {'dtype':>10s} {'device':>8s} grad")
    for pname, p in module.named_parameters():
        n = p.numel()
        total += n
        trainable += n if p.requires_grad else 0
        print(f"{pname:32s} {str(tuple(p.shape)):>18s} {n:10d} "
              f"{str(p.dtype).replace('torch.',''):>10s} {str(p.device):>8s} {p.requires_grad}")
    print(f"\n  total params     {total:,}")
    print(f"  trainable params {trainable:,}  ({_mb(total * 4)} as fp32)")


# ----------------------------------------------------------------------------- #
# CUDA: memory, timing, precision, profiling                                     #
# ----------------------------------------------------------------------------- #

def cuda_available() -> bool:
    ok = torch.cuda.is_available()
    if not ok:
        print("No CUDA device visible - the CUDA steps need a GPU + a CUDA torch build.")
    return ok


def cuda_mem(label: str = "") -> None:
    """Snapshot GPU memory: allocated (live tensors) vs reserved (allocator pool)."""
    if not torch.cuda.is_available():
        print("no CUDA")
        return
    free, total = torch.cuda.mem_get_info()
    print(f"[{label}] allocated={_mb(torch.cuda.memory_allocated())} "
          f"reserved={_mb(torch.cuda.memory_reserved())} "
          f"peak={_mb(torch.cuda.max_memory_allocated())} "
          f"| device free={free/1e9:.2f}/{total/1e9:.2f} GB")


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_time(fn, iters: int = 50, warmup: int = 10) -> float:
    """Median-ish GPU time per call, measured with CUDA events (honest, synced).

    Also prints CPU wall time. If CPU time >> GPU time, your work is *launch bound*
    (the CPU can't dispatch kernels fast enough to keep the GPU busy) -- the central
    lesson for tiny models on tiny curves.
    """
    import time as _time

    if not torch.cuda.is_available():
        print("no CUDA -- timing skipped")
        return float("nan")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    cpu0 = _time.perf_counter()
    start_evt.record()
    for _ in range(iters):
        fn()
    end_evt.record()
    torch.cuda.synchronize()
    cpu_ms = (_time.perf_counter() - cpu0) * 1e3 / iters
    gpu_ms = start_evt.elapsed_time(end_evt) / iters

    bound = "LAUNCH-bound (CPU can't feed the GPU)" if cpu_ms > gpu_ms * 1.5 else "compute/mem-bound"
    print(f"  gpu {gpu_ms:.3f} ms/iter | cpu {cpu_ms:.3f} ms/iter -> {bound}")
    return gpu_ms


def autocast_probe(device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
    """Show autocast's op-by-op casting policy.

    Reveals that matmul/conv run in low precision under autocast while reductions
    and softmax stay fp32 for stability -- you don't cast anything by hand.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    a = torch.randn(64, 64, device=device)
    b = torch.randn(64, 64, device=device)
    with torch.autocast(device_type=("cuda" if device == "cuda" else "cpu"), dtype=dtype):
        mm = a @ b
        sm = torch.softmax(mm, dim=-1)
        red = mm.sum()
    print(f"autocast requested: {dtype} on {device}")
    print(f"  input a/b       {a.dtype}   <- your stored data is left untouched")
    print(f"  matmul (a@b)    {mm.dtype}   <- autocast casts heavy linear-algebra ops")
    print(f"  softmax         {sm.dtype}")
    print(f"  sum reduction   {red.dtype}")
    print("note: on CUDA, autocast keeps softmax/reductions in fp32 for stability;")
    print("      the exact op policy is device-dependent (CPU coverage differs), so")
    print("      trust the dtypes printed above, not a fixed rule.")


def profile(fn, row_limit: int = 15, trace_path: str | None = None) -> None:
    """Profile a callable and print the hottest kernels (by CUDA time).

    Wrap one training step in a lambda. Optionally dump a chrome trace you can open
    in chrome://tracing or Perfetto to see the timeline of kernels.
    """
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as _profile

    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)

    with _profile(activities=acts, record_shapes=True) as prof:
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=row_limit))
    if trace_path:
        prof.export_chrome_trace(trace_path)
        print(f"\nchrome trace written to {trace_path} (open in chrome://tracing or ui.perfetto.dev)")
