"""Small helpers to make CUDA behaviour visible while learning.

The lessons this project is built around:

* On 61-point curves the whole dataset fits in VRAM (1M curves ~= 244 MB), so you
  can hold ``X`` as one resident CUDA tensor and index batches on-device — zero
  DataLoader/PCIe overhead. With a tiny model you'll then observe that training is
  *kernel-launch bound*, not compute bound: GPU utilization stays low no matter the
  batch size. That's the point — it's the clearest way to feel the difference.
* To actually saturate the 4060 Ti you need heavier compute: a wider/deeper net,
  bigger batches with AMP, or the 2D recurrence-plot representation. These helpers
  let you measure each change.
"""

from __future__ import annotations

import time
from contextlib import contextmanager


def device() -> "object":
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device() -> str:
    import torch

    if not torch.cuda.is_available():
        return "CPU (no CUDA device visible)"
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    return (
        f"{props.name} | {props.total_memory / 1e9:.1f} GB | "
        f"sm_{props.major}{props.minor} | torch {torch.__version__}"
    )


def memory_summary() -> str:
    import torch

    if not torch.cuda.is_available():
        return "no CUDA"
    alloc = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    peak = torch.cuda.max_memory_allocated() / 1e6
    return f"alloc={alloc:.0f}MB reserved={reserved:.0f}MB peak={peak:.0f}MB"


@contextmanager
def timed(label: str):
    """Wall-clock a block, synchronizing CUDA so the timing is honest."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - start
        print(f"[timed] {label}: {dt * 1e3:.1f} ms")
