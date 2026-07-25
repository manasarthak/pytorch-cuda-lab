# PyTorch + CUDA, from the ground up — a see-it-yourself lab

This is a **hands-on path**, not a tutorial to copy. At each step you:

- **See it** — call an inspection helper from `gpulab.learn` to watch the internals.
- **Write it** — you write the small bit of PyTorch (the guide tells you *what*, not *how*).
- **Look for** — what the under-the-hood output should teach you.

You learn by running one-liners in a REPL/notebook and watching what changes. The
helpers only *reveal* state — they never do the learning for you.

### Setup

```bash
uv sync --extra torch   # installs CUDA torch from the pinned index (Ada = sm_89)
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Then, in a REPL or a notebook under `notebooks/`:

```python
import torch
from gpulab.learn import inspect as I
from gpulab.learn import viz
```

Keep `GUIDE.md` open on one side and the REPL on the other. Work top to bottom.

---

## Step 0 — Where am I running?

**See it**
```python
from gpulab.train.cuda_utils import describe_device
print(describe_device())          # name, VRAM, sm_XX, torch version
I.cuda_available()
```
**Look for:** `sm_89` and `16 GB` = your 4060 Ti (Ada). If you ever run on the org's
T4 you'll see `sm_75` — remember Turing has **no bf16** (matters in Step 10).

---

## Step 1 — A tensor is *storage + a view*

The single most clarifying idea in PyTorch: a tensor is a flat block of memory
(`storage`) plus metadata (`shape`, `stride`, `offset`) describing how to read it.
Reshapes, slices, and transposes usually make a **new view over the same memory**.

**Write it:** make a `3×4` tensor with `torch.arange(12).reshape(3, 4)`. Make its
transpose. Make a slice (e.g. every other column).

**See it**
```python
I.describe_tensor(x, "x")
I.show_storage(x, "x"); I.show_storage(x.t(), "x.t()")
I.shares_memory(x, x.t())          # transpose = same storage, different strides
I.shares_memory(x, x.reshape(-1))
I.shares_memory(x, x + 0)          # arithmetic makes a NEW tensor
```
**Look for:** `x` and `x.t()` share storage (`shares_memory → True`) but have
different `strides`, and `x.t().is_contiguous()` is `False`. That's why some ops
need `.contiguous()` (a real copy). `x + 0` does *not* share storage — it allocated.

**Write it (memory):** call `.contiguous()` on the transpose and re-run
`describe_tensor` — watch `data_ptr` change (a copy happened).

---

## Step 2 — Dtype is memory *and* speed

**Write it:** take a float tensor and make `.half()` (fp16), `.bfloat16()`, `.double()`.

**See it**
```python
for t in (x.float(), x.half(), x.bfloat16(), x.double()):
    I.describe_tensor(t, str(t.dtype))
```
**Look for:** `element_size` and total bytes halve fp32→fp16/bf16. On 120-point
curves this is tiny, but at scale it's why mixed precision is faster *and* leaner.
Note fp16 and bf16 are both 2 bytes but trade range vs precision differently.

---

## Step 3 — Moving to the GPU (and the sync trap)

**Write it:** create a CPU tensor, then move it with `.to("cuda")`. Do a GPU op.
Then read a value back with `.item()`.

**See it**
```python
I.cuda_mem("before")
g = big_cpu_tensor.to("cuda")       # H2D copy happens here
I.cuda_mem("after .to(cuda)")
y = (g * 2).sum()                   # queued on the GPU, async
val = y.item()                      # <-- forces a sync: CPU waits for the GPU
```
**Look for:** memory jumps after `.to("cuda")`. Understand that GPU ops are
**asynchronous** — they're queued; `.item()`/`.cpu()`/printing forces a
synchronization. Sprinkling `.item()` in a hot loop secretly serializes CPU↔GPU.

---

## Step 4 — Autograd: PyTorch records a graph

Set `requires_grad=True` on leaves and every op is recorded into a backward graph.
`.backward()` walks it to fill `.grad`.

**Write it:** `a = torch.tensor(2.0, requires_grad=True)`, `b = torch.tensor(3.0,
requires_grad=True)`, `L = (a * b + a**2)`. Then `L.backward()`.

**See it**
```python
I.trace_graph(L)          # the tree of grad_fn nodes (MulBackward, PowBackward, ...)
print(a.grad, b.grad)     # dL/da, dL/db — check them by hand!
```
**Look for:** the graph mirrors your expression. `a.grad` should equal `b + 2a = 7`.
Now wrap the forward in `with torch.no_grad():` and re-check `L.grad_fn` — it's
`None` (no graph built; this is what you use at inference to save memory).

**Write it (the gotcha):** call `.backward()` twice without zeroing — watch `.grad`
*accumulate*. This is why training loops call `zero_grad()` every step.

---

## Step 5 — `nn.Module` and parameters

**Write it:** subclass `nn.Module` with one `nn.Linear(120, 8)` and a `forward`.
(You already have a real one to study in `gpulab/models/cnn1d.py` — but write a tiny
one yourself first.)

**See it**
```python
I.param_table(model)      # every weight/bias: shape, count, dtype, device, trainable
print(model.state_dict().keys())
```
**Look for:** parameters are registered automatically because you assigned
`nn.Linear` as an attribute. `param_table` shows total/trainable counts — this is
your model's memory footprint. Move `model.to("cuda")` and re-run: `device` flips.

---

## Step 6 — Loss, backward, and *does gradient reach every layer?*

**Write it:** feed a batch of curves through your model, compute
`nn.CrossEntropyLoss()(logits, labels)`, call `.backward()`.

**See it**
```python
I.grad_norms(model)             # per-parameter grad norm (None = got no gradient)
viz.plot_grad_flow(model)       # bar chart of the same
```
**Look for:** every trainable parameter should have a non-None, non-zero grad norm.
A layer stuck at `None` means it's disconnected from the loss; near-zero across
early layers = vanishing gradients (Step 3 of ResNet's motivation).

---

## Step 7 — The optimizer step

**Write it:** make `opt = torch.optim.AdamW(model.parameters(), lr=1e-3)`. Do the
canonical dance once: `opt.zero_grad(set_to_none=True)` → forward → loss →
`.backward()` → `opt.step()`.

**See it**
```python
w = model.head[-1].weight          # or any parameter
before = w.detach().clone()
# ... do one opt.step() ...
print("param moved by:", (w.detach() - before).norm().item())
```
**Look for:** the parameter actually changed after `step()`. Try `set_to_none=True`
vs `False` and read the docs note: `None` grads skip a kernel and save memory.

---

## Step 8 — A real training loop (your milestone)

Now assemble Steps 4–7 into a loop over the melt-curve data. **Write this yourself**
— it's the whole point. Load a dataset (`gpulab.data.dataset.Dataset.load`), split by
chip (`grouped_train_test_split`), then loop: shuffle → minibatch → zero → forward →
loss → backward → step; evaluate each epoch.

There's a reference implementation in `gpulab/train/loop.py` — **don't read it until
you've written your own**, then diff yours against it to see what you missed.

**See it**
```python
viz.plot_history(history)      # loss down, accuracy up
```
**Look for:** compare your final accuracy to the XGBoost baseline
(`scripts/run_baseline.py`). Beating it is the goal of the neural track.

---

## Step 9 — CUDA utilization: why the GPU looks *idle*

Your whole dataset fits in VRAM (1M curves ≈ 480 MB). Put it there once and slice
batches on-device — no DataLoader. Then measure.

**See it**
```python
I.reset_cuda_peak(); I.cuda_mem("dataset on GPU")
step = lambda: model(X_gpu[:batch])          # one forward
I.cuda_time(step, iters=100)                  # gpu vs cpu ms/iter
```
**Look for:** with a tiny model, `cuda_time` reports **LAUNCH-bound** — CPU time
dominates because each kernel is so cheap the CPU can't dispatch them fast enough.
Sweep `batch` from 64 → 4096 and watch GPU util (`nvidia-smi dmon` in another
terminal) barely move. *This is the key CUDA lesson.* To become compute-bound you
must add real work: a wider/deeper net, or the 2-D recurrence-plot representation.

---

## Step 10 — Mixed precision (AMP)

**See it**
```python
I.autocast_probe(device="cuda", dtype=torch.bfloat16)
```
**Look for:** under `autocast`, matmul/conv run in bf16 while softmax/reductions
stay fp32 — you cast *nothing* by hand. **Write it:** wrap your Step 8 forward in
`with torch.autocast(device_type="cuda", dtype=torch.bfloat16):` and compare
`cuda_time` and `cuda_mem` with/without.

**Portability note:** bf16 is the Ada default (no gradient scaler needed). On the
org's **T4 (Turing) there's no bf16** — you'd use `torch.float16` **plus**
`torch.cuda.amp.GradScaler()`. Same loop, different precision knob.

---

## Step 11 — `torch.compile`

**Write it:** `cmodel = torch.compile(model)`. Time the *second* call onward (the
first pays a one-time compile cost).

**See it**
```python
I.cuda_time(lambda: model(X_gpu[:1024]),  iters=100)   # eager
I.cuda_time(lambda: cmodel(X_gpu[:1024]), iters=100)   # compiled
```
**Look for:** compile fuses kernels, which helps most exactly when you were
launch-bound (Step 9) — fewer, bigger kernels. Bigger win on Ada than on T4.

---

## Step 12 — Profiling: read the kernels

**See it**
```python
I.profile(lambda: (model(X_gpu[:1024]).sum().backward()), row_limit=15,
          trace_path="runs/step.json")
```
**Look for:** the table ranks ops by GPU time; the chrome trace
(`chrome://tracing` or Perfetto) shows the timeline with gaps = the GPU waiting on
the CPU. Now you can *point* at what to optimize instead of guessing.

---

## Where this leads

Once Steps 1–12 feel natural, the repo's higher rungs (`PLAN.md`) are just more of
the same: ResNet1D (residuals fix the vanishing grads you saw in Step 6), a GPU
port of the preprocessing (Savitzky-Golay = a fixed `conv1d`), and the 2-D
recurrence-plot CNN that finally makes the 4060 Ti sweat.
