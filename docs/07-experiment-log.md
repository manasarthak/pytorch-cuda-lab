# 07 — Experiment log

Append one entry per run. A number without its configuration is not a result.

## Environment (record once, update when it changes)

```
GPU              : RTX 4060 Ti 16 GB (Ada, sm_89, 34 SMs)
CPU              : Ryzen 5 5600X (6C/12T)
RAM              :
OS               : Windows 11
Driver           :
torch / CUDA     :
```

---

## Template

```
### <id> - <short title>
Date        :
Module/doc  :        # e.g. M3 (bandwidth), 03-neural-training
Question    :        # what am I trying to find out
Config      :        # model, batch, dtype, seed, iters, warmup
Command     :

Result      :        # the number(s), with units
Expected    :        # what the doc predicted
Match?      : yes / no
Explanation :        # if no - why. This is the valuable part.
Next        :        # what this suggests trying
```

---

## Entries

### E001 - Device inventory (M0)
Date        :
Config      : n/a
Result      : SMs = ___ , VRAM = ___ GB, cc = sm___
Derived     : FP32 peak ~___ TFLOPS, bandwidth ~___ GB/s, balance ~___ FLOP/byte
Notes       :

### E002 - Host-to-device bandwidth (M1)
Config      : 1 GB float32, pageable vs pinned
Result      : pageable ___ GB/s , pinned ___ GB/s , D2H ___ GB/s
Expected    : ceiling ~16 GB/s (PCIe 4.0 x8)
Explanation :

### E003 - Kernel launch overhead (M2)
Config      : 1-element add, iters=1000
Result      : ___ us/launch ; with CUDA Graph ___ us
Expected    : 3-10 us
Explanation :

<!-- copy the template for each new experiment -->

---

## Results summary

Keep the headline numbers here so comparisons stay easy.

| Model | Split seed | Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|
| XGBoost + features | 0 | | | baseline |
| MiniROCKET + ridge | 0 | | | |
| CNN1D | 0 | | | |
| ResNet1D | 0 | | | |

| Benchmark | Config | Result | % of ceiling |
|---|---|---|---|
| Elementwise bandwidth | 256 MB, `y=x+1` | GB/s | |
| Matmul FP32 | 4096^3 | TFLOPS | |
| Matmul BF16 | 4096^3 | TFLOPS | |
| Training step | batch 512, bf16 | ms/step | |

---

## Things that did not work

Negative results are worth more than they feel like at the time — they stop you
repeating the attempt in three weeks.

| What I tried | Expected | What happened | Why |
|---|---|---|---|
| | | | |
