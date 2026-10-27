# Profiling autofm on Leonardo

How to profile `autofm.train_mae` on Leonardo's A100 GPUs with NVIDIA Nsight,
and the loop for turning a profile into a speedup.

Both tools run inside a GPU job (an interactive `srun` or a batch step) - never
on the login node, which has no GPU. See [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md)
for getting onto a compute node.

The training code already emits NVTX ranges (`mae/epoch_*`, `mae/forward`,
`mae/backward`, `mae/optim`) on CUDA, so the timeline is annotated with the
parts that matter.

> **Prereqs these commands assume.** `config_leonardo.yaml` is the full-scale
> config (copy `config_autofm.yaml`, point `dataset_root` at `$WORK`, widen
> `max_days` / `max_gnbs`). `--epochs` overrides `mae_epochs` for short runs -
> keep profiling runs to 1-2 epochs so the capture stays small. Load the Nsight
> tools first (`module avail nsight` / `module load <nsight-systems>`), or use
> the versions bundled with the CUDA module.

---

## 1. Nsight Systems (nsys) - the timeline (start here)

Nsight Systems shows the whole-run timeline: where time goes across CPU, GPU,
data loading, and the NVTX ranges. Use it first to find *which* phase is slow
before zooming into kernels.

```bash
nsys profile -t cuda,nvtx,osrt,cudnn,cublas \
  -o artifacts/nsys_mae --force-overwrite true \
  python -m autofm.train_mae --config config_leonardo.yaml --epochs 2
```

What the flags do:

- `-t cuda,nvtx,osrt,cudnn,cublas` - trace CUDA runtime/kernels, our NVTX
  ranges, OS runtime (threads, I/O, syscalls), and the cuDNN/cuBLAS libraries.
- `-o artifacts/nsys_mae` - write `artifacts/nsys_mae.nsys-rep`.
- `--force-overwrite true` - replace a previous capture of the same name.
- `--epochs 2` - two epochs is enough to see steady-state after warmup.

Read the result:

```bash
# quick text summaries without the GUI
nsys stats artifacts/nsys_mae.nsys-rep            # kernel + API time breakdown
nsys stats --report nvtx_sum artifacts/nsys_mae.nsys-rep   # time per NVTX range
```

Or open `artifacts/nsys_mae.nsys-rep` in the Nsight Systems GUI (copy it to your
laptop). Questions to answer from the timeline:

- Is the GPU **busy or idle between steps**? Gaps before `mae/forward` usually
  mean the input pipeline (DataLoader / `host->device` copy) is the bottleneck,
  not the math.
- How much time is `mae/forward` vs `mae/backward` vs `mae/optim`?
- Are `host->device` memcpys overlapping compute, or serializing it?

---

## 2. Nsight Compute (ncu) - kernel deep-dive (only once a kernel is the suspect)

Nsight Compute profiles individual GPU kernels in depth (occupancy, memory vs
compute bound, warp stalls). It is much slower and serializes execution, so run
it for **one epoch** and cap the number of kernel launches captured.

```bash
ncu --set full --launch-count 20 \
  -o artifacts/ncu_mae --force-overwrite \
  python -m autofm.train_mae --config config_leonardo.yaml --epochs 1
```

What the flags do:

- `--set full` - collect the full metric set (occupancy, memory, compute,
  stalls).
- `--launch-count 20` - profile only the first 20 kernel launches; enough to
  characterize the hot kernels without profiling the entire run (which would be
  very slow).
- `-o artifacts/ncu_mae` - write `artifacts/ncu_mae.ncu-rep`.

Read the result in the Nsight Compute GUI, or:

```bash
ncu --import artifacts/ncu_mae.ncu-rep --page details | less
```

Look for: is the hot kernel **memory-bound or compute-bound**, what's the
achieved **occupancy**, and the top **warp-stall reason**. That tells you the
*kind* of fix (bigger batch, fewer tiny kernels, better memory access, AMP).

---

## 3. The loop: run → find → fix one thing → run again → record

Profiling is iterative. Change **one** thing per iteration so you can attribute
the effect. Keep a running log (a table in this file, or a per-iteration JSON
from `autofm.bench`).

1. **Run** - capture a baseline with `nsys` (and `autofm.bench run` for a
   single throughput number: median ms/step and samples/sec).
2. **Find the slow part** - from the `nsys` timeline / `nvtx_sum`, identify the
   single biggest cost (idle gaps → input pipeline; a dominant kernel → go to
   `ncu`).
3. **Fix one thing** - apply exactly one change aimed at that cost. Typical
   first moves:
   - GPU idle between steps → raise `num_workers`, enable `pin_memory` (on by
     default on CUDA), increase `batch_size`.
   - Compute-bound matmuls → confirm AMP is on (`amp: true`) and TF32 is enabled
     (it is, set in `train_mae`), raise `batch_size` to improve occupancy.
   - Many tiny kernels / low occupancy → larger batch, larger model width.
4. **Run again** - re-capture with the **same** command and compare.
5. **Record before/after** - log the throughput delta and keep or revert the
   change.

### Before/after log

| Iter | Change | median ms/step | samples/sec | GPU util | Keep? |
|------|--------|---------------:|------------:|---------:|:-----:|
| 0 | baseline | _fill in_ | _fill in_ | _fill in_ | - |
| 1 | e.g. num_workers 0 → 8 | | | | |
| 2 | e.g. batch 512 → 2048 | | | | |
| 3 | e.g. AMP off → on | | | | |

Get the throughput numbers consistently with:

```bash
python -m autofm.bench run --config config_leonardo.yaml --gpus 1 \
  --compare --out artifacts/bench_iterN.json
```

`--compare` reports the AMP off-vs-on speedup directly, which is usually the
first and largest win on A100.

---

## 4. Multi-GPU scaling (after single-GPU is tuned)

Tune one GPU first, then measure how throughput scales across the node and chart
it:

```bash
# 1 GPU
python -m autofm.bench run --gpus 1 --out artifacts/bench_1gpu.json
# 4 GPUs (whole node)
torchrun --nproc_per_node=4 -m autofm.bench run --gpus 4 \
  --out artifacts/bench_4gpu.json
# scaling chart + efficiency %
python -m autofm.bench plot artifacts/bench_1gpu.json artifacts/bench_4gpu.json \
  --out artifacts/scaling.png
```

Scaling efficiency `= throughput_N / (N × throughput_1) × 100`; 100% is perfect
linear scaling. If it drops off sharply, go back to `nsys` on a multi-GPU run
and look for communication (NCCL all-reduce) stalls in the timeline.
