# energy-efficiency-AI-sovereignty

This is the companion code to the paper [insert link]. It contains the scripts ran across three UCL nodes. 

Three kernels are run at two precisions on three accelerators, 20 repeats each,
for 360 measurements in total. Each measurement records a 1 Hz power trace
alongside a timing record, from which energy per sample is computed.

## Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Repository structure](#repository-structure)
- [Method](#method)
- [Limitations](#limitations)
- [Citation](#citation)

## Requirements

One GPU node with `nvidia-smi`, or one IPU node with the Poplar SDK enabled and
`gc-monitor` on the path. `tmux` is needed on both, since the power sampler runs
in its own session.

Python 3.9 with PyTorch. On the GPUs, `pynvml` if available; the sampler falls
back to parsing `nvidia-smi` without it. On the IPU, PopTorch.

The exact versions the published measurements were taken on are recorded in
`shared/controls.py`, which checks the live environment against them at the
start of every run and notes any difference in the result file.

## Setup

The scripts expect to live flat in `~/scripts` on the node, because that is
where `run_one_measurement.sh` looks for them and how they invoke each other.
On the node:

    git clone <this repository>
    cd energy-efficiency-AI-sovereignty
    mkdir -p ~/scripts
    find . \( -name '*.py' -o -name '*.sh' \) -not -path './.git/*' \
         -exec cp {} ~/scripts/ \;
    cd ~/scripts

Copying all of them to every node is harmless; only the scripts matching the
platform are ever invoked.

## Usage

### Running one measurement

    bash run_one_measurement.sh <host> <idle_w> <ipu|gpu> <kernel> <n> <dtype> <run>

| argument | values | notes |
|---|---|---|
| `host` | `mandelbrot` / `cricket` / `locust` | must match the machine's hostname; names the output files |
| `idle_w` | `41.6` / `43.14` / `89.41` | idle baseline for that node, recorded for context |
| `ipu\|gpu` | `ipu` for mandelbrot, `gpu` for cricket and locust | selects the sampler and kernel scripts |
| `kernel` | `stencil` / `triad` / `gemm` | |
| `n` | `384` / `68000000` / `4096` | problem size, identical across nodes |
| `dtype` | `fp32` / `fp16` | one script handles both |
| `run` | `1`, `2`, ... | repeat index, appears in the filename |

One example per node:

    bash run_one_measurement.sh cricket    43.14 gpu stencil 384      fp32 1
    bash run_one_measurement.sh locust     89.41 gpu triad   68000000 fp16 1
    bash run_one_measurement.sh mandelbrot 41.6  ipu gemm    4096     fp32 1

Duration defaults to 300 s, overridable with `SECS`. The script waits for other
users' processes to clear before starting and takes a per-node lock so two
measurements cannot overlap. Each run writes a `_log.json` (timing, sample
count, environment, occupancy before and after) and a matching `_power.csv`.

The published figures repeat this 20 times per kernel, precision and node, with
a 60 s gap between measurements.

### Analysis

    python3 analysis/extract_runs.py <host> <ipu|gpu> > raw_<host>.csv
    python3 analysis/summarise_runs.py <host> <ipu|gpu>

`extract_runs.py` emits one row per measurement, pairing each timing record with
its power trace by filename. Power is averaged only over the marker-gated
region; on cricket, the busier of the two GPUs is used and the idle sibling
ignored.

    samples    = calls
    throughput = samples / total_time_s
    energy     = mean_power_w * total_time_s / samples * 1000    (mJ/sample)

Figures and spreadsheets were produced separately and are not included.

### Batching comparison

The main results are unbatched.
`batching/` tests whether that understates the GPUs, by running each kernel
twice at each precision, once unbatched and once at the largest batch that fits,
so the only difference between the two arms is batch size.

    bash batching/run_batch_comparison.sh <host> <idle_w>
    python3 batching/extract_batch_comparison.py <host> > batch_comparison_<host>.csv

Twelve measurements per node: 3 kernels x 2 precisions x {unbatched, batched},
one run each. It finds the largest workable batch by running the real kernel for
3 s at descending sizes, rather than allocating operands in isolation, because
the scripts hold input and output resident while the kernel allocates its own
result, so a batch that fits in isolation can still exhaust memory in use.

GPU only. The IPU is not batched here: its operands are already resident, and
the per-tile memory limit makes large batches infeasible.

`gpu/gpu_batch_sweep.py` is a standalone utility reporting the largest batch
that fits on a card, for each kernel and precision.

## Repository structure

    run_one_measurement.sh

Runs one measurement end to end: starts the power sampler, runs the kernel,
stops the sampler, and checks the trace came out the right length.

    shared/controls.py

Seed, dtype, TF32 and cuDNN settings, applied before any tensor is allocated.
Also provides `RunMarker`, the context manager that tells the sampler when the
timed region starts and stops, and records the live environment into each
result file.

    gpu/gpu_power_sampler.py
    ipu/ipu_power_sampler.py

Poll the power sensor once a second and write a CSV row per reading, only while
the marker file exists. The GPU version reads every GPU on the node and labels
each row with its index.

    gpu/gpu_stencil.py      ipu/ipu_stencil.py
    gpu/gpu_triad.py        ipu/ipu_triad.py
    gpu/gpu_gemm.py         ipu/ipu_gemm.py

The three measured kernels, one script per platform. Each takes the precision
and problem size as arguments and writes a JSON record of the run.

    gpu/gpu_gemm_batched.py

GEMM with a batch dimension, for the batching comparison. Uses `torch.bmm`
above B=1 and plain matmul at B=1, so the unbatched arm is the same computation
as `gpu_gemm.py`.

    gpu/gpu_batch_sweep.py

Reports the largest batch that fits on a card, per kernel and precision.

    batching/run_batch_comparison.sh
    batching/extract_batch_comparison.py

Run and then summarise the batched-versus-unbatched comparison.

    analysis/extract_runs.py
    analysis/summarise_runs.py

Pair each timing record with its power trace and emit one CSV row per run, then
reduce those to a mean and standard deviation per kernel, precision and node.

    verification/correctness_and_scaling.py
    verification/device_counter_check.py

Checks on the IPU sample counts, described under [Limitations](#limitations).

Kernel scripts carry the platform in the filename because the scripts are
copied into a single directory on each node, where the folder structure is not
preserved.

## Method

### Nodes

| host | accelerator | measured | idle |
|---|---|---|---|
| mandelbrot | Graphcore GC200 IPU | one chip, `gc-monitor` chip power | 41.6 W |
| cricket | NVIDIA A100 80GB PCIe | one GPU, `nvidia-smi power.draw` | 43.14 W |
| locust | NVIDIA GH200 | one GPU, `nvidia-smi power.draw` | 89.41 W |

### Kernels

Chosen to span arithmetic intensity, so that memory-bound and compute-bound
behaviour can be separated.

| kernel | operation | size | FLOP/byte |
|---|---|---|---|
| triad | `a = b + 2c`, elementwise | N = 68,000,000 | 0.17 |
| stencil | 3D 27-point convolution | N = 384 | 1.6 |
| GEMM | square matrix multiply | N = 4096 | 2917 |

One sample is one complete application of the kernel: one grid update, one pass
over the vectors, one matmul. Each kernel has a single script per platform, with
precision selected at runtime.

### Measurement protocol

Every run excludes compilation and 10 warm-up calls from **both** the timing and
the power window, then measures for 300 s.

Exclusion is enforced by a marker file. The kernel creates it on entering the
timed region and deletes it on leaving; the power sampler, running as a separate
process, records only while it exists. 

Operands are resident on the device throughout. On the IPU this requires
`register_buffer`: PopTorch streams anything passed as a `forward()` argument
from the host on every call, which would otherwise make the measurement a test
of PCIe bandwidth rather than of the accelerator.

Idle power is included in the reported figure and never subtracted.

FP16 runs accumulate in FP32 on both platforms
(`allow_fp16_reduced_precision_reduction=False` on the GPUs,
`setPartialsType(float32)` on the IPU), so reduced precision means the same
thing in both cases. TF32 is disabled, so does FP32.

## Limitations

GPU sample counts are confirmed by a device-side counter. The IPU's counts are
host-side call counts: PopTorch streams registered buffers in as inputs on each
call, so an on-device counter resets every invocation, and PopTorch 3.2.0
provides no interface for reading a buffer back.

They are instead validated two other ways. Output correctness: the IPU's FP32
stencil result matches a float64 CPU reference to 1.24e-06 relative, closer than
CPU FP32 itself. Linear scaling: 45 s and 90 s runs give a sample ratio of
2.0026, with 0.143% drift in rate. `verification/` contains both checks.

`gc-monitor` only reports power for IPUs with an attached process, so it cannot
measure an idle baseline. The IPU idle figure was measured separately.

## Citation

If you use this code, please cite the paper:

    [citation to add]
