# Results

Measurements from 360 runs: 3 kernels x 2 precisions x 3 nodes x 20 repeats.
Each run is 300 s, with compilation and 10 warm-up calls excluded from both the
timing and the power trace.

## raw/

One pair of files per measurement, as the instruments wrote them.

`<kernel><precision>_<host>_run<N>_log.json` holds the timing record: sample
count, duration, start and end timestamps, the environment the run executed in,
and device occupancy before and after.

`<kernel>_<host>_run<N>_power.csv` holds the power trace, one row per second,
covering only the measured region. Columns differ by platform: the GPU files
carry `power_w` per `gpu_index`, the IPU files carry `chip_w` and `chassis_w`.
Both start with `unix_time` and `elapsed_s`.

Every file follows the same pattern, so a timing record and its power trace
differ only in the suffix.

On a two-GPU node both GPUs are logged and the analysis uses the busier one,
which also shows the idle sibling was not doing anything.

## per_run/

One row per measurement, pairing each timing record with its power trace.
Produced by `analysis/extract_runs.py`. Mean power here is the average over the
~300 samples inside that run's measured region.

## summary/

`results.csv` holds one row per kernel, precision and node: mean and standard
deviation over the 20 runs in that cell.

`allocation.csv` and `tradeoff.csv` are derived from it by
`analysis/derive_comparison_metrics.py`. Allocation inefficiency is each node's
energy per sample divided by the lowest for that workload, so 1.0 marks the most
efficient node and the baseline differs between workloads. The trade-off gives
the energy saved and the throughput given up by running on the most efficient
node rather than the fastest, both relative to the fastest.

## How the numbers are derived

    throughput   = samples / duration
    energy       = mean power x duration / samples    [mJ per sample]

A sample is one complete application of the kernel: one grid update, one pass
over the vectors, one matrix multiplication.

Each run produces a single energy-per-sample value from that run's own mean
power, duration and sample count. The reported figure is the mean of the 20
such values, and the standard deviation is their spread. It is therefore
run-to-run variation, not the variation between power samples within a run.

Idle power is included throughout and never subtracted.
