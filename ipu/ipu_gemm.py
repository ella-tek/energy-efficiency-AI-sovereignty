"""
Square matmul on IPU, N = 4096. Compile and the 10 warm-up calls are excluded
from the timing and the power trace.
"""

__version__ = "2026-08-24a"

import argparse
import csv
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import controls as C

import torch
import poptorch


class GemmResident(torch.nn.Module):
    """C = A @ B, all three matrices resident on-die.

    A square matmul rather than nn.Linear, so it stays matrix-matrix at all
    times. One call is one sample.
    """

    def __init__(self, n, half=False):
        super().__init__()
        dt = torch.float16 if half else torch.float32
        self.register_buffer("A", torch.randn(n, n).to(dt))
        self.register_buffer("B", torch.randn(n, n).to(dt))
        self.register_buffer("C", torch.zeros(n, n).to(dt))
        # int32 so PopTorch's half-casting cannot corrupt the counter
        self.register_buffer("count", torch.zeros(1, dtype=torch.int32))

    def forward(self, tick):
        self.C.copy_(torch.matmul(self.A, self.B))
        self.count += 1
        return self.count.float() * tick


def occupancy():
    """Which users are attached to the IPU. Recorded before and after each run."""
    import subprocess
    try:
        out = subprocess.run(["gc-monitor", "--no-card-info"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception as e:
        return {"error": str(e)[:120]}
    procs = [l.strip() for l in out.splitlines()
             if "python" in l.lower() or "PID" in l]
    return {"attached_process_lines": len(procs),
            "no_attached_processes": "No attached processes" in out}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=int, required=True, help="run index, 1..10")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--dtype", choices=["fp32","fp16"], default="fp32")
    p.add_argument("--device-iter", type=int, default=1,
                   help="1 keeps the loop structurally identical to the "
                        "GPU: one host call, one device execution, one "
                        "return per sample. Higher values amortise host "
                        "overhead the GPU has no equivalent knob for, "
                        "worth ~3%, so 1 is the symmetric choice.")
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--idle-w", type=float, default=41.6,
                   help="chip idle W, recorded in the log for the "
                        "energy calculation; matches the GPU scripts")
    p.add_argument("--marker", default="official/RUNNING")
    p.add_argument("--out-dir", default="official")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    tag = f"gemm{args.dtype}_{host}_run{args.run}"

    print(f"OFFICIAL GEMM run {args.run}  |  {host}  |  N={args.n}  "
          f"resident B=1  deviceIter={args.device_iter}")
    env = C.print_environment(node=host)
    C.apply_controls(seed=C.SEED, verbose=True)

    model = GemmResident(args.n, half=(args.dtype=='fp16'))
    opts = C.poptorch_options(device_iterations=args.device_iter)
    if args.dtype == 'fp16':
        try: opts.Precision.setPartialsType(torch.float)   # fp32 accumulate
        except Exception: pass
    pop = poptorch.inferenceModel(model, opts)
    tick = torch.ones(args.device_iter, 1)

    # compile, excluded from timing and from the power trace
    t0 = time.time()
    pop.compile(tick)
    compile_s = time.time() - t0
    print(f"  compile {compile_s:.1f}s  (excluded)")

    # warm-up, excluded from timing and from the power trace
    t0 = time.time()
    for _ in range(args.warmup):
        pop(tick)
    warmup_s = time.time() - t0
    print(f"  warmup  {warmup_s:.1f}s, {args.warmup} calls  (excluded)")

    pop.copyWeightsToHost()
    count_before = float(model.count.item())

    occ_before = occupancy()

    # measured region; the marker exists for exactly this block
    print(f"  measuring {args.seconds:.0f}s ...", flush=True)
    calls = 0
    with C.RunMarker(args.marker, phase=tag):
        start_unix = time.time()
        while time.time() - start_unix < args.seconds:
            pop(tick)
            calls += 1
        end_unix = time.time()
    elapsed = end_unix - start_unix

    pop.copyWeightsToHost()
    count_after = float(model.count.item())
    pop.detachFromDevice()

    occ_after = occupancy()
    measured_iters = count_after - count_before
    assumed_iters = calls * args.device_iter
    samples = assumed_iters                 # one sample = one complete matmul

    rec = {
        "run": args.run, "node": host, "kernel": "gemm",
        "N": args.n, "device_iter": args.device_iter, "dtype": args.dtype,
        "residency": "resident_buffers",
        "warmup_calls": args.warmup, "warmup_s": round(warmup_s, 2),
        "compile_time_s": round(compile_s, 2),
        "compile_in_timing": False, "warmup_in_timing": False,
        "seconds_requested": args.seconds,
        "total_time_s": round(elapsed, 4),
        "calls": calls,
        "total_samples": samples,
        "iterations_measured_on_device": measured_iters,
        "count_verified": abs(measured_iters - assumed_iters) < 0.5,
        "throughput_samples_per_s": round(samples / elapsed, 4),
        "ms_per_sample": round(1000 * elapsed / samples, 6),
        "TFLOP_per_s": round(2 * args.n ** 3 * samples / elapsed / 1e12, 5),
        "flops_per_sample": 2 * args.n ** 3,
        "samples_per_call": 1,
        "start_unix": round(start_unix, 3),
        "end_unix": round(end_unix, 3),
        "seed": C.SEED, "idle_w": args.idle_w,
        "occupancy_before": occ_before, "occupancy_after": occ_after,
        "python": env["python"], "torch": env["torch"],
        "poptorch": env.get("poptorch"),
    }

    print(f"  throughput {rec['throughput_samples_per_s']:.3f} samples/s")
    print(f"  calls      {calls:,}  (device counter: {measured_iters:,.0f}, "
          f"verified {rec['count_verified']})")
    print(f"  samples    {samples:,}  = calls x N ({args.n:,} output rows per call)")

    log = os.path.join(args.out_dir, f"{tag}_log.csv")
    with open(log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec.keys()))
        w.writeheader(); w.writerow(rec)
    with open(os.path.join(args.out_dir, f"{tag}_log.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"  wrote {log}")


if __name__ == "__main__":
    main()
