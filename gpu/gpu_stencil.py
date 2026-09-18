"""
Stencil kernel on GPU. 3D 27-point convolution, N=384.

Runs 300 s after 10 warm-up calls. Warm-up is excluded from the timing and from
the power trace. Operands stay on the device.

  python3 gpu_stencil.py --run 1 --idle-w 43.14
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


def stencil_weights():
    w = torch.zeros(1, 1, 3, 3, 3)
    w[0, 0, 1, 1, 1] = 6.0
    for d, h, k in [(1, 1, 2), (1, 1, 0), (1, 2, 1), (1, 0, 1), (2, 1, 1), (0, 1, 1)]:
        w[0, 0, d, h, k] = -1.0
    return w



def occupancy():
    """Other processes on the GPUs. Recorded before and after every run."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=20).stdout.strip()
    except Exception as e:
        return {"error": str(e)[:120]}
    procs = [l for l in out.splitlines() if l.strip()]
    return {"n_processes": len(procs), "processes": procs[:5]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=int, required=True)
    p.add_argument("--n", type=int, default=384)
    p.add_argument("--dtype", choices=["fp32","fp16"], default="fp32")
    p.add_argument("--b", type=int, default=1)
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--idle-w", type=float, required=True)
    p.add_argument("--marker", default="official/RUNNING")
    p.add_argument("--out-dir", default="official")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    tag = f"stencil{args.dtype}_{host}_run{args.run}"
    dev = f"cuda:{args.gpu}"

    print(f"OFFICIAL stencil run {args.run}  |  {host}  |  N={args.n}  B={args.b}")
    env = C.print_environment(node=host)
    C.apply_controls(seed=C.SEED, verbose=True)

    conv = torch.nn.Conv3d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        conv.weight.copy_(stencil_weights())
    conv.weight.requires_grad_(False)
    dt = torch.float16 if args.dtype=='fp16' else torch.float32
    conv = conv.to(dev).to(dt).eval()

    x = torch.randn(args.b, 1, args.n, args.n, args.n, device=dev, dtype=dt)
    out = torch.empty_like(x)
    # device-side call counter, to check against the host count
    counter = torch.zeros(1, device=dev)

    # warm-up, excluded from timing and from the power trace
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.warmup):
            out.copy_(conv(x))
    torch.cuda.synchronize(dev)
    warmup_s = time.time() - t0
    print(f"  warmup {warmup_s:.1f}s, {args.warmup} calls (excluded; covers "
          "cuDNN algo selection)")

    occ_before = occupancy()

    # measured region
    print(f"  measuring {args.seconds:.0f}s ...", flush=True)
    calls = 0
    with C.RunMarker(args.marker, phase=tag):
        start_unix = time.time()
        with torch.no_grad():
            while time.time() - start_unix < args.seconds:
                out.copy_(conv(x))
                counter += 1.0
                torch.cuda.synchronize(dev)
                calls += 1
        end_unix = time.time()
    elapsed = end_unix - start_unix
    samples = calls * args.b
    occ_after = occupancy()
    measured_calls = float(counter.item())

    rec = {
        "run": args.run, "node": host, "kernel": "stencil",
        "N": args.n, "B": args.b, "residency": "device_resident", "dtype": args.dtype,
        "gpu_name": torch.cuda.get_device_name(dev),
        "warmup_calls": args.warmup, "warmup_s": round(warmup_s, 2),
        "compile_time_s": 0.0,
        "compile_in_timing": False, "warmup_in_timing": False,
        "seconds_requested": args.seconds,
        "total_time_s": round(elapsed, 4),
        "calls": calls, "total_samples": samples,
        "calls_measured_on_device": measured_calls,
        "count_verified": abs(measured_calls - calls) < 0.5,
        "throughput_samples_per_s": round(samples / elapsed, 4),
        "ms_per_sample": round(1000 * elapsed / samples, 6),
        "GB_per_s": round(2 * args.n ** 3 * 4 * samples / elapsed / 1e9, 4),
        "start_unix": round(start_unix, 3), "end_unix": round(end_unix, 3),
        "idle_w": args.idle_w, "seed": C.SEED,
        "occupancy_before": occ_before, "occupancy_after": occ_after,
        "python": env["python"], "torch": env["torch"], "cuda": env.get("cuda"),
    }
    print(f"  throughput {rec['throughput_samples_per_s']:.3f} samples/s  "
          f"({samples:,} samples)")
    print(f"  device counter {measured_calls:,.0f} vs {calls:,} calls  "
          f"verified {rec['count_verified']}")

    log = os.path.join(args.out_dir, f"{tag}_log.csv")
    with open(log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec.keys()))
        w.writeheader(); w.writerow(rec)
    with open(os.path.join(args.out_dir, f"{tag}_log.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"  wrote {log}")


if __name__ == "__main__":
    main()
