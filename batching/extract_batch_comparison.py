#!/usr/bin/env python3
"""
Writes one CSV row per batched or unbatched measurement.

  python3 extract_batch_comparison.py <host> > batch_comparison_<host>.csv
"""
import sys, os, json, csv, glob

HOST = sys.argv[1]
D = os.path.expanduser("~/scripts/batchcmp")
MAXB = os.path.expanduser(f"~/scripts/batchcmp/max_batch_{HOST}.txt")

caps = {}
if os.path.exists(MAXB):
    for line in open(MAXB):
        k, d, b = line.split()
        caps[(k, d)] = int(b)


def mean_power(path):
    rows = [r for r in csv.DictReader(open(path))
            if (r.get("phase") or "").strip()]
    if not rows:
        return None, None
    by = {}
    for r in rows:
        try:
            p, u = float(r["power_w"]), float(r.get("util_pct") or 0)
        except (TypeError, ValueError):
            continue
        by.setdefault(r["gpu_index"], []).append((p, u))
    if not by:
        return None, None
    g = max(by, key=lambda k: sum(u for _, u in by[k]) / len(by[k]))
    ps = [p for p, _ in by[g]]
    us = [u for _, u in by[g]]
    return sum(ps) / len(ps), sum(us) / len(us)


def find_json(kern, dt, B):
    for pat in (f"{kern}{dt}_{HOST}_run{B}_log.json",
                f"{kern}B{B}{dt}_{HOST}_run{B}_log.json"):
        hit = glob.glob(os.path.join(D, pat))
        if hit:
            return hit[0]
    return None


w = csv.writer(sys.stdout)
w.writerow(["node", "kernel", "precision", "batch", "arm",
            "mean_power_w", "mean_util_pct", "samples", "total_time_s",
            "throughput_samples_per_s", "energy_mj_per_sample"])

for kern in ("stencil", "triad", "gemm"):
    for dt in ("fp32", "fp16"):
        bmax = caps.get((kern, dt), 1)
        for B, arm in ((1, "unbatched"), (bmax, "batched")):
            if arm == "batched" and bmax == 1:
                continue
            pf = os.path.join(D, f"{kern}{dt}B{B}_{HOST}_power.csv")
            jf = find_json(kern, dt, B)
            if not jf or not os.path.exists(pf):
                continue
            j = json.load(open(jf))
            mw, mu = mean_power(pf)
            if mw is None:
                continue
            ts, ns = j["total_time_s"], j["calls"] * j.get("B", 1)
            w.writerow([HOST, kern, dt, B, arm, f"{mw:.4f}", f"{mu:.2f}",
                        ns, f"{ts:.4f}", f"{ns/ts:.4f}",
                        f"{mw*ts/ns*1000:.6f}"])
