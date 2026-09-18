#!/usr/bin/env python3
"""
One CSV row per batched run, for the repeated batched measurements.

Same joins as extract_runs.py, but the filenames carry the batch size and the
sample count is calls x B.

  python3 extract_batched_rounds.py <host> > batched_<host>.csv
"""
import sys, os, json, csv, glob, re

HOST = sys.argv[1]
KIND = "gpu"
D = os.path.expanduser("~/scripts/batchedrounds")
KERNELS = ["stencil", "triad", "gemm"]
DTYPES = ["fp32", "fp16"]


def in_region(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if rows and "phase" in rows[0]:
        rows = [r for r in rows if (r.get("phase") or "").strip()]
    return rows


def power_of(path):
    """(mean_watts, device_label, n_rows) for the working device."""
    rows = in_region(path)
    if not rows:
        return None, None, 0
    if KIND == "ipu":
        w = [float(r["chip_w"]) for r in rows
             if r.get("chip_w") not in (None, "", "nan")]
        return (sum(w) / len(w), "ipu", len(w)) if w else (None, None, 0)
    by = {}
    for r in rows:
        try:
            p, u = float(r["power_w"]), float(r.get("util_pct") or 0)
        except (TypeError, ValueError):
            continue
        by.setdefault(r.get("gpu_index", "0"), []).append((p, u))
    if not by:
        return None, None, 0
    busiest = max(by, key=lambda g: sum(u for _, u in by[g]) / len(by[g]))
    ps = [p for p, _ in by[busiest]]
    return sum(ps) / len(ps), f"gpu{busiest}", len(ps)


w = csv.writer(sys.stdout)
# one sample is one kernel invocation, so B of them per call
w.writerow(["node", "kernel", "precision", "batch", "run", "mean_power_w",
            "calls", "samples", "total_time_s", "device", "power_rows",
            "idle_w"])

for kern in KERNELS:
    for dt in DTYPES:
        # the gemm script puts the batch before the dtype, the others after
        pats = [f"{kern}{dt}_{HOST}_run*_log.json",
                f"{kern}B*{dt}_{HOST}_run*_log.json"]
        seen = set()
        rows = []
        for pat in pats:
            for jf in sorted(glob.glob(os.path.join(D, pat))):
                m = re.search(r"_run(\d+)_log\.json$", jf)
                if not m:
                    continue
                run = int(m.group(1))
                if run in seen:
                    continue
                j = json.load(open(jf))
                if j.get("dtype") and j["dtype"] != dt:
                    continue
                B = j.get("B", 1)
                pf = os.path.join(D, f"{kern}{dt}B{B}_{HOST}_run{run}_power.csv")
                if not os.path.exists(pf):
                    continue
                mw, dev, n = power_of(pf)
                if mw is None:
                    continue
                seen.add(run)
                rows.append([HOST, kern, dt, B, run, f"{mw:.4f}",
                             j["calls"], j["calls"] * B,
                             f"{j['total_time_s']:.4f}",
                             dev, n, j.get("idle_w", "")])
        for r in sorted(rows, key=lambda x: x[4]):
            w.writerow(r)
