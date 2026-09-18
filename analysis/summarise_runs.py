#!/usr/bin/env python3
"""Mean and standard deviation per kernel, precision and node.
"""
import sys, os, json, csv, glob, re, statistics as st

HOST = sys.argv[1]
KIND = sys.argv[2]
D = os.path.expanduser("~/scripts/official")

KERNELS = ["stencil", "triad", "gemm"]
DTYPES = ["fp32", "fp16"]


def power_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    if "phase" in rows[0]:
        rows = [r for r in rows if (r.get("phase") or "").strip()]
    return rows


def mean_watts(path):
    """Mean device power over the measured region, for the working device."""
    rows = power_rows(path)
    if not rows:
        return None
    if KIND == "ipu":
        w = [float(r["chip_w"]) for r in rows
             if r.get("chip_w") not in (None, "", "nan")]
        return sum(w) / len(w) if w else None

    # GPU: pick the busiest gpu_index, then average its power
    by_gpu = {}
    for r in rows:
        gi = r.get("gpu_index", "0")
        try:
            p = float(r["power_w"]); u = float(r.get("util_pct") or 0)
        except (TypeError, ValueError):
            continue
        by_gpu.setdefault(gi, []).append((p, u))
    if not by_gpu:
        return None
    busiest = max(by_gpu, key=lambda g: sum(u for _, u in by_gpu[g]) / len(by_gpu[g]))
    ps = [p for p, _ in by_gpu[busiest]]
    return sum(ps) / len(ps)


def cell(kern, dt):
    """Return list of (run, mJ_per_sample, throughput) for one cell."""
    pats = [f"{kern}{dt}_{HOST}_run*_log.json"]
    if dt == "fp32":
        pats.append(f"{kern}_{HOST}_run*_log.json")   # round-1 legacy naming
    seen, out = set(), []
    for pat in pats:
        for jf in sorted(glob.glob(os.path.join(D, pat))):
            m = re.search(r"_run(\d+)_log\.json$", jf)
            if not m:
                continue
            run = int(m.group(1))
            if run in seen:
                continue
            with open(jf) as f:
                j = json.load(f)
            if j.get("dtype") and j["dtype"] != dt:
                continue
            tag = kern if dt == "fp32" else f"{kern}{dt}"
            pf = os.path.join(D, f"{tag}_{HOST}_run{run}_power.csv")
            if not os.path.exists(pf):
                continue
            w = mean_watts(pf)
            if w is None:
                continue
            ts, ns = j["total_time_s"], j["calls"] * j.get("B", 1)
            if not ns:
                continue
            seen.add(run)
            out.append((run, w * ts / ns * 1000.0, ns / ts, w))
    return sorted(out)


def fmt(vals):
    if not vals:
        return "n/a", "n/a"
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


print(f"# {HOST} ({KIND})")
print(f"{'kernel':<8} {'dtype':<5} {'n':>3} "
      f"{'mJ/sample':>14} {'sd':>10} {'cv%':>6} "
      f"{'samples/s':>16} {'sd':>12} {'meanW':>8}")
for kern in KERNELS:
    for dt in DTYPES:
        rows = cell(kern, dt)
        e = [r[1] for r in rows]
        t = [r[2] for r in rows]
        w = [r[3] for r in rows]
        if not rows:
            print(f"{kern:<8} {dt:<5} {0:>3}   (no data)")
            continue
        em, es = fmt(e)
        tm, ts_ = fmt(t)
        wm, _ = fmt(w)
        cv = (es / em * 100) if em else 0
        print(f"{kern:<8} {dt:<5} {len(rows):>3} "
              f"{em:>14.4f} {es:>10.4f} {cv:>6.2f} "
              f"{tm:>16.3f} {ts_:>12.3f} {wm:>8.2f}")
        runs = [r[0] for r in rows]
        missing = sorted(set(range(1, 21)) - set(runs))
        if missing:
            print(f"{'':<8} {'':<5}     missing runs: {missing}")
