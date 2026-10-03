#!/usr/bin/env python3
"""
Allocation inefficiency and the performance/efficiency trade-off.

  python3 derive_comparison_metrics.py results/summary/results.csv results/summary/

Allocation inefficiency is each node's energy per sample divided by the lowest
energy per sample for that kernel and precision, so 1.0 marks the most
efficient node and the baseline changes between workloads.

The trade-off compares the fastest node against the most efficient one: the
energy saved and the throughput given up by choosing the latter, both relative
to the fastest node. Where one node is both, no trade-off arises.
"""
import sys, os, csv

SRC, OUT = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(SRC)))
for r in rows:
    r["E"] = float(r["energy_mj_per_sample"])
    r["R"] = float(r["throughput_samples_per_s"])

cells = sorted({(r["kernel"], r["precision"]) for r in rows},
               key=lambda c: (["stencil", "triad", "gemm"].index(c[0]), c[1]))

alloc = [["kernel", "precision", "node", "energy_mj_per_sample",
          "allocation_inefficiency", "is_most_efficient"]]
trade = [["kernel", "precision", "fastest_node", "most_efficient_node",
          "energy_saved_pct", "throughput_lost_pct"]]

for kern, dt in cells:
    d = [r for r in rows if r["kernel"] == kern and r["precision"] == dt]
    best_E = min(d, key=lambda r: r["E"])
    best_R = max(d, key=lambda r: r["R"])
    for r in sorted(d, key=lambda r: r["E"]):
        alloc.append([kern, dt, r["node"], round(r["E"], 4),
                      round(r["E"] / best_E["E"], 4),
                      "yes" if r is best_E else "no"])
    if best_R["node"] == best_E["node"]:
        trade.append([kern, dt, best_R["node"], best_E["node"], "", ""])
    else:
        trade.append([kern, dt, best_R["node"], best_E["node"],
                      round((1 - best_E["E"] / best_R["E"]) * 100, 2),
                      round((1 - best_E["R"] / best_R["R"]) * 100, 2)])

for name, data in (("allocation.csv", alloc), ("tradeoff.csv", trade)):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        csv.writer(f).writerows(data)
    print("wrote", os.path.join(OUT, name), "(%d rows)" % (len(data) - 1))
