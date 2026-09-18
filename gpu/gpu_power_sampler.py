"""
Samples GPU power and writes it to CSV.

Waits for the marker file given by --wait-for, then records one row per interval
while that file exists, tagging each row with the phase name it contains. Reads
power through pynvml where available, otherwise nvidia-smi.

  python3 gpu_power_sampler.py --out power.csv --wait-for official/RUNNING \
      --interval 1.0 --idle-w 43.14
"""

__version__ = "2026-08-19a"


import csv
import os
import re
import subprocess
import threading
import time


def _try_pynvml():
    try:
        import pynvml
    except ImportError:
        return None, 0
    try:
        pynvml.nvmlInit()
    except Exception:
        return None, 0
    n = pynvml.nvmlDeviceGetCount()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]

    def read():
        out = []
        for i, h in enumerate(handles):
            try:
                p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                p = None
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            except Exception:
                u = None
            try:
                t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                t = None
            try:
                c = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            except Exception:
                c = None
            out.append({"gpu_index": i, "power_w": p, "util_pct": u,
                        "temp_c": t, "clock_mhz": c})
        return out

    return read, n


def _try_nvidia_smi():
    q = "index,power.draw,utilization.gpu,temperature.gpu,clocks.sm"
    cmd = ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None, 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, 0
    n = len([l for l in r.stdout.strip().splitlines() if l.strip()])

    def read():
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=10).stdout
        except subprocess.TimeoutExpired:
            return []
        rows = []
        for line in out.strip().splitlines():
            cells = [c.strip() for c in line.split(",")]
            if len(cells) < 5:
                continue
            rows.append({
                "gpu_index": _f(cells[0], int),
                "power_w": _f(cells[1], float),
                "util_pct": _f(cells[2], float),
                "temp_c": _f(cells[3], float),
                "clock_mhz": _f(cells[4], float),
            })
        return rows

    return read, n


def _f(s, cast):
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return cast(float(m.group())) if m else None


def get_reader():
    r, n = _try_pynvml()
    if r is not None:
        return r, n, "pynvml"
    r, n = _try_nvidia_smi()
    if r is not None:
        return r, n, "nvidia-smi"
    raise RuntimeError("No NVIDIA power source available (no pynvml, no nvidia-smi).")


class PowerSampler:
    """Samples every visible GPU. Use as a context manager.

        with PowerSampler("out.csv") as s:
            ...workload...
        print(s.summary(gpu_index=0, idle_w=62.0))
    """

    FIELDS = ["unix_time", "elapsed_s", "gpu_index", "power_w",
              "util_pct", "temp_c", "clock_mhz", "phase"]

    def __init__(self, out_path, interval_s=0.1):
        self.out_path = out_path
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._read, self.n_gpus, self.backend = get_reader()
        self.phase = None
        self.t0 = None

    def _loop(self):
        next_t = time.time()
        while not self._stop.is_set():
            now = time.time()
            for row in self._read():
                row["unix_time"] = round(now, 4)
                row["elapsed_s"] = round(now - self.t0, 4)
                row["phase"] = self.phase
                self.samples.append(row)
            next_t += self.interval_s
            sleep = next_t - time.time()
            if sleep > 0:
                self._stop.wait(sleep)
            else:
                next_t = time.time()

    def start(self):
        self.t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        d = os.path.dirname(os.path.abspath(self.out_path))
        os.makedirs(d, exist_ok=True)
        with open(self.out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            for s in self.samples:
                w.writerow({k: s.get(k) for k in self.FIELDS})
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def series(self, gpu_index):
        return [(s["elapsed_s"], s["power_w"]) for s in self.samples
                if s["gpu_index"] == gpu_index and s.get("power_w") is not None]

    def integrate(self, gpu_index):
        pts = self.series(gpu_index)
        return sum((pts[i + 1][0] - pts[i][0]) * (pts[i + 1][1] + pts[i][1]) / 2.0
                   for i in range(len(pts) - 1))

    def summary(self, gpu_index=0, idle_w=None):
        pts = self.series(gpu_index)
        if not pts:
            return {"n_samples": 0, "error": f"no samples for gpu {gpu_index}"}
        w = [p for _, p in pts]
        out = {
            "backend": self.backend,
            "n_gpus_visible": self.n_gpus,
            "gpu_index": gpu_index,
            "n_samples": len(w),
            "duration_s": round(pts[-1][0], 4),
            "mean_w": round(sum(w) / len(w), 3),
            "min_w": round(min(w), 3),
            "max_w": round(max(w), 3),
            "energy_j": round(self.integrate(gpu_index), 3),
        }
        # intra-run drift: first decile vs last
        # was 15-60x larger than the run-to-run spread that was reported.
        k = max(1, len(w) // 10)
        first, last = sum(w[:k]) / k, sum(w[-k:]) / k
        out["drift_w"] = round(last - first, 3)
        out["drift_pct"] = round(100 * (last - first) / first, 3) if first else None

        if idle_w is not None:
            out["idle_w"] = idle_w
            out["dynamic_energy_j"] = round(
                out["energy_j"] - idle_w * out["duration_s"], 3)
            out["duty_cycle"] = round(
                len([x for x in w if x > idle_w * 1.10]) / len(w), 4)
        return out

    def all_gpu_means(self):
        idx = sorted({s["gpu_index"] for s in self.samples})
        return {i: self.summary(i)["mean_w"] for i in idx}


# Standalone mode: run in a separate terminal alongside the benchmark.
#
#   Tab 1:  python3 gpu_power_sampler.py --out results/gemm_run1_power.csv
#   Tab 2:  <run the benchmark>
#   Tab 1:  Ctrl-C when the benchmark finishes
#
# Every sample carries an absolute Unix timestamp, so the CSV can be aligned
# against the benchmark log's start_unix/end_unix afterwards. That alignment is
# absolute timestamps, so a trace can be sliced to any sub-interval
# counter at 1 s resolution.

if __name__ == "__main__":
    import argparse
    import signal
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import controls as C

    p = argparse.ArgumentParser(description="Standalone GPU power logger")
    p.add_argument("--out", default="results/power.csv")
    p.add_argument("--interval", type=float, default=0.1)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--idle-w", type=float, default=None)
    p.add_argument("--wait-for", default=None, metavar="MARKER",
                   help="record ONLY while this marker file exists, so warm-up "
                        "and idle never enter the trace")
    p.add_argument("--once", action="store_true")
    p.add_argument("--version", action="store_true")
    a = p.parse_args()

    print(f"gpu_power_sampler {__version__}")
    if a.version:
        raise SystemExit(0)

    s = PowerSampler(a.out, a.interval)
    print(f"backend {s.backend}, {s.n_gpus} GPU(s) visible, {a.interval}s interval")

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    if a.wait_for:
        marker = os.path.expanduser(a.wait_for)
        print(f"armed. watching {marker}")
        s.start(); s._stop.set()          # armed but paused
        recording = False
        try:
            while not stop["now"]:
                phase = C.read_marker(marker)
                if phase and not recording:
                    s.phase = phase; s.t0 = time.time(); s._stop.clear()
                    s._thread = threading.Thread(target=s._loop, daemon=True)
                    s._thread.start(); recording = True
                    print(f"  >> RECORDING  phase={phase}")
                elif not phase and recording:
                    s._stop.set(); recording = False
                    print(f"  << stopped. {len(s.samples)} samples so far")
                    if a.once:
                        break
                elif phase and recording and phase != s.phase:
                    s.phase = phase
                time.sleep(0.05)
        finally:
            s._stop.set()
            if s._thread:
                s._thread.join(timeout=5)
            with open(s.out_path, "w", newline="") as f:
                import csv as _csv
                w = _csv.DictWriter(f, fieldnames=s.FIELDS); w.writeheader()
                for row in s.samples:
                    w.writerow({k: row.get(k) for k in s.FIELDS})
    else:
        print(f"logging to {a.out}, Ctrl-C to stop")
        s.start()
        try:
            t0 = time.time()
            while not stop["now"]:
                if a.seconds and time.time() - t0 >= a.seconds:
                    break
                time.sleep(0.2)
        finally:
            s.stop()

    print("\nstopped.")
    for i in sorted({x["gpu_index"] for x in s.samples}):
        d = s.summary(i, idle_w=a.idle_w)
        print(f"  gpu{i}: " + "  ".join(f"{k}={v}" for k, v in d.items()
                                        if k not in ("backend","n_gpus_visible","gpu_index")))
    print(f"  wrote {a.out}")
