"""
Samples IPU power and writes it to CSV.

Waits for the marker file given by --wait-for, then records one row per interval
while that file exists. Reads power by parsing gc-monitor output.

  python3 ipu_power_sampler.py --out power.csv --wait-for official/RUNNING \
      --interval 1.0 --idle-w 41.6
"""

__version__ = "2026-08-19a"


import csv
import os
import re
import subprocess
import threading
import time
# Backend detection

# Column indices in `gc-monitor -c -s --no-headers` CSV output, confirmed
_GCM_DEVICE_ID = 6
_GCM_CHIP_W    = 25   # "ipu power",   e.g. "40.9 W"
_GCM_BOARD_W   = 18   # "board power", e.g. "151.8 W"
_GCM_DIE_C     = 16
_GCM_BOARD_C   = 17
_GCM_CLOCK     = 15
_GCM_UTIL      = 23


def _try_gc_monitor():
    """Reader using the gc-monitor CLI.

    Note gc-monitor only reports power for IPUs with an attached process, so
    it reads nothing while the device is idle.
    """
    cmd = ["gc-monitor", "-c", "-s", "--no-headers"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if r.returncode != 0 and not r.stdout.strip():
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    def read(device_id):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=20).stdout
        except subprocess.TimeoutExpired:
            return None
        for line in out.splitlines():
            cells = line.split(",")
            if len(cells) <= _GCM_CHIP_W:
                continue
            if _to_float(cells[_GCM_DEVICE_ID]) != float(device_id):
                continue
            chip = _to_float(cells[_GCM_CHIP_W])
            if chip is None:          # no attached process -> no power reported
                return None
            return {
                "chip_w": chip,
                "chassis_w": _to_float(cells[_GCM_BOARD_W]),
                "die_temp_c": _to_float(cells[_GCM_DIE_C]),
                "board_temp_c": _to_float(cells[_GCM_BOARD_C]),
                "clock_mhz": _to_float(cells[_GCM_CLOCK]),
                "util": _to_float(cells[_GCM_UTIL]),
            }
        return None

    return read


def _to_float(v):
    if v is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else None


def get_reader():
    """Return (reader_fn, backend_name) for gc-monitor."""
    r = _try_gc_monitor()
    if r is None:
        raise RuntimeError("gc-monitor not on PATH or not responding.")
    return r, "gc-monitor"


def rate_probe(device_id=0, seconds=20.0):
    """Empirically determine how fast the power sensor actually updates.

    Polls as fast as the interface allows and counts how often the VALUE
    changes, as opposed to how often we asked. Sampling faster than the sensor
    refreshes yields duplicate readings, not information, so this sets the
    ceiling on any sensible sampling rate.
    """
    read, backend = get_reader()
    t0 = time.time()
    samples = []
    while time.time() - t0 < seconds:
        v = read(device_id)
        if v and v.get("chip_w") is not None:
            samples.append((time.time(), v["chip_w"]))
    if len(samples) < 10:
        print("too few samples to analyse"); return

    dur = samples[-1][0] - samples[0][0]
    poll_hz = len(samples) / dur

    changes, dwell, last_t, last_v = 0, [], samples[0][0], samples[0][1]
    for t, v in samples[1:]:
        if v != last_v:
            changes += 1
            dwell.append(t - last_t)
            last_t, last_v = t, v
    update_hz = changes / dur if dur else 0

    print(f"backend            : {backend}")
    print(f"polled             : {len(samples)} times in {dur:.2f}s "
          f"= {poll_hz:.1f} Hz")
    print(f"value changed      : {changes} times = {update_hz:.2f} Hz")
    print(f"distinct values    : {len(set(v for _, v in samples))}")
    if dwell:
        dwell.sort()
        print(f"time between changes: median {dwell[len(dwell)//2]*1000:.0f} ms  "
              f"min {min(dwell)*1000:.0f} ms  max {max(dwell)*1000:.0f} ms")
    print()
    if update_hz < 0.5:
        print("The sensor updates very slowly. Sampling fast buys nothing;")
        print("anything from 0.5-2 Hz captures all available information.")
    else:
        rec = min(20.0, max(2.0, 4 * update_hz))
        print(f"Sensor refreshes about {update_hz:.1f} times per second.")
        print(f"Recommended sampling: ~{rec:.0f} Hz (interval {1/rec:.2f}s),")
        print("i.e. roughly 4x the refresh rate, enough to catch every")
        print("update without wasting effort on duplicates.")
    print()
    print("Whatever you pick, report BOTH the sampling rate and this measured")
    print("sensor refresh rate. They are different numbers and reviewers ask.")


class ChassisReader:
    """Board/chassis power for one device, via gc-monitor.

    Polled on a slower cadence than the die reading, since it spawns a
    subprocess and changes slowly.
    """

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.available = False
        try:
            r = subprocess.run(["gc-monitor", "--no-card-info"],
                               capture_output=True, text=True, timeout=15)
            self.available = r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.available = False

    def read(self):
        if not self.available:
            return None
        try:
            out = subprocess.run(["gc-monitor", "--no-card-info"],
                                 capture_output=True, text=True,
                                 timeout=15).stdout
        except subprocess.TimeoutExpired:
            return None
        for line in out.splitlines():
            if "|" not in line:
                continue
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if str(self.device_id) not in cells:
                continue
            watts = [_to_float(c) for c in cells if c.endswith(" W")]
            watts = [w for w in watts if w is not None]
            if len(watts) >= 2:
                # Two power columns per row: chip (smaller) and chassis
                # (larger, includes gateway, DDR, fans, PSU losses).
                return max(watts)
        return None


class PowerSampler:
    """Background power sampler.

    Usage:
        with PowerSampler("out_power.csv", device_id=0) as s:
            ...workload...
        print(s.summary())
    """

    FIELDS = ["unix_time", "elapsed_s", "chip_w", "chassis_w",
              "die_temp_c", "board_temp_c", "clock_mhz", "util", "phase"]

    def __init__(self, out_path, device_id=0, interval_s=0.1,
                 chassis_interval_s=0, backend="auto"):
        self.out_path = out_path
        self.device_id = device_id
        self.interval_s = interval_s
        self.chassis_interval_s = chassis_interval_s
        self._chassis = ChassisReader(device_id) if chassis_interval_s else None
        self._chassis_value = None
        self._chassis_fresh = False
        self._chassis_thread = None
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._read, self.backend = get_reader(backend)
        self.phase = None
        self.read_errors = 0
        self.t0 = None

    def _loop(self):
        next_t = time.time()
        while not self._stop.is_set():
            now = time.time()
            try:
                row = self._read(self.device_id)
            except Exception as e:
                self.read_errors += 1
                if self.read_errors == 1:
                    print(f"\n  POWER SAMPLER READ FAILED: {type(e).__name__}: {e}")
                    print("  Sampling continues but no data is being captured.")
                row = None
            if row is not None:
                row["unix_time"] = round(now, 4)
                row["elapsed_s"] = round(now - self.t0, 4)
                row["phase"] = self.phase
                if self._chassis_fresh:
                    row["chassis_w"] = self._chassis_value
                    self._chassis_fresh = False
                self.samples.append(row)
            next_t += self.interval_s
            sleep = next_t - time.time()
            if sleep > 0:
                self._stop.wait(sleep)
            else:
                # Sampling can't keep up; resync rather than spiral.
                next_t = time.time()

    def _chassis_loop(self):
        while not self._stop.is_set():
            v = self._chassis.read()
            if v is not None:
                self._chassis_value = v
                self._chassis_fresh = True
            self._stop.wait(self.chassis_interval_s)

    def start(self):
        self.t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if self._chassis is not None and self._chassis.available:
            self._chassis_thread = threading.Thread(
                target=self._chassis_loop, daemon=True)
            self._chassis_thread.start()
        elif self._chassis is not None:
            print("  NOTE: gc-monitor unavailable, so no chassis power will be "
                  "recorded. Chip power only.")
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._write()
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def _write(self):
        d = os.path.dirname(os.path.abspath(self.out_path))
        os.makedirs(d, exist_ok=True)
        with open(self.out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            for s in self.samples:
                w.writerow({k: s.get(k) for k in self.FIELDS})

    def summary(self, idle_w=None):
        """Summary stats. Energy is integrated (trapezoid), not mean x duration."""
        chip = [s["chip_w"] for s in self.samples if s.get("chip_w") is not None]
        chas = [s["chassis_w"] for s in self.samples if s.get("chassis_w") is not None]
        if not chip:
            return {"n_samples": 0, "error": "no samples captured"}

        out = {
            "backend": self.backend,
            "n_samples": len(self.samples),
            "duration_s": round(self.samples[-1]["elapsed_s"], 4),
            "chip_mean_w": round(sum(chip) / len(chip), 3),
            "chip_min_w": round(min(chip), 3),
            "chip_max_w": round(max(chip), 3),
            "chip_energy_j": round(self._integrate("chip_w"), 3),
        }
        if chas:
            out.update({
                "chassis_mean_w": round(sum(chas) / len(chas), 3),
                "chassis_energy_j": round(self._integrate("chassis_w"), 3),
            })
        if idle_w is not None:
            out["idle_w"] = idle_w
            out["chip_dynamic_energy_j"] = round(
                out["chip_energy_j"] - idle_w * out["duration_s"], 3)
            above = [w for w in chip if w > idle_w * 1.10]
            out["duty_cycle"] = round(len(above) / len(chip), 4)
        return out

    def _integrate(self, key):
        """Trapezoidal integration of power over time -> joules."""
        pts = [(s["elapsed_s"], s[key]) for s in self.samples
               if s.get(key) is not None]
        return sum(
            (pts[i + 1][0] - pts[i][0]) * (pts[i + 1][1] + pts[i][1]) / 2.0
            for i in range(len(pts) - 1)
        )


if __name__ == "__main__":
    import argparse
    import signal
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import controls as C

    p = argparse.ArgumentParser(description="Standalone IPU power logger")
    p.add_argument("--out", default="results/power.csv")
    p.add_argument("--interval", type=float, default=0.1,
                   help="chip sampling interval")
    p.add_argument("--chassis-interval", type=float, default=0,
                   help="OFF by default. Chip power is the measurement. Set to "
                        "e.g. 2.0 only if you separately need M2000 board power "
                        "for a GPU boundary comparison.")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--idle-w", type=float, default=None)
    p.add_argument("--seconds", type=float, default=None,
                   help="free-running mode: stop after N seconds")
    p.add_argument("--wait-for", default=None, metavar="MARKER",
                   help="record ONLY while this marker file exists. The "
                        "benchmark creates it when the timed region starts and "
                        "removes it when the region ends, so the log contains "
                        "no idle head or tail.")
    p.add_argument("--version", action="store_true",
                   help="print the file version and exit")
    p.add_argument("--rate-probe", type=float, default=0, metavar="SECONDS",
                   help="measure how fast the power sensor actually updates, "
                        "then exit. Run this once to choose a defensible "
                        "sampling rate. e.g. --rate-probe 20")
    p.add_argument("--once", action="store_true",
                   help="with --wait-for: exit after the first phase instead "
                        "of waiting for the next one")
    a = p.parse_args()

    print(f"ipu_power_sampler {__version__}")
    if a.version:
        raise SystemExit(0)


    if a.rate_probe:
        rate_probe(a.device, a.rate_probe)
        raise SystemExit(0)

    s = PowerSampler(a.out, a.device, a.interval, a.chassis_interval,
                     backend=a.backend)
    print(f"backend {s.backend}, device {a.device}, {a.interval}s interval")

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    if a.wait_for:
        marker = os.path.expanduser(a.wait_for)
        print(f"armed. watching {marker}")
        print("recording starts the moment the benchmark creates it. Ctrl-C to quit.")
        s.start()
        s._stop.set()          # armed but paused: no samples collected yet
        recording = False
        try:
            while not stop["now"]:
                phase = C.read_marker(marker)
                if phase and not recording:
                    s.phase = phase
                    s.t0 = time.time()
                    s._stop.clear()
                    s._thread = threading.Thread(target=s._loop, daemon=True)
                    s._thread.start()
                    recording = True
                    print(f"  >> RECORDING  phase={phase}")
                elif not phase and recording:
                    s._stop.set()
                    recording = False
                    n = len(s.samples)
                    print(f"  << stopped. {n} samples so far")
                    if a.once:
                        break
                elif phase and recording and phase != s.phase:
                    s.phase = phase          # phase changed, keep going
                    print(f"  >> phase={phase}")
                time.sleep(0.05)
        finally:
            s._stop.set()
            if s._thread:
                s._thread.join(timeout=5)
            s._write()
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
    d = s.summary(idle_w=a.idle_w)
    for k, v in d.items():
        print(f"  {k}: {v}")
    phases = sorted({x.get("phase") for x in s.samples if x.get("phase")})
    if phases:
        print(f"  phases captured: {phases}")
    print(f"  wrote {a.out}")
