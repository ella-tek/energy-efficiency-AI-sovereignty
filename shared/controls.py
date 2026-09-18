"""
Shared experimental controls.

Sets the seed, fixes the dtype, disables TF32 and cuDNN autotuning, provides
RunMarker, which tells the power sampler when the timed region starts and stops.
"""

import os
import platform
import random
import re
import socket
import time

import numpy as np
import torch

SEED = 42

# expected stack per node; record_environment() flags anything different
EXPECTED = {
    "mandelbrot": {"python": "3.9.16", "torch": "1.13.1", "poptorch": "3.2.0"},
    "cricket":    {"python": "3.9.18", "torch": "2.5.1", "cuda": "12.4",
                   "cudnn": "90100", "driver": "565.57.01"},
    "locust":     {"python": "3.9.18", "torch": "2.5.1", "cuda": "12.4",
                   "cudnn": "90100", "driver": "565.57.01"},
}


def apply_controls(seed=SEED, deterministic_algorithms=False, verbose=True):
    """Apply the controls. Call before allocating any tensor."""
    # must be set before the first cuBLAS call
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    torch.set_default_dtype(torch.float32)

    applied = {"seed": seed, "default_dtype": "float32",
               "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}

    # guarded so the IPU node does not fail on a GPU-only setting
    for path, value in [("cudnn.benchmark", False),
                        ("cudnn.deterministic", True),
                        ("cudnn.allow_tf32", False),
                        ("cuda.matmul.allow_tf32", False)]:
        try:
            obj = torch.backends
            parts = path.split(".")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], value)
            applied[path] = value
        except Exception as e:
            applied[path] = f"<unavailable: {type(e).__name__}>"

    if deterministic_algorithms:
        try:
            torch.use_deterministic_algorithms(True)
            applied["use_deterministic_algorithms"] = True
        except Exception as e:
            applied["use_deterministic_algorithms"] = f"<failed: {e}>"

    if verbose:
        print("controls: seed=%d dtype=float32 tf32=off cudnn.deterministic=on"
              % seed)
        for k, v in applied.items():
            if isinstance(v, str) and v.startswith("<"):
                print(f"  NOTE {k}: {v}")

    return applied


def poptorch_options(device_iterations=1, replication=1, seed=SEED):
    """PopTorch Options with the seed set."""
    import poptorch
    opts = poptorch.Options()
    opts.deviceIterations(device_iterations)
    opts.replicationFactor(replication)
    opts.randomSeed(seed)
    return opts


def _base_version(v):
    """Strip build metadata: '1.13.1+cpu' -> '1.13.1', '3.2.0-bb50ce43ab' -> '3.2.0'."""
    m = re.match(r"\d+(?:\.\d+)*", str(v))
    return m.group() if m else str(v)


def record_environment(node=None):
    """Capture the live software stack and flag anything different from EXPECTED."""
    host = socket.gethostname().split(".")[0]
    node = node or host

    env = {
        "hostname": host,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    try:
        env["cuda"] = torch.version.cuda
        env["cudnn"] = (str(torch.backends.cudnn.version())
                        if torch.backends.cudnn.is_available() else None)
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["device_count"] = torch.cuda.device_count()
            env["devices"] = [torch.cuda.get_device_name(i)
                              for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        import poptorch
        env["poptorch"] = poptorch.__version__
    except ImportError:
        env["poptorch"] = None

    # Effective control state, read back rather than assumed.
    state = {}
    for path in ["cudnn.benchmark", "cudnn.deterministic", "cudnn.allow_tf32",
                 "cuda.matmul.allow_tf32"]:
        try:
            obj = torch.backends
            for p in path.split("."):
                obj = getattr(obj, p)
            state[path] = obj
        except Exception:
            state[path] = None
    state["default_dtype"] = str(torch.get_default_dtype())
    state["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    env["control_state"] = state

    exp = EXPECTED.get(node)
    drift = {}
    build_tags = {}
    if exp:
        for k, want in exp.items():
            got = env.get(k)
            if got is None:
                continue
            if _base_version(got) != _base_version(want):
                drift[k] = {"expected": want, "now": str(got)}
            elif str(got) != str(want):
                build_tags[k] = {"expected": want, "actual": str(got)}
    env["expected_for_node"] = exp
    env["drift_from_expected"] = drift
    env["build_metadata_only"] = build_tags
    return env


def print_environment(env=None, node=None):
    env = env or record_environment(node)
    print(f"environment: {env['hostname']}  python {env['python']}  "
          f"torch {env['torch']}  poptorch {env.get('poptorch')}  "
          f"cuda {env.get('cuda')}")
    if env["drift_from_expected"]:
        print("  stack differs from expected:")
        for k, v in env["drift_from_expected"].items():
            print(f"    {k}: expected {v['expected']}, got {v['now']}")
        print("  Results are not on the same stack as the published runs.")
    else:
        print("  stack matches expected")
    for k, v in env.get("build_metadata_only", {}).items():
        print(f"    note: {k} is {v['actual']}, expected {v['expected']}. "
              "Same version, build tag only.")
    return env
# Run marker, synchronising a separate-tab power sampler

class RunMarker:
    """Tells a sampler in another process when the timed region is live.

    Creates the marker file on entry and removes it on exit:

        with RunMarker(args.marker, phase=tag):
            ...timed region only...

    The file holds the phase name, which the sampler writes into each row.
    """

    def __init__(self, path, phase="run"):
        self.path = os.path.expanduser(path) if path else None
        self.phase = phase

    def __enter__(self):
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w") as f:
                f.write(f"{self.phase}\n{time.time()}\n")
        return self

    def __exit__(self, *exc):
        if self.path and os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass
        return False


def read_marker(path):
    """Return the phase name if the marker exists, else None."""
    try:
        with open(path) as f:
            return f.readline().strip() or "run"
    except (OSError, IOError):
        return None
