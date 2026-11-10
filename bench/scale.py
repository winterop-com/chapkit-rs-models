"""How chap-core's built-in baselines and the Rust ports scale with locations and periods.

    python3 bench/scale.py        (from the chapkit-rs-models root; make bench-scale)

Each size runs in its own process (bench/scale_builtin.py with chap-core's environment,
crates/baselines/examples/scale.rs in release mode) so peak memory is per run. Writes
bench/scale_results.json and prints a table.
"""

import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAP_PY = ROOT.parent.parent / "chap-core" / ".venv" / "bin" / "python"
RUST = ROOT / "target" / "release" / "examples" / "scale"
MODELS = ["global_median", "seasonal_median", "seasonal_shift_persistence"]
# (locations, periods): a locations sweep at 10 years of months, a periods sweep at 100 locations,
# then both growing together.
SIZES = [(100, 120), (500, 120), (1000, 120), (2000, 120), (5000, 120),
         (100, 500), (100, 1000), (100, 2000), (100, 5000),
         (1000, 1000), (2000, 2000)]
TIMEOUT_S = 900


def timed(cmd):
    """Run cmd under /usr/bin/time; returns (parsed JSON line or error string, peak RSS MB)."""
    flag = "-l" if platform.system() == "Darwin" else "-v"
    try:
        p = subprocess.run(["/usr/bin/time", flag, *map(str, cmd)], capture_output=True, text=True,
                           timeout=TIMEOUT_S, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return f"timeout > {TIMEOUT_S} s", None
    rss = None
    if m := re.search(r"(\d+)\s+maximum resident set size", p.stderr):
        rss = int(m.group(1)) / 1e6
    elif m := re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.stderr):
        rss = int(m.group(1)) / 1e3
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode != 0 or not lines:
        tail = (p.stderr.strip().splitlines() or ["?"])
        err = next((l for l in reversed(tail) if "Error" in l or "error" in l), tail[-1])
        return f"failed: {err[:160]}", rss
    return json.loads(lines[-1]), rss


def main():
    subprocess.run(["cargo", "build", "-q", "--release", "-p", "baselines", "--example", "scale"], cwd=ROOT, check=True)
    results = []
    for model in MODELS:
        gave_up = False
        for locs, periods in SIZES:
            rs, rs_rss = timed([RUST, model, locs, periods])
            if gave_up:
                py, py_rss = "skipped (a smaller size timed out)", None
            else:
                py, py_rss = timed([CHAP_PY, ROOT / "bench" / "scale_builtin.py", model, locs, periods])
                gave_up = isinstance(py, str) and py.startswith("timeout")
            row = {"model": model, "locations": locs, "periods": periods,
                   "rust": rs, "rust_peak_mb": rs_rss, "chap_core": py, "chap_core_peak_mb": py_rss}
            results.append(row)
            print(json.dumps(row), flush=True)
    (ROOT / "bench" / "scale_results.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
