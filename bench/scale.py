"""How chap-core's built-in baselines and the chapkit-rs ports scale with locations and periods.

    python3 bench/scale.py                     (from the chapkit-rs-models root; make bench-scale)
    python3 bench/scale.py 50000x120 100000x120   (only these sizes, merged into the results file)

Each size runs in its own process (bench/scale_builtin.py with chap-core's environment,
the baseline-bench binary in release mode) so peak memory is per run. The chapkit-rs ports run
on one core (RAYON_NUM_THREADS=1, like chap-core) and on all cores, then once more to time
decoding the data from JSON. CPU time is measured inside each process around train and predict. Writes
bench/scale_results.json and prints a table.
"""

import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# CHAP_PY=<python> picks chap-core's environment (default: ../../chap-core/.venv).
CHAP_PY = os.environ.get("CHAP_PY") or ROOT.parent.parent / "chap-core" / ".venv" / "bin" / "python"
RUST = ROOT / "target" / "release" / "baseline-bench"
MODELS = ["global_median", "persistence", "seasonal_median", "seasonal_shift_persistence"]
# (locations, periods): a locations sweep at 10 years of months up to facility-level counts
# (100,000 locations), a periods sweep at 100 locations, then both growing together.
SIZES = [(100, 120), (500, 120), (1000, 120), (2000, 120), (5000, 120),
         (10000, 120), (20000, 120), (50000, 120), (100000, 120),
         (100, 500), (100, 1000), (100, 2000), (100, 5000),
         (1000, 1000), (2000, 2000)]
TIMEOUT_S = 900


def timed(cmd, env=None):
    """Run cmd under /usr/bin/time; returns (parsed JSON line or error string, peak RSS MB)."""
    flag = "-l" if platform.system() == "Darwin" else "-v"
    try:
        p = subprocess.run(["/usr/bin/time", flag, *map(str, cmd)], capture_output=True, text=True,
                           timeout=TIMEOUT_S, cwd=ROOT, env={**os.environ, **(env or {})})
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
    sizes = [tuple(map(int, a.split("x"))) for a in sys.argv[1:]] or SIZES
    out = ROOT / "bench" / "scale_results.json"
    # Runs of other sizes already in the results file are kept.
    previous = json.loads(out.read_text()) if sys.argv[1:] and out.exists() else []
    subprocess.run(["cargo", "build", "-q", "--release", "-p", "baseline-bench"], cwd=ROOT, check=True)
    results = []
    for model in MODELS:
        gave_up = False
        for locs, periods in sizes:
            # Model runs without the JSON copy, so peak memory is the data plus the model.
            rs, rs_rss = timed([RUST, model, locs, periods], {"RAYON_NUM_THREADS": "1", "SCALE_NO_DECODE": "1"})
            rs_par, rs_par_rss = timed([RUST, model, locs, periods], {"SCALE_NO_DECODE": "1"})
            # Decoding the same data from JSON, as it arrives over HTTP, timed separately.
            rs_dec, _ = timed([RUST, model, locs, periods], {"RAYON_NUM_THREADS": "1"})
            if gave_up:
                py, py_rss = "skipped (a smaller size timed out)", None
            else:
                py, py_rss = timed([CHAP_PY, ROOT / "bench" / "scale_builtin.py", model, locs, periods])
                gave_up = isinstance(py, str) and py.startswith("timeout")
            row = {"model": model, "locations": locs, "periods": periods,
                   "rust": rs, "rust_peak_mb": rs_rss, "rust_all_cores": rs_par, "rust_all_cores_peak_mb": rs_par_rss,
                   "rust_decode_ms": rs_dec.get("decode_ms") if isinstance(rs_dec, dict) else rs_dec,
                   "json_mb": rs_dec.get("json_mb") if isinstance(rs_dec, dict) else None, "chap_core": py, "chap_core_peak_mb": py_rss}
            results.append(row)
            print(json.dumps(row), flush=True)
    done = {(r["model"], r["locations"], r["periods"]) for r in results}
    kept = [r for r in previous if (r["model"], r["locations"], r["periods"]) not in done]
    order = {s: i for i, s in enumerate(SIZES)}
    merged = sorted(kept + results, key=lambda r: (MODELS.index(r["model"]), order.get((r["locations"], r["periods"]), len(order))))
    out.write_text(json.dumps(merged, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
