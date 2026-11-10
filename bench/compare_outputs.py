"""Check that the chapkit-rs ports give chap-core's numbers on the benchmark data, not just on the fixtures.

    python3 bench/compare_outputs.py      (from the chapkit-rs-models root; make bench-compare)

Runs chap-core's built-in baselines (bench/scale_builtin.py) and the chapkit-rs ports (baseline-bench)
on the same generated data, both writing every forecast, and compares all 100 samples of every
forecast row. Prints the largest difference and the number of samples that are not bit-identical
per model and size; fails if any sample is not bit-identical. CHAP_PY=<python> picks chap-core's
environment (default ../../chap-core/.venv).
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAP_PY = os.environ.get("CHAP_PY") or ROOT.parent.parent / "chap-core" / ".venv" / "bin" / "python"
RUST = ROOT / "target" / "release" / "baseline-bench"
MODELS = ["global_median", "persistence", "seasonal_median", "seasonal_shift_persistence"]
SIZES = [(100, 120), (1000, 120), (5000, 120), (100, 1000), (1000, 1000)]


def key(k):
    """`location|1850-01` and chap-core's `location|185001` to one form."""
    loc, period = k.split("|")
    return f"{loc}|{period.replace('-', '')}"


def run(cmd, dump):
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True, capture_output=True, env={"SCALE_DUMP": dump, "PATH": "/usr/bin:/bin"})
    return {key(k): v for k, v in json.loads(Path(dump).read_text()).items()}


def main():
    subprocess.run(["cargo", "build", "-q", "--release", "-p", "baseline-bench"], cwd=ROOT, check=True)
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for model in MODELS:
            for locs, periods in SIZES:
                rs = run([RUST, model, locs, periods], f"{tmp}/rs.json")
                py = run([CHAP_PY, ROOT / "bench" / "scale_builtin.py", model, locs, periods], f"{tmp}/py.json")
                worst, inexact = 0.0, 0
                problems = []
                if rs.keys() != py.keys():
                    problems.append(f"rows differ: {len(rs.keys() - py.keys())} only in Rust, {len(py.keys() - rs.keys())} only in chap-core")
                for k in rs.keys() & py.keys():
                    a, b = rs[k], py[k]
                    if len(a) != len(b):
                        problems.append(f"{k}: {len(a)} vs {len(b)} samples")
                        continue
                    for x, y in zip(a, b):
                        inexact += x != y
                        worst = max(worst, abs(x - y) / max(abs(y), 1.0))
                ok = not problems and inexact == 0
                failed += not ok
                cells = len(rs) * 100
                print(f"{'ok  ' if ok else 'FAIL'} {model:28} {locs:>5} x {periods:<5} {len(rs):>6} rows, {cells:>9,} samples, {inexact:>6} not bit-identical, max relative difference {worst:.1e}"
                      + ("" if not problems else "  " + "; ".join(problems[:3])), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
