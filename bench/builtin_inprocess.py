"""Time chap-core's built-in global_median in-process, on the benchmark dataset.

    ../../chap-core/.venv/bin/python bench/builtin_inprocess.py   (from the chapkit-rs-models root)

Starts the Rust service once to generate the same dataset as bench/run.py (same seed and size),
then times train + predict of chap_core.models.builtin GlobalMedian with no HTTP, no jobs.
Prints JSON: median train and predict milliseconds, and the process's peak RSS.
"""

import json
import os
import resource
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

from chap_core.models.builtin import get_builtin_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

ROOT = Path(__file__).resolve().parent.parent


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def frame(wire):
    return pd.DataFrame(wire["data"], columns=wire["columns"])


def main():
    proc = subprocess.Popen([str(ROOT / "bin" / "baseline-global-median")], env={**os.environ, "PORT": "8198", "LOG_LEVEL": "warning"})
    try:
        for _ in range(200):
            try:
                get("http://127.0.0.1:8198/health")
                break
            except Exception:
                time.sleep(0.02)
        q = "num_locations=20&num_periods=120&num_features=3&seed=1"
        train = frame(get("http://127.0.0.1:8198/api/v1/ml/$generate-sample-data?" + q)["data"])
        future = frame(get("http://127.0.0.1:8198/api/v1/ml/$generate-sample-data?kind=predict&" + q.replace("num_periods=120", "num_periods=3"))["future"])
    finally:
        proc.terminate()
    # chap-core's DataSet wants monthly ids it can parse; YYYY-MM works.
    train_ds, future_ds = DataSet.from_pandas(train), DataSet.from_pandas(future.drop(columns=["disease_cases"]))
    t_train, t_pred = [], []
    for _ in range(50):
        model = get_builtin_model("global_median")()
        t = time.perf_counter()
        model.train(train_ds)
        t_train.append(time.perf_counter() - t)
        t = time.perf_counter()
        model.predict(train_ds, future_ds)
        t_pred.append(time.perf_counter() - t)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3
    print(json.dumps({
        "train_ms": round(statistics.median(t_train) * 1000, 3),
        "predict_ms": round(statistics.median(t_pred) * 1000, 3),
        "peak_rss_mb": round(rss_mb, 1),
        "rows": len(train),
    }))


if __name__ == "__main__":
    main()
