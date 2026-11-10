"""Time one of chap-core's built-in baselines in-process on a generated dataset. Used by bench/scale.py.

    ../../chap-core/.venv/bin/python bench/scale_builtin.py <model> <locations> <periods>

Same deterministic data as baseline_core::synthetic (crates/baseline-core/src/synthetic.rs). Prints one JSON line with median
milliseconds for converting pandas to chap-core's DataSet, train and predict.

With SCALE_DUMP=<file>, also writes every forecast as JSON {"location|period": [samples]} for
bench/compare_outputs.py.
"""

import json
import os
import statistics
import sys
import time

import numpy as np
import pandas as pd

from chap_core.models.builtin import get_builtin_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

START_YEAR = 1850
HORIZON = 3


def frame(locs, start, n, with_y):
    p = np.tile(np.arange(start, start + n), locs)
    loc = np.repeat(np.arange(locs), n)
    season = np.sin((p % 12) / 12.0 * 2 * np.pi)
    df = pd.DataFrame({
        "time_period": [f"{START_YEAR + q // 12}-{q % 12 + 1:02d}" for q in p],
        "location": [f"loc{l}" for l in loc],
        "rainfall": 100.0 + 50.0 * season + loc,
        "mean_temperature": 25.0 + 3.0 * season,
    })
    if with_y:
        # Rust's f64::round rounds halves away from zero; all values here are positive.
        df["disease_cases"] = np.floor(20.0 + 10.0 * season + 2.0 * (loc % 50) + (p % 7) + 0.5)
    return df


def main():
    model_name, locs, periods = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    train_df = frame(locs, 0, periods, True)
    future_df = frame(locs, periods, HORIZON, False)
    reps = 5 if len(train_df) <= 100_000 else 1
    conv, fit, pred = [], [], []
    fit_cpu, pred_cpu = [], []  # process CPU time (all threads) during train and predict
    out_rows = 0
    for _ in range(reps):
        t = time.perf_counter()
        train, future = DataSet.from_pandas(train_df), DataSet.from_pandas(future_df)
        conv.append((time.perf_counter() - t) * 1e3)

        model = get_builtin_model(model_name)()
        t, c = time.perf_counter(), time.process_time()
        model.train(train)
        fit.append((time.perf_counter() - t) * 1e3)
        fit_cpu.append((time.process_time() - c) * 1e3)

        t, c = time.perf_counter(), time.process_time()
        out = model.predict(train, future)
        pred.append((time.perf_counter() - t) * 1e3)
        pred_cpu.append((time.process_time() - c) * 1e3)
        out_rows = sum(len(v.time_period) for v in out.values())
    if path := os.environ.get("SCALE_DUMP"):
        dump = {}
        for location, samples in out.items():
            for period, row in zip(samples.time_period, samples.samples):
                dump[f"{location}|{period.id}"] = [float(x) for x in row]
        with open(path, "w") as f:
            json.dump(dump, f)
    print(json.dumps({
        "rows": len(train_df),
        "convert_ms": statistics.median(conv),
        "train_ms": statistics.median(fit),
        "predict_ms": statistics.median(pred),
        "train_cpu_ms": statistics.median(fit_cpu),
        "predict_cpu_ms": statistics.median(pred_cpu),
        "out_rows": out_rows,
    }))


if __name__ == "__main__":
    main()
