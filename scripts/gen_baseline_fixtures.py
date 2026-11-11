"""Generate golden outputs from chap-core's built-in baselines.

Run with chap-core's environment, from the chapkit-rs-models root:

    ../../chap-core/.venv/bin/python scripts/gen_baseline_fixtures.py

or inside a chap-core container (the fixtures were last made with chap-core 2.7.0):

    docker cp scripts/gen_baseline_fixtures.py CONTAINER:/tmp/gen.py
    docker exec -e FIXTURES_OUT=/tmp/fixtures CONTAINER python /tmp/gen.py
    docker cp CONTAINER:/tmp/fixtures/. crates/baseline-core/fixtures/

Writes crates/baseline-core/fixtures/baselines_<case>.json with the
train/historic/future frames (chapkit {columns, data} format) and the samples
each baseline predicts, so the Rust ports can be checked number for number.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from chap_core.models.builtin import get_builtin_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

OUT = Path(os.environ.get("FIXTURES_OUT") or Path(__file__).resolve().parent.parent / "crates/baseline-core/fixtures")
MODELS = ["global_median", "persistence", "seasonal_median", "seasonal_shift_persistence"]


def monthly(i: int) -> str:
    return f"{2018 + i // 12}-{i % 12 + 1:02d}"


def weekly(i: int) -> str:
    # 2020 has an ISO week 53; start in 2019W40 so the series crosses it.
    year, week = 2019, 40 + i
    while True:
        weeks = 53 if year in (2015, 2020, 2026) else 52
        if week <= weeks:
            return f"{year}W{week:02d}"
        week -= weeks
        year += 1


def frames(period, n_train: int, horizon: int, rng: np.random.Generator):
    rows = []
    for li, loc in enumerate(["A", "B", "C"]):
        for i in range(n_train + horizon):
            season = math.sin(2 * math.pi * i / 12)
            cases = float(rng.poisson(max(0.5, 20 + 10 * season + 5 * li)))
            # holes: B misses some values, C's last historic value is missing
            if loc == "B" and i % 7 == 3:
                cases = float("nan")
            if loc == "C" and i == n_train - 1:
                cases = float("nan")
            rows.append({"time_period": period(i), "location": loc, "disease_cases": cases, "rainfall": 100 + 30 * season})
    df = pd.DataFrame(rows)
    train = df[df.time_period.isin([period(i) for i in range(n_train)])]
    future = df[df.time_period.isin([period(i) for i in range(n_train, n_train + horizon)])].drop(columns="disease_cases")
    return train.reset_index(drop=True), future.reset_index(drop=True)


def wire(df: pd.DataFrame) -> dict:
    data = [[None if isinstance(v, float) and math.isnan(v) else v for v in row] for row in df.itertuples(index=False)]
    return {"columns": list(df.columns), "data": data}


def run(case: str, period, n_train: int, horizon: int) -> None:
    rng = np.random.default_rng(42)
    train, future = frames(period, n_train, horizon, rng)
    # an extra location "D" only in the future, to exercise the fallbacks
    extra = future[future.location == "A"].copy()
    extra["location"] = "D"
    future_all = pd.concat([future, extra], ignore_index=True)
    historic_extra = train[train.location == "A"].copy()
    historic_extra["location"] = "D"
    historic = pd.concat([train, historic_extra], ignore_index=True)
    write(case, train, historic, future_all)


def write(case: str, train: pd.DataFrame, historic: pd.DataFrame, future: pd.DataFrame) -> None:
    out = {"train": wire(train), "historic": wire(historic), "future": wire(future), "predictions": {}}
    for name in MODELS:
        model = get_builtin_model(name)()
        model.train(DataSet.from_pandas(train))
        pred = model.predict(DataSet.from_pandas(historic), DataSet.from_pandas(future))
        out["predictions"][name] = {
            f"{loc}|{period.id}": [float(v) for v in row]
            for loc, samples in pred.items()
            for period, row in zip(samples.time_period, samples.samples, strict=True)
        }
    path = OUT / f"baselines_{case}.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path} ({len(future)} future rows)")


def table(periods: list[str], series: dict[str, list[float] | None]) -> pd.DataFrame:
    """One row per location and period; `None` series leave out disease_cases (future data)."""
    rows = [
        {"time_period": p, "location": loc}
        | ({} if values is None else {"disease_cases": float(values[i])})
        for loc, values in series.items()
        for i, p in enumerate(periods)
    ]
    return pd.DataFrame(rows)


def short() -> None:
    """Short series, so ratios run out and the fallbacks for missing ratios decide the forecast.

    `single`: one location, January and February [1, 3], forecast 2 months: the second month has
    no ratio at horizon 2 and falls back to the single ratio at horizon 1.
    `fallbacks`: four months and four locations with holes, forecast 4 months, so horizons run
    past those trained, and own single ratios meet pooled ones.
    """
    nan = float("nan")
    train = table(["2018-01", "2018-02"], {"A": [1, 3]})
    write("single", train, train, table(["2018-03", "2018-04"], {"A": None}))
    months = ["2018-01", "2018-02", "2018-03", "2018-04"]
    train = table(
        months,
        {"A": [nan, nan, 1, 3], "B": [5, 2, 8, 4], "C": [1, nan, 2, nan], "E": [nan, nan, nan, nan]},
    )
    historic = table(
        months,
        {"A": [nan, nan, 1, 3], "B": [5, 2, 8, 4], "C": [1, nan, 2, nan], "E": [nan, 6, nan, nan]},
    )
    future = table(["2018-05", "2018-06", "2018-07", "2018-08"], dict.fromkeys("ABCE"))
    write("fallbacks", train, historic, future)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    run("monthly", monthly, 40, 6)
    run("weekly", weekly, 80, 8)
    short()
