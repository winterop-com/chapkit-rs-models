"""Generate golden outputs for the configurable baseline (chap-models/configurable_baseline_model).

Run with chap-core's environment (2.7.0), from the chapkit-rs-models root:

    ../../chap-core/.venv/bin/python scripts/gen_configurable_fixtures.py

The model is chap-core's seasonal shift persistence with its samples adjusted in log(1 + cases)
space. This script runs chap-core's built-in `seasonal_shift_persistence` on the data of the
baseline fixtures (crates/baseline-core/fixtures, made by gen_baseline_fixtures.py) and applies
`adjust` from the model's predict.py, copied below operation for operation, for the option sets
in OPTIONS. Writes crates/baseline-configurable/fixtures/configurable_<case>.json in the same
format as the baseline fixtures, with one prediction set per option set, keyed by its label.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from chap_core.models.builtin import get_builtin_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "crates/baseline-core/fixtures"
OUT = Path(os.environ.get("FIXTURES_OUT") or ROOT / "crates/baseline-configurable/fixtures")

# label -> (spread_factor, log_bias): the defaults, the model's chap_seed.yaml configurations,
# and both options together.
OPTIONS = {
    "default": (1.0, 0.0),
    "biased_over": (1.0, 0.3),
    "biased_under": (1.0, -0.3),
    "point": (0.0, 0.0),
    "narrow": (0.5, 0.0),
    "wide": (2.0, 0.0),
    "wide_biased": (2.0, 0.3),
}


def adjust(samples: np.ndarray, spread_factor: float, log_bias: float) -> np.ndarray:
    """predict.py's `adjust`, with the config fields passed directly."""
    log_samples = np.log1p(samples)
    median = np.median(log_samples, axis=1, keepdims=True)
    adjusted = median + spread_factor * (log_samples - median) + log_bias
    return np.maximum(np.expm1(adjusted), 0.0)


def frame(wire: dict) -> pd.DataFrame:
    df = pd.DataFrame(wire["data"], columns=wire["columns"])
    if "disease_cases" in df:
        df["disease_cases"] = df["disease_cases"].astype(float)
    return df


def run(case: str) -> None:
    fx = json.loads((SOURCE / f"baselines_{case}.json").read_text())
    train, historic, future = (frame(fx[k]) for k in ("train", "historic", "future"))
    model = get_builtin_model("seasonal_shift_persistence")()
    model.train(DataSet.from_pandas(train))
    pred = model.predict(DataSet.from_pandas(historic), DataSet.from_pandas(future))
    out = {"train": fx["train"], "historic": fx["historic"], "future": fx["future"], "predictions": {}}
    for label, (spread_factor, log_bias) in OPTIONS.items():
        out["predictions"][label] = {
            f"{loc}|{period.id}": [float(v) for v in row]
            for loc, samples in pred.items()
            for period, row in zip(
                samples.time_period, adjust(np.asarray(samples.samples), spread_factor, log_bias), strict=True
            )
        }
    path = OUT / f"configurable_{case}.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for case in ("monthly", "weekly", "single", "fallbacks"):
        run(case)
