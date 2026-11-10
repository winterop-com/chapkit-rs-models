"""Generate the feature-matrix fixture for crates/simple-multistep from the original model.

The original is chap-models/chapkit_simple_multistep_model. Clone it, `uv sync` in it, then run
this script with its environment, from the chapkit-rs-models root:

    ORIGINAL=/path/to/chapkit_simple_multistep_model
    $ORIGINAL/.venv/bin/python scripts/gen_simple_multistep_fixtures.py $ORIGINAL

The random forest and the sampling cannot be matched number for number (sklearn's RNG), but the
features the forest is fitted on and predicts from are deterministic. This script takes a slice of
the original's example_data (3 locations, 40 months of history from the 81st month on, 3 future
months), blanks one
target and one covariate value so the gap filling is exercised, and records:

- `train`: the rows and feature matrix `MultistepModel.fit_multi` passes to the forest;
- `predict`: per location, the target lags the first step starts from and the covariate features
  of each future step, as `on_predict` builds them.

Writes crates/simple-multistep/fixtures/features.json.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ORIGINAL = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ORIGINAL / "src"))

from chapkit_simple_multistep_model.multistep import (  # noqa: E402
    _build_lag_matrix_xr,
    features_to_xarray,
    target_to_xarray,
)
from chapkit_simple_multistep_model.transformations import transform_data  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "crates/simple-multistep/fixtures/features.json"
COVARIATES = ["rainfall", "mean_temperature", "mean_relative_humidity"]
N_LAGS = 6
START, HISTORY, STEPS = 80, 40, 3


def clean(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def frame(df: pd.DataFrame) -> dict:
    """chapkit's {columns, data} wire format."""
    cols = list(df.columns)
    data = [[clean(v) if not isinstance(v, str) else v for v in row] for row in df.itertuples(index=False)]
    return {"columns": cols, "data": data}


data = pd.read_csv(ORIGINAL / "example_data/historic_data.csv")
locations = sorted(data["location"].unique())[:3]
data = data[data["location"].isin(locations)].sort_values(["location", "time_period"])
periods = sorted(data["time_period"].unique())[START:]
historic = data[data["time_period"].isin(periods[:HISTORY])][
    ["time_period", "location", "disease_cases", *COVARIATES]
].reset_index(drop=True)
future = data[data["time_period"].isin(periods[HISTORY : HISTORY + STEPS])][
    ["time_period", "location", *COVARIATES]
].reset_index(drop=True)
# Gaps: a missing target in the second location, a missing covariate in the third.
historic.loc[(historic["location"] == locations[1]) & (historic["time_period"] == periods[20]), "disease_cases"] = (
    np.nan
)
historic.loc[(historic["location"] == locations[2]) & (historic["time_period"] == periods[10]), "rainfall"] = np.nan

# Training features, as train.on_train and MultistepModel.fit_multi build them.
X = transform_data(historic[["time_period", "location", *COVARIATES]])
y = historic[["time_period", "location", "disease_cases"]]
y_xr = target_to_xarray(y)
X_xr = features_to_xarray(X)
lags = _build_lag_matrix_xr(y_xr, N_LAGS).rename(lag="feature")
features = xr.concat(
    [X_xr.isel(time=slice(N_LAGS, None)).transpose("feature", "location", "time"), lags], dim="feature"
)
stacked = features.stack(sample=("location", "time")).transpose("sample", "feature")
y_stacked = y_xr.isel(time=slice(N_LAGS, None)).stack(sample=("location", "time"))
X_np, y_np = stacked.values, y_stacked.values
mask = ~(np.isnan(X_np).any(axis=1) | np.isnan(y_np))
names = [c for c in X.columns if c not in ("time_period", "location")] + [
    f"target_lag{k}" for k in range(N_LAGS, 0, -1)
]
train_rows = [
    {
        "location": str(loc),
        "time_period": pd.Timestamp(t).strftime("%Y-%m"),
        "x": [clean(v) for v in X_np[i]],
        "y": clean(y_np[i]),
    }
    for i, (loc, t) in enumerate(stacked.indexes["sample"])
    if mask[i]
]

# Prediction features, as predict.on_predict and DataFrameMultistepModel.predict build them.
cols = ["time_period", "location", *COVARIATES]
both = pd.concat([historic[cols], future[cols]], ignore_index=True).sort_values(by=["time_period", "location"])
Xp = transform_data(both)
previous_y = target_to_xarray(historic[["time_period", "location", "disease_cases"]]).isel(time=slice(-N_LAGS, None))
Xf = features_to_xarray(Xp).isel(time=slice(-STEPS, None))
predict = {
    str(loc): {
        "previous_y": [clean(v) for v in previous_y.sel(location=loc).values],
        "exog": [[clean(v) for v in row] for row in Xf.sel(location=loc).values],
    }
    for loc in locations
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps(
        {
            "n_target_lags": N_LAGS,
            "covariates": COVARIATES,
            "historic": frame(historic),
            "future": frame(future),
            "feature_names": names,
            "train": train_rows,
            "predict": predict,
        },
        indent=1,
    )
    + "\n"
)
print(f"wrote {OUT}: {len(train_rows)} training rows x {len(names)} features")
