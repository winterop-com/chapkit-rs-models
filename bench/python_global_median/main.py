"""chap-core's global_median baseline served by Python chapkit, for benchmarking against chapkit-rs.

Same logic as chap_core/models/builtin/global_median.py and the Rust port: per-location quantile
samples of all observed disease_cases; unseen locations get the median per quantile level.

    ../../chapkit/.venv/bin/python -m uvicorn main:app --port 8101   (from this directory)
"""

import numpy as np
import pandas as pd
from geojson_pydantic import FeatureCollection

from chapkit import BaseConfig
from chapkit.api import MLServiceBuilder, MLServiceInfo, ModelMetadata, PeriodType
from chapkit.artifact import ArtifactHierarchy
from chapkit.data import DataFrame
from chapkit.ml import FunctionalModelRunner

N_SAMPLES = 100
LEVELS = np.sort(np.append((np.arange(N_SAMPLES - 1) + 0.5) / (N_SAMPLES - 1), 0.5))


class GlobalMedianConfig(BaseConfig):
    """No user options."""


async def on_train(config: GlobalMedianConfig, data: DataFrame, geo: FeatureCollection | None = None) -> dict:
    df = data.to_pandas()
    by_location = {}
    for loc, group in df.groupby("location", sort=False):
        observed = pd.to_numeric(group["disease_cases"], errors="coerce").to_numpy(dtype=float)
        observed = observed[np.isfinite(observed)]
        if len(observed):
            by_location[str(loc)] = np.quantile(observed, LEVELS)
    fallback = np.median(np.stack(list(by_location.values())), axis=0)
    return {"by_location": by_location, "fallback": fallback}


async def on_predict(
    config: GlobalMedianConfig, model: dict, historic: DataFrame, future: DataFrame, geo: FeatureCollection | None = None
) -> DataFrame:
    fut = future.to_pandas()
    rows = np.stack([model["by_location"].get(str(loc), model["fallback"]) for loc in fut["location"]])
    out = fut[["time_period", "location"]].copy()
    samples = pd.DataFrame(rows, columns=[f"sample_{i}" for i in range(N_SAMPLES)], index=out.index)
    return DataFrame.from_pandas(pd.concat([out, samples], axis=1))


info = MLServiceInfo(
    id="global-median-py",
    display_name="Global median (Python chapkit)",
    version="1.0.0",
    model_metadata=ModelMetadata(author="Morten Hansen", contact_email="morten@winterop.com"),
    period_type=PeriodType.any,
)

app = MLServiceBuilder(
    info=info,
    config_schema=GlobalMedianConfig,
    hierarchy=ArtifactHierarchy(name="ml_pipeline", level_labels={0: "ml_training_workspace", 1: "ml_prediction"}),
    runner=FunctionalModelRunner(on_train=on_train, on_predict=on_predict),
).build()
