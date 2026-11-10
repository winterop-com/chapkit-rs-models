//! chap-core's `global_median` baseline as a chapkit [`Model`].
//!
//! **The calculation is in [`model`]** (`src/model.rs`). This file only connects it to chapkit:
//! splitting the table per location, running locations in parallel, building the prediction table.

pub mod model;

use std::collections::BTreeMap;

use baseline_core::N_SAMPLES;
use chapkit::locations::{predict_by_location, values_by_location};
use chapkit::{DataFrame, Error, Model, ModelInfo, NoConfig, PredictInput, Result, TrainInput};
use rayon::prelude::*;

pub use model::GlobalMedianSamples;

/// Forecasts every period with all observations of the same location in the training data.
#[derive(Debug, Clone, Copy, Default)]
pub struct GlobalMedian;

/// Train on a table: the samples of each location, in parallel. The seasonal median baseline
/// uses this as its last fallback.
pub fn train(data: &DataFrame, target: &str) -> Result<GlobalMedianSamples> {
    let by_location: BTreeMap<String, Vec<f64>> = values_by_location(data, target)?
        .into_par_iter()
        .filter_map(|(loc, cases)| model::location_samples(&cases).map(|s| (loc, s)))
        .collect();
    Ok(GlobalMedianSamples::new(by_location))
}

/// The error when no location in the training data had any observed cases.
pub fn nothing_observed() -> Error {
    Error::invalid("no observed disease_cases for any location in the training data")
}

impl Model for GlobalMedian {
    type Config = NoConfig;
    type Trained = GlobalMedianSamples;

    fn info(&self) -> ModelInfo {
        baseline_core::info(
            "Baseline that uses all observed values for the same location as forecast samples for every period, \
             without smoothing or model fitting.",
        )
    }

    fn train(&self, _: &NoConfig, input: TrainInput<'_>) -> Result<GlobalMedianSamples> {
        train(input.data, &self.info().target)
    }

    fn predict(
        &self,
        trained: &GlobalMedianSamples,
        _: &NoConfig,
        input: PredictInput<'_>,
    ) -> Result<DataFrame> {
        predict_by_location(input.future, N_SAMPLES, |loc, rows| {
            let samples = trained.for_location(loc).ok_or_else(nothing_observed)?;
            Ok(vec![samples.to_vec(); rows.len()])
        })
    }
}
