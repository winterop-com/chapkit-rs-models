//! chap-core's `persistence` baseline as a chapkit [`Model`].
//!
//! **The calculation is in [`model`]** (`src/model.rs`). This file only connects it to chapkit:
//! splitting the table per location, running locations in parallel, building the prediction table.

pub mod model;

use std::collections::BTreeMap;

use baseline_core::N_SAMPLES;
use chapkit::locations::{predict_by_location, series_by_location};
use chapkit::{DataFrame, Error, Model, ModelInfo, NoConfig, PredictInput, Result, TrainInput};
use rayon::prelude::*;

pub use model::PersistenceTrained;

/// Forecasts the last observed value, with a spread from how cases changed over the same horizon.
#[derive(Debug, Clone, Copy, Default)]
pub struct Persistence;

impl Model for Persistence {
    type Config = NoConfig;
    type Trained = PersistenceTrained;

    fn info(&self) -> ModelInfo {
        baseline_core::info(
            "Baseline that forecasts the last observed value for the same location, with a spread from how \
             cases changed over the same horizon in the training data.",
            "persistence",
        )
    }

    fn train(&self, _: &NoConfig, input: TrainInput<'_>) -> Result<PersistenceTrained> {
        let series = series_by_location(input.data, &self.info().target)?;
        // Horizons up to one year: 12 for monthly, 52 for weekly data.
        let max_horizon = series
            .values()
            .find_map(|s| s.periods.first())
            .ok_or_else(|| Error::invalid("no training data"))?
            .periods_per_year() as usize;
        let by_location: BTreeMap<String, Vec<f64>> = series
            .into_par_iter()
            .map(|(loc, s)| (loc, model::log_cases(&s.values)))
            .collect();
        Ok(PersistenceTrained::new(by_location, max_horizon))
    }

    fn predict(
        &self,
        trained: &PersistenceTrained,
        _: &NoConfig,
        input: PredictInput<'_>,
    ) -> Result<DataFrame> {
        let historic = series_by_location(input.historic, &self.info().target)?;
        predict_by_location(input.future, N_SAMPLES, |loc, rows| {
            let cases = historic
                .get(loc)
                .map(|s| s.values.as_slice())
                .unwrap_or_default();
            let (last_value, since_last) = model::last_observed(cases).ok_or_else(|| {
                Error::invalid(format!(
                    "no observed disease_cases for location '{loc}' in the historic data"
                ))
            })?;
            // The k-th future period is `since_last + k` periods after the last observed value.
            Ok((0..rows.len())
                .map(|k| model::forecast(last_value, &trained.spread(loc, since_last + k)))
                .collect())
        })
    }
}
