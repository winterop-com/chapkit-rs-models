//! chap-core's `seasonal_shift_persistence` baseline as a chapkit [`Model`].
//!
//! **The calculation is in [`model`]** (`src/model.rs`). This file only connects it to chapkit:
//! splitting the table per location, running locations in parallel, building the prediction table.

pub mod model;

use std::collections::BTreeMap;

use baseline_core::N_SAMPLES;
use chapkit::locations::{predict_by_location, series_by_location};
use chapkit::{DataFrame, Error, Model, ModelInfo, NoConfig, PredictInput, Result, TrainInput};
use rayon::prelude::*;

pub use model::{SeasonalShiftTrained, SeasonalShifts};

/// Forecasts the last observed value, scaled by how cases changed over the same horizon and
/// time of year in earlier years.
#[derive(Debug, Clone, Copy, Default)]
pub struct SeasonalShiftPersistence;

impl Model for SeasonalShiftPersistence {
    type Config = NoConfig;
    type Trained = SeasonalShiftTrained;

    fn info(&self) -> ModelInfo {
        baseline_core::info(
            "Baseline that forecasts the last observed value for the same location, scaled by how cases changed \
             over the same horizon and time of year in the training data.",
            "seasonal-shift-persistence",
        )
    }

    fn train(&self, _: &NoConfig, input: TrainInput<'_>) -> Result<SeasonalShiftTrained> {
        let series = series_by_location(input.data, &self.info().target)?;
        // Horizons up to one year: 12 for monthly, 52 for weekly data.
        let max_horizon = series
            .values()
            .find_map(|s| s.periods.first())
            .ok_or_else(|| Error::invalid("no training data"))?
            .periods_per_year() as usize;
        let by_location: BTreeMap<String, SeasonalShifts> = series
            .into_par_iter()
            .map(|(loc, s)| {
                (
                    loc,
                    SeasonalShifts::from_cases(&s.values, &s.seasons(), max_horizon),
                )
            })
            .collect();
        Ok(SeasonalShiftTrained::new(by_location))
    }

    fn predict(
        &self,
        trained: &SeasonalShiftTrained,
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
            rows.iter()
                .enumerate()
                .map(|(k, (_, period))| {
                    let ratios = trained.ratios(loc, period.period_of_year(), since_last + k);
                    model::forecast(last_value, &ratios)
                        .ok_or_else(|| Error::model("no ratios to forecast with"))
                })
                .collect()
        })
    }
}
