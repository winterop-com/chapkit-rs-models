//! chap-core's `seasonal_median` baseline as a chapkit [`Model`].
//!
//! **The calculation is in [`model`]** (`src/model.rs`). This file only connects it to chapkit:
//! splitting the table per location, running locations in parallel, building the prediction table.

pub mod model;

use baseline_core::N_SAMPLES;
use chapkit::locations::{predict_by_location, series_by_location};
use chapkit::{DataFrame, Model, ModelInfo, NoConfig, PredictInput, Result, TrainInput};
use rayon::prelude::*;

pub use model::SeasonalMedianTrained;

/// Forecasts each period with all observations of the same location and period of year.
#[derive(Debug, Clone, Copy, Default)]
pub struct SeasonalMedian;

impl Model for SeasonalMedian {
    type Config = NoConfig;
    type Trained = SeasonalMedianTrained;

    fn info(&self) -> ModelInfo {
        baseline_core::info(
            "Baseline that uses all observed values for the same location and month or week of the year as \
             forecast samples, without smoothing or model fitting.",
            "seasonal-median",
        )
    }

    fn train(&self, _: &NoConfig, input: TrainInput<'_>) -> Result<SeasonalMedianTrained> {
        let target = self.info().target;
        let by_location = series_by_location(input.data, &target)?
            .into_par_iter()
            .map(|(loc, series)| {
                (
                    loc,
                    model::season_samples(&series.values, &series.seasons()),
                )
            })
            .collect();
        let global = baseline_global_median::train(input.data, &target)?;
        Ok(SeasonalMedianTrained::new(by_location, global))
    }

    fn predict(
        &self,
        trained: &SeasonalMedianTrained,
        _: &NoConfig,
        input: PredictInput<'_>,
    ) -> Result<DataFrame> {
        predict_by_location(input.future, N_SAMPLES, |loc, rows| {
            rows.iter()
                .map(|(_, period)| {
                    trained
                        .samples(loc, period.period_of_year())
                        .ok_or_else(baseline_global_median::nothing_observed)
                })
                .collect()
        })
    }
}
