//! chap-models' `configurable_baseline_model` as a chapkit [`Model`]: chap-core's seasonal shift
//! persistence with a configurable spread and bias in `log(1 + cases)` space.
//!
//! **The calculation is in [`model`]** (`src/model.rs`); the forecast it adjusts is the
//! `baseline-seasonal-shift-persistence` crate's. This file only connects them to chapkit: the
//! options, the search space for HPO, and the prediction table.

pub mod model;

use baseline_core::N_SAMPLES;
use baseline_seasonal_shift_persistence::{
    SeasonalShiftPersistence, SeasonalShiftTrained, forecast_location,
};
use chapkit::locations::{predict_by_location, series_by_location};
use chapkit::{
    AssessedStatus, DataFrame, Diagnostic, Error, Model, ModelInfo, ModelMetadata, NoConfig, Param,
    PredictInput, Result, SearchSpace, TrainInput,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The model's options, as in the original's `ModelConfig`. Unknown fields (chapkit's reserved
/// keys such as `prediction_periods`) are ignored.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Options {
    /// Multiplies the distance of each sample from the median in log(1 + cases) space. 1 keeps the
    /// seasonal shift persistence samples, 0 gives a point forecast at the median, below 1
    /// narrows and above 1 widens the forecast.
    #[serde(default = "one")]
    #[schemars(range(min = 0.0))]
    pub spread_factor: f64,
    /// Added to every sample in log(1 + cases) space. 0.1 scales forecasts by about 1.1.
    #[serde(default)]
    pub log_bias: f64,
}

fn one() -> f64 {
    1.0
}

impl Default for Options {
    fn default() -> Self {
        Self {
            spread_factor: 1.0,
            log_bias: 0.0,
        }
    }
}

impl Options {
    /// Why the options cannot be used, if they cannot.
    fn problem(&self) -> Option<String> {
        if !(self.spread_factor.is_finite() && self.spread_factor >= 0.0) {
            return Some(format!(
                "spread_factor must be a number of at least 0, got {}",
                self.spread_factor
            ));
        }
        if !self.log_bias.is_finite() {
            return Some(format!("log_bias must be a number, got {}", self.log_bias));
        }
        None
    }

    fn diagnostics(&self) -> Vec<Diagnostic> {
        self.problem()
            .map(|p| Diagnostic::error("invalid_option", p))
            .into_iter()
            .collect()
    }

    fn check(&self) -> Result<()> {
        self.problem().map_or(Ok(()), |p| Err(Error::invalid(p)))
    }
}

/// Seasonal shift persistence, its samples adjusted by [`Options`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ConfigurableBaseline;

impl Model for ConfigurableBaseline {
    type Config = Options;
    type Trained = SeasonalShiftTrained;

    fn info(&self) -> ModelInfo {
        ModelInfo {
            model_metadata: ModelMetadata {
                author: Some("Morten Hansen".into()),
                author_note: Some(
                    "chapkit-rs port of chap-models' configurable_baseline_model (Knut Rand): \
                     chap-core's seasonal shift persistence, with the spread around the median \
                     scaled and a bias added in log(1 + cases) space. With the default options \
                     the samples are seasonal shift persistence's. A yardstick for evaluation \
                     metrics, not a forecasting model."
                        .into(),
                ),
                author_assessed_status: Some(AssessedStatus::Red),
                contact_email: Some("morten@winterop.com".into()),
                repository_url: Some("https://github.com/winterop-com/chapkit-rs-models".into()),
                documentation_url: Some(
                    "https://winterop-com.github.io/chapkit-rs-models/models/configurable-baseline.html"
                        .into(),
                ),
                ..ModelMetadata::default()
            },
            ..ModelInfo::default()
        }
    }

    /// Spread from a point forecast to three times seasonal shift persistence's; bias up to a
    /// factor of about 1.65 either way on `1 + cases`.
    fn search_space(&self) -> Option<SearchSpace> {
        let float = |low, high| Param::Float {
            low,
            high,
            step: None,
            log: false,
        };
        Some(
            SearchSpace::new()
                .with("spread_factor", float(0.0, 3.0))
                .with("log_bias", float(-0.5, 0.5)),
        )
    }

    fn validate_train(&self, config: &Options, _: TrainInput<'_>) -> Vec<Diagnostic> {
        config.diagnostics()
    }

    fn validate_predict(&self, config: &Options, _: PredictInput<'_>) -> Vec<Diagnostic> {
        config.diagnostics()
    }

    /// Training is seasonal shift persistence's: the options only act on the forecast.
    fn train(&self, config: &Options, input: TrainInput<'_>) -> Result<SeasonalShiftTrained> {
        config.check()?;
        SeasonalShiftPersistence.train(&NoConfig {}, input)
    }

    fn predict(
        &self,
        trained: &SeasonalShiftTrained,
        config: &Options,
        input: PredictInput<'_>,
    ) -> Result<DataFrame> {
        config.check()?;
        let historic = series_by_location(input.historic, &self.info().target)?;
        predict_by_location(input.future, N_SAMPLES, |loc, rows| {
            let cases = historic.get(loc).map(|s| s.values.as_slice());
            let mut samples = forecast_location(trained, loc, cases.unwrap_or_default(), rows)?;
            for period in &mut samples {
                model::adjust(period, config.spread_factor, config.log_bias);
            }
            Ok(samples)
        })
    }
}
