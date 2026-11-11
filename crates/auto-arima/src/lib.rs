//! chap-models' `auto_arima_chapkit` as a chapkit [`Model`].
//!
//! **The calculation is in [`model`]** (`src/model.rs`); the ARIMA itself is chapkit's
//! [`chapkit::ts`]. This file connects them to chapkit: the configuration, reading each location's
//! series onto a gap-free period grid, the seasonal period of the data, and seeding.

pub mod model;

use std::collections::BTreeMap;

use chapkit::locations::{Series, predict_by_location, series_by_location};
use chapkit::rng::Rng;
use chapkit::{
    AssessedStatus, Error, Model, ModelInfo, ModelMetadata, Period, PeriodType, PredictInput,
    Result, TrainInput,
};
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use model::Fitted;

/// The model's options. The original has none besides chapkit's `prediction_periods`; `n_samples`
/// is its fixed 100 and `seed` makes the sampling repeatable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Config {
    /// Number of samples drawn per forecast period.
    #[serde(default = "default_n_samples")]
    pub n_samples: usize,
    /// Seed for the sampling; random when not set, as in the original.
    #[serde(default)]
    pub seed: Option<u64>,
}

fn default_n_samples() -> usize {
    100
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_samples: default_n_samples(),
            seed: None,
        }
    }
}

impl Config {
    fn check(&self) -> Result<()> {
        if self.n_samples == 0 {
            return Err(Error::Config("n_samples must be at least 1".into()));
        }
        Ok(())
    }

    /// The seed of this call: the configured one, else a fresh one.
    fn seed(&self) -> u64 {
        self.seed.unwrap_or_else(|| Rng::from_time().next_u64())
    }
}

/// The trained model: one fitted ARIMA per location.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trained {
    /// Periods per seasonal cycle of the training data: 12 (monthly) or 52 (weekly).
    pub season: usize,
    /// Each location's model, by location.
    pub locations: BTreeMap<String, Fitted>,
}

/// The model.
#[derive(Debug, Clone, Copy, Default)]
pub struct AutoArima;

impl Model for AutoArima {
    type Config = Config;
    type Trained = Trained;

    fn info(&self) -> ModelInfo {
        ModelInfo {
            period_type: PeriodType::Any,
            min_prediction_periods: 0,
            max_prediction_periods: 12,
            allow_free_additional_continuous_covariates: false,
            required_covariates: Vec::new(),
            model_metadata: ModelMetadata {
                author: Some("Morten Hansen".into()),
                author_note: Some(
                    "chapkit-rs port of chap-models' auto_arima_chapkit (Harsha, Cardiff \
                     University; adapted to chapkit by Halvard): per location, an ARIMA with its \
                     orders chosen as fable's ARIMA() chooses them, its state updated with the \
                     latest history, and normal samples from its forecast. Monthly and weekly, \
                     where the original is monthly only."
                        .into(),
                ),
                author_assessed_status: Some(AssessedStatus::Red),
                contact_email: Some("morten@winterop.com".into()),
                repository_url: Some("https://github.com/winterop-com/chapkit-rs-models".into()),
                documentation_url: Some(
                    "https://winterop-com.github.io/chapkit-rs-models/models/auto-arima.html"
                        .into(),
                ),
                ..ModelMetadata::default()
            },
            ..ModelInfo::default()
        }
    }

    fn train(&self, config: &Config, input: TrainInput<'_>) -> Result<Trained> {
        config.check()?;
        let target = self.info().target;
        let series = series_by_location(input.data, &target)?;
        let season = season_of(&series)?;
        let locations = series
            .par_iter()
            .map(|(location, s)| {
                let values = on_grid(location, s)?;
                let fitted = model::fit(&values, season).map_err(|e| {
                    Error::model(format!("no ARIMA for location '{location}': {e}"))
                })?;
                Ok((location.clone(), fitted))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Trained { season, locations })
    }

    fn predict(
        &self,
        trained: &Trained,
        config: &Config,
        input: PredictInput<'_>,
    ) -> Result<chapkit::DataFrame> {
        config.check()?;
        let target = self.info().target;
        let history = series_by_location(input.historic, &target)?;
        let seed = config.seed();
        predict_by_location(input.future, config.n_samples, |location, rows| {
            let fitted = trained.locations.get(location).ok_or_else(|| {
                Error::invalid(format!(
                    "location '{location}' was not in the training data"
                ))
            })?;
            let s = history.get(location).ok_or_else(|| {
                Error::invalid(format!("location '{location}' has no historic data"))
            })?;
            let values = on_grid(location, s)?;
            let last = *s.periods.last().ok_or_else(|| {
                Error::invalid(format!("location '{location}' has no historic data"))
            })?;
            let steps = rows
                .iter()
                .map(|&(_, period)| steps_after(last, period))
                .collect::<Result<Vec<_>>>()?;
            let horizon = steps.iter().copied().max().unwrap_or(0);
            let forecast = model::forecast(fitted, &values, horizon).map_err(|e| {
                Error::model(format!("forecast for location '{location}' failed: {e}"))
            })?;
            let mut rng = Rng::new(seed ^ fnv1a(location));
            steps
                .iter()
                .map(|&h| {
                    let (mean, sd) = model::step(&forecast, h).ok_or_else(|| {
                        Error::model(format!("no forecast {h} periods ahead for '{location}'"))
                    })?;
                    Ok(model::samples(mean, sd, config.n_samples, &mut rng))
                })
                .collect()
        })
    }
}

/// The seasonal period of the data: 12 for months, 52 for weeks (chap-core's count; a 53rd ISO
/// week is one more step of the series, not a season of its own).
fn season_of(series: &BTreeMap<String, Series>) -> Result<usize> {
    let mut kinds = series
        .values()
        .flat_map(|s| s.periods.iter().map(|p| p.periods_per_year()));
    let first = kinds
        .next()
        .ok_or_else(|| Error::invalid("no training data"))?;
    if kinds.any(|k| k != first) {
        return Err(Error::invalid(
            "training data mixes monthly and weekly periods",
        ));
    }
    Ok(first as usize)
}

/// A location's values on the gap-free grid from its first to its last period: a period missing
/// from the table counts as missing (NaN), as chap-core fills it, so the ARIMA steps over it.
fn on_grid(location: &str, s: &Series) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(s.periods.len());
    let mut expected: Option<Period> = None;
    for (&period, &value) in s.periods.iter().zip(&s.values) {
        while let Some(next) = expected.filter(|e| *e < period) {
            out.push(f64::NAN);
            expected = next.succ();
        }
        if expected.is_some_and(|e| e != period) {
            return Err(Error::invalid(format!(
                "location '{location}': period {period} is not on the series' grid"
            )));
        }
        out.push(value);
        expected = Some(period.succ().ok_or_else(|| {
            Error::invalid(format!(
                "location '{location}': cannot step past {period} (Sunday-start weeks are not supported)"
            ))
        })?);
    }
    Ok(out)
}

/// How many periods `period` is after `last` (1 for the next one).
fn steps_after(last: Period, period: Period) -> Result<usize> {
    let mut at = last;
    for h in 1..=10_000 {
        at = at
            .succ()
            .ok_or_else(|| Error::invalid(format!("cannot step past {at}")))?;
        if at == period {
            return Ok(h);
        }
        if at > period {
            break;
        }
    }
    Err(Error::invalid(format!(
        "future period {period} is not after the last historic period {last}"
    )))
}

/// A 64-bit FNV-1a hash: gives each location its own random stream, whatever the order or
/// thread it is forecast in.
fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn month(year: i32, month: u32) -> Period {
        Period::Month { year, month }
    }

    #[test]
    fn grid_fills_gaps_and_counts_steps() {
        let s = Series {
            periods: vec![month(2020, 11), month(2021, 2)],
            values: vec![1.0, 4.0],
        };
        let v = on_grid("a", &s).unwrap();
        assert_eq!(v.len(), 4);
        assert!(v[1].is_nan() && v[2].is_nan());
        assert_eq!(steps_after(month(2020, 11), month(2021, 2)).unwrap(), 3);
        assert!(steps_after(month(2021, 2), month(2021, 2)).is_err());
    }
}
