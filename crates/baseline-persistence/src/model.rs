//! # Persistence: how the forecast is calculated
//!
//! **Idea.** Forecast the last observed value `y`, with a spread taken from how much cases
//! changed over the same distance in the training data.
//!
//! **Log differences (training).** For a location's series `y_1, y_2, ...` and a horizon `h`
//! (1 up to one year: 12 months or 52 weeks), every pair of periods `h` apart gives
//!
//! ```text
//!     d = ln(y_t + 1) - ln(y_{t-h} + 1)
//! ```
//!
//! the change on a log scale. Adding 1 keeps zero counts usable; a missing value on either
//! side gives no difference.
//!
//! **Forecast.** For a future period `h` periods after the last observed value `y`, take the
//! location's differences for horizon `h`, subtract their median (centring), turn them into
//! 100 quantile samples `c_i` (see [`quantile_samples`]), and forecast
//!
//! ```text
//!     sample_i = max((y + 1) * exp(c_i) - 1, 0)
//! ```
//!
//! Centring puts the median sample at exactly `y`: a trend in the training data does not move
//! the forecast, only the spread comes from the data, and it widens with the horizon.
//!
//! **When differences are missing**, horizon by horizon from `h` down to 1:
//!
//! 1. The location's own differences for the horizon, if there are at least two (one
//!    difference gives no spread once centred).
//! 2. Else all locations' differences for the horizon pooled together, if at least two.
//! 3. Else the next shorter horizon (a horizon longer than the training data, or than a year,
//!    has none).
//! 4. Nothing anywhere: the last value is forecast unchanged.
//!
//! A location without any observed value in the historic data has nothing to persist and
//! cannot be forecast.
//!
//! Port of chap-core's `persistence` baseline (version 2, chap-core 2.7.0).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError};

use serde::{Deserialize, Serialize};

use baseline_core::quantiles::{median, quantile_samples};

/// Training, for one location: `ln(y + 1)` of every period, in time order (NaN where missing).
/// The differences for a horizon are taken from these when a forecast needs them.
pub fn log_cases(cases: &[f64]) -> Vec<f64> {
    cases.iter().map(|y| (y + 1.0).ln()).collect()
}

/// The differences `log[t] - log[t-h]` for every `t`, leaving out missing ones.
pub fn differences(log: &[f64], h: usize) -> Vec<f64> {
    log.iter()
        .skip(h)
        .zip(log)
        .map(|(now, before)| now - before)
        .filter(|d| d.is_finite())
        .collect()
}

/// The differences minus their median. `None` with fewer than two (no spread).
pub fn centred(mut differences: Vec<f64>) -> Option<Vec<f64>> {
    if differences.len() < 2 {
        return None;
    }
    // `median` sorts the differences; their order does not matter from here on.
    let m = median(&mut differences);
    for d in &mut differences {
        *d -= m;
    }
    Some(differences)
}

/// The trained model: each location's log cases, and the longest horizon (one year).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PersistenceTrained {
    #[serde(with = "missing_as_null")]
    by_location: BTreeMap<String, Vec<f64>>,
    max_horizon: usize,
    /// Quantile samples of all locations' centred differences (fallback 2), by horizon. Not
    /// stored: built the first time a forecast falls back to a horizon.
    #[serde(skip)]
    pooled: Mutex<BTreeMap<usize, Arc<Option<Vec<f64>>>>>,
}

impl Clone for PersistenceTrained {
    fn clone(&self) -> Self {
        Self::new(self.by_location.clone(), self.max_horizon)
    }
}

impl PersistenceTrained {
    /// The trained model, from each location's log cases and the longest horizon.
    pub fn new(by_location: BTreeMap<String, Vec<f64>>, max_horizon: usize) -> Self {
        Self {
            by_location,
            max_horizon,
            pooled: Mutex::default(),
        }
    }

    /// Quantile samples of one location's centred differences for horizon `h`.
    fn own(&self, location: &str, h: usize) -> Option<Vec<f64>> {
        let log = self.by_location.get(location)?;
        quantile_samples(&centred(differences(log, h))?)
    }

    /// Quantile samples of all locations' differences for horizon `h`, pooled, then centred.
    fn pooled(&self, h: usize) -> Arc<Option<Vec<f64>>> {
        let mut cache = self.pooled.lock().unwrap_or_else(PoisonError::into_inner);
        cache
            .entry(h)
            .or_insert_with(|| {
                let all: Vec<f64> = self
                    .by_location
                    .values()
                    .flat_map(|log| differences(log, h))
                    .collect();
                Arc::new(centred(all).and_then(|c| quantile_samples(&c)))
            })
            .clone()
    }

    /// The 100 centred quantile samples to forecast with, following the fallbacks above;
    /// all zeros when there is nothing (the last value unchanged).
    pub fn spread(&self, location: &str, horizon: usize) -> Vec<f64> {
        // Horizons beyond a year were never kept; start at the longest one that was.
        for h in (1..=horizon.min(self.max_horizon)).rev() {
            if let Some(samples) = self.own(location, h) {
                return samples;
            }
            if let Some(samples) = self.pooled(h).as_ref() {
                return samples.clone();
            }
        }
        quantile_samples(&[0.0]).unwrap_or_default()
    }
}

/// JSON has no NaN: store a missing log case as `null`.
mod missing_as_null {
    use std::collections::BTreeMap;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    type Logs = BTreeMap<String, Vec<f64>>;

    pub fn serialize<S: Serializer>(logs: &Logs, s: S) -> Result<S::Ok, S::Error> {
        let stored: BTreeMap<&String, Vec<Option<f64>>> = logs
            .iter()
            .map(|(loc, log)| {
                (
                    loc,
                    log.iter().map(|v| v.is_finite().then_some(*v)).collect(),
                )
            })
            .collect();
        stored.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Logs, D::Error> {
        let stored = BTreeMap::<String, Vec<Option<f64>>>::deserialize(d)?;
        Ok(stored
            .into_iter()
            .map(|(loc, log)| {
                (
                    loc,
                    log.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                )
            })
            .collect())
    }
}

/// The last observed (non-missing) value of a series, and how many periods before the end of
/// the series it is (1 when it is the last period). `None` when nothing was observed.
pub fn last_observed(cases: &[f64]) -> Option<(f64, usize)> {
    let at = cases.iter().rposition(|c| c.is_finite())?;
    Some((*cases.get(at)?, cases.len() - at))
}

/// The forecast samples: the last value moved by each centred difference, never below zero.
pub fn forecast(last_value: f64, spread: &[f64]) -> Vec<f64> {
    spread
        .iter()
        .map(|c| ((last_value + 1.0) * c.exp() - 1.0).max(0.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn differences_skip_missing_values() {
        let log = log_cases(&[0.0, f64::NAN, 3.0, 7.0]);
        // h = 1: (nan - 0), (3 - nan) are missing; ln 8 - ln 4 = ln 2.
        let d = differences(&log, 1);
        assert_eq!(d.len(), 1);
        assert!((d[0] - 2f64.ln()).abs() < 1e-15);
    }

    #[test]
    fn centring_keeps_the_median_at_the_last_value() {
        let trained = PersistenceTrained::new(
            BTreeMap::from([("a".into(), log_cases(&[1.0, 2.0, 4.0, 8.0, 16.0]))]),
            12,
        );
        let samples = forecast(10.0, &trained.spread("a", 1));
        let mid = f64::midpoint(samples[49], samples[50]);
        assert!((mid - 10.0).abs() < 1e-12, "median {mid}");
    }

    #[test]
    fn nothing_to_go_on_persists_the_last_value() {
        let trained =
            PersistenceTrained::new(BTreeMap::from([("a".into(), log_cases(&[5.0]))]), 12);
        assert!(
            forecast(3.0, &trained.spread("a", 2))
                .iter()
                .all(|&s| (s - 3.0).abs() < 1e-12)
        );
    }
}
