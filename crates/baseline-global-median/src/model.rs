//! # Global median: how the forecast is calculated
//!
//! **Forecast.** For a location, every future period gets the same 100 samples: the quantiles
//! of *all* observed `disease_cases` of that location in the training data (see
//! [`quantile_samples`]). The median of the samples is the median of the observations. There
//! is no trend, no seasonality and no fitting.
//!
//! **A location with no observations** (missing values only, or not in the training data)
//! gets, quantile level by quantile level, the median of every other location's samples.
//!
//! Port of chap-core's `global_median` baseline (version 2).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use baseline_core::quantiles::{median_per_level, quantile_samples};

/// Training, for one location: the quantiles of its observed (non-missing) cases.
/// `None` when the location has no observed cases.
pub fn location_samples(cases: &[f64]) -> Option<Vec<f64>> {
    let observed: Vec<f64> = cases.iter().copied().filter(|c| c.is_finite()).collect();
    quantile_samples(&observed)
}

/// The trained model: samples per location, plus the samples for locations without any.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlobalMedianSamples {
    by_location: BTreeMap<String, Vec<f64>>,
    /// Per quantile level, the median over all locations' samples.
    fallback: Option<Vec<f64>>,
}

impl GlobalMedianSamples {
    /// Training, for all locations: keep each location's samples and work out the fallback.
    pub fn new(by_location: BTreeMap<String, Vec<f64>>) -> Self {
        let fallback = median_per_level(by_location.values());
        Self {
            by_location,
            fallback,
        }
    }

    /// Forecast samples for a location: its own, else the fallback. `None` only when no
    /// location in the training data had any observed cases.
    pub fn for_location(&self, location: &str) -> Option<&[f64]> {
        self.by_location
            .get(location)
            .or(self.fallback.as_ref())
            .map(Vec::as_slice)
    }
}
