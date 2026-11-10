//! # Seasonal median: how the forecast is calculated
//!
//! **Season.** Every period has a *period of year*: its month (1-12) for monthly data, its week
//! (1-52) for weekly data. Week 53 counts as week 52.
//!
//! **Forecast.** For a location and a future period in season `s`, the 100 samples are the
//! quantiles (see [`quantile_samples`]) of all observed `disease_cases` of that location in the
//! training data whose period is also in season `s`. A forecast for March uses every March seen
//! before. The median of the samples is the median of those observations.
//!
//! **When there is nothing to take quantiles of**, in this order:
//!
//! 1. The location has observations, but none in season `s`: use the location's samples over all
//!    seasons, which is the global median baseline for that location.
//! 2. The location has no observations at all: per quantile level, the median of every other
//!    location's samples for season `s`.
//! 3. No location has observations in season `s` either: the global median fallback (the median
//!    of all locations' samples over all seasons).
//!
//! Port of chap-core's `seasonal_median` baseline (version 2).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use baseline_core::quantiles::{median_per_level, quantile_samples};
use baseline_global_median::GlobalMedianSamples;

/// Training, for one location: for each season with observations, the quantiles of the
/// observed cases in that season. `cases` and `seasons` are aligned, one entry per period.
pub fn season_samples(cases: &[f64], seasons: &[u32]) -> BTreeMap<u32, Vec<f64>> {
    // Collect the observed (non-missing) cases of each season ...
    let mut observed: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    for (&case, &season) in cases.iter().zip(seasons) {
        if case.is_finite() {
            observed.entry(season).or_default().push(case);
        }
    }
    // ... and turn each season's observations into samples.
    observed
        .into_iter()
        .filter_map(|(season, values)| Some((season, quantile_samples(&values)?)))
        .collect()
}

/// The trained model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SeasonalMedianTrained {
    /// Samples per location and season (only seasons with observations).
    by_location: BTreeMap<String, BTreeMap<u32, Vec<f64>>>,
    /// The global median model, for fallbacks 1 and 3.
    global: GlobalMedianSamples,
}

impl SeasonalMedianTrained {
    /// Put together the per-location season samples and the global median model.
    pub fn new(
        by_location: BTreeMap<String, BTreeMap<u32, Vec<f64>>>,
        global: GlobalMedianSamples,
    ) -> Self {
        Self {
            by_location,
            global,
        }
    }

    /// Forecast samples for a location and season, following the fallbacks above. `None` only
    /// when no location in the training data had any observed cases.
    pub fn samples(&self, location: &str, season: u32) -> Option<Vec<f64>> {
        let own = self.by_location.get(location);
        // The location's own observations in this season.
        if let Some(samples) = own.and_then(|seasons| seasons.get(&season)) {
            return Some(samples.clone());
        }
        // Fallback 2: a location without any observations borrows from the others.
        let observed_anything = own.is_some_and(|seasons| !seasons.is_empty());
        if !observed_anything && let Some(samples) = self.other_locations(season) {
            return Some(samples);
        }
        // Fallbacks 1 and 3: the global median of the location, else of all locations.
        self.global.for_location(location).map(<[f64]>::to_vec)
    }

    /// Per quantile level, the median of every location's samples for the season.
    fn other_locations(&self, season: u32) -> Option<Vec<f64>> {
        median_per_level(
            self.by_location
                .values()
                .filter_map(|seasons| seasons.get(&season)),
        )
    }
}
