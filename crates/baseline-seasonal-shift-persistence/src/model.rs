//! # Seasonal shift persistence: how the forecast is calculated
//!
//! **Idea.** Start from the last observed value `y` and scale it by how cases changed over
//! the same distance and time of year in earlier years.
//!
//! **Ratios (training).** For a location's series `y_1, y_2, ...` and a horizon `h` (1 up to one
//! year: 12 months or 52 weeks), every pair of periods `h` apart gives a ratio
//!
//! ```text
//!     r = (y_t + 1) / (y_{t-h} + 1)
//! ```
//!
//! filed under the period of year (season) of `t`. Adding 1 keeps zero counts usable.
//!
//! **Forecast.** For a future period `h` periods after the last observed value `y`, in season
//! `s`, take the location's ratios for horizon `h` and season `s`, turn them into 100 quantile
//! samples `r_i` (see [`quantile_samples`]), and forecast
//!
//! ```text
//!     sample_i = max((y + 1) * r_i - 1, 0)
//! ```
//!
//! The median is `y` scaled by the median ratio; the spread widens with the horizon.
//!
//! **When ratios are missing**, in this order:
//!
//! 1. No ratios for season `s`: use the ratios of the nearest seasons before and after `s` in the
//!    year together; if there are not both, the ratios of every season (see [`SeasonalShifts::ratios`]).
//! 2. The location has none for the horizon: all locations' ratios pooled together.
//! 3. The horizon has none at all (longer than the training data, or than a year): the longest
//!    shorter horizon that has them.
//! 4. A single ratio gives no spread, so a later candidate with more than one ratio is preferred.
//! 5. No ratio anywhere: the last value is forecast unchanged (ratio 1).
//!
//! A location without any observed value in the historic data has nothing to persist and
//! cannot be forecast.
//!
//! Port of chap-core's `seasonal_shift_persistence` baseline (version 1).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError};

use serde::{Deserialize, Serialize};

use baseline_core::quantiles::quantile_samples;

/// One horizon's ratios, by the season (month or week of the year) they end in.
pub type BySeason = BTreeMap<u32, Vec<f64>>;

/// Ratios `(y_t + 1) / (y_{t-h} + 1)` by horizon `h`, then by the season of `t`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SeasonalShifts {
    by_horizon: BTreeMap<usize, BySeason>,
}

impl SeasonalShifts {
    /// Training, for one location: all ratios for horizons 1 to `max_horizon`.
    /// `cases` and `seasons` are aligned, one entry per period in time order.
    pub fn from_cases(cases: &[f64], seasons: &[u32], max_horizon: usize) -> Self {
        let mut by_horizon = BTreeMap::new();
        for h in 1..=max_horizon.min(cases.len().saturating_sub(1)) {
            // One list of ratios per season (1-52), indexed directly.
            let mut by_season: Vec<Vec<f64>> = vec![Vec::new(); 54];
            for t in h..cases.len() {
                let (now, before) = (cases[t], cases[t - h]);
                let ratio = (now + 1.0) / (before + 1.0);
                // A missing value on either side gives no ratio.
                if ratio.is_finite()
                    && let Some(list) = by_season.get_mut(seasons[t] as usize)
                {
                    list.push(ratio);
                }
            }
            let by_season: BySeason = by_season
                .into_iter()
                .enumerate()
                .filter(|(_, ratios)| !ratios.is_empty())
                .filter_map(|(season, mut ratios)| {
                    ratios.shrink_to_fit();
                    Some((u32::try_from(season).ok()?, ratios))
                })
                .collect();
            by_horizon.insert(h, by_season);
        }
        Self { by_horizon }
    }

    /// The ratios for a season and horizon, with fallback 1 (see [`for_season`]).
    pub fn ratios(&self, season: u32, horizon: usize) -> Option<Vec<f64>> {
        for_season(self.by_horizon.get(&horizon)?, season)
    }
}

/// Fallback 1, for one horizon's ratios by season: the season's own ratios, else those of the
/// nearest seasons before and after it together, else those of every season. `None` when there
/// are no ratios at all.
fn for_season(by_season: &BySeason, season: u32) -> Option<Vec<f64>> {
    if by_season.is_empty() {
        return None;
    }
    if let Some(ratios) = by_season.get(&season) {
        return Some(ratios.clone());
    }
    let before = by_season.range(..season).next_back();
    let after = by_season.range(season + 1..).next();
    Some(match (before, after) {
        (Some((_, a)), Some((_, b))) => [a.as_slice(), b.as_slice()].concat(),
        _ => by_season.values().flatten().copied().collect(),
    })
}

/// The trained model: each location's ratios.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SeasonalShiftTrained {
    by_location: BTreeMap<String, SeasonalShifts>,
    /// All locations' ratios pooled (fallback 2), by horizon. Not stored: built the first time a
    /// forecast needs a horizon, so the ratios are not held twice when no location falls back.
    #[serde(skip)]
    pooled: Mutex<BTreeMap<usize, Arc<BySeason>>>,
}

impl Clone for SeasonalShiftTrained {
    fn clone(&self) -> Self {
        Self::new(self.by_location.clone())
    }
}

impl SeasonalShiftTrained {
    /// The trained model, from each location's ratios.
    pub fn new(by_location: BTreeMap<String, SeasonalShifts>) -> Self {
        Self {
            by_location,
            pooled: Mutex::default(),
        }
    }

    /// All locations' ratios for one horizon, by season, in location order.
    fn pooled(&self, horizon: usize) -> Arc<BySeason> {
        let mut cache = self.pooled.lock().unwrap_or_else(PoisonError::into_inner);
        cache
            .entry(horizon)
            .or_insert_with(|| {
                let mut by_season = BySeason::new();
                for shifts in self.by_location.values() {
                    for (season, ratios) in shifts.by_horizon.get(&horizon).into_iter().flatten() {
                        by_season
                            .entry(*season)
                            .or_default()
                            .extend_from_slice(ratios);
                    }
                }
                Arc::new(by_season)
            })
            .clone()
    }

    /// The ratios to forecast with, following fallbacks 2 to 5: horizon by horizon from `horizon`
    /// down to 1, try the location's ratios and then the pooled ones; the first candidate with
    /// more than one ratio wins, else the first single ratio, else `[1]`.
    pub fn ratios(&self, location: &str, season: u32, horizon: usize) -> Vec<f64> {
        let own = self.by_location.get(location);
        let mut first_single: Option<Option<Vec<f64>>> = None;
        for h in (1..=horizon).rev() {
            // The location's own ratios first, then all locations' (fallback 2).
            let own_ratios = own.map(|shifts| shifts.ratios(season, h));
            for ratios in own_ratios
                .into_iter()
                .chain(std::iter::once_with(|| for_season(&self.pooled(h), season)))
            {
                if ratios.as_ref().is_some_and(|r| r.len() > 1) {
                    return ratios.unwrap_or_default();
                }
                if first_single.is_none() {
                    first_single = Some(ratios);
                }
            }
        }
        first_single.flatten().unwrap_or_else(|| vec![1.0])
    }
}

/// The last observed (non-missing) value of a series, and how many periods before the end of
/// the series it is (1 when it is the last period). `None` when nothing was observed.
pub fn last_observed(cases: &[f64]) -> Option<(f64, usize)> {
    let at = cases.iter().rposition(|c| c.is_finite())?;
    Some((cases[at], cases.len() - at))
}

/// The forecast samples: the last value scaled by the quantiles of the ratios, never below zero.
pub fn forecast(last_value: f64, ratios: &[f64]) -> Option<Vec<f64>> {
    Some(
        quantile_samples(ratios)?
            .into_iter()
            .map(|r| ((last_value + 1.0) * r - 1.0).max(0.0))
            .collect(),
    )
}
