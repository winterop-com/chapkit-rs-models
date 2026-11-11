//! # Simple multistep: how the forecast is calculated
//!
//! One model for all locations predicts next month's `disease_cases` from the months before. A
//! forecast several months ahead is made by feeding the model its own sampled predictions, one
//! month at a time.
//!
//! **Features.** The row for location `l` and month `t` holds, in this order:
//!
//! 1. for each covariate the configuration names (`additional_continuous_covariates`, none by
//!    default), its values 1, 2 and 3 months before `t` ([`COVARIATE_LAGS`]);
//! 2. one indicator per training location (1 for `l`, 0 for the others), in sorted order;
//! 3. the target `n_target_lags` months back up to the month before, oldest first.
//!
//! Gaps are filled before the lags are read: on the grid of all months in the data, each
//! location's series is filled forward (a missing month takes the last value before it), then
//! backward (missing months at the start take the first value). The covariate lags are counted in
//! the location's own rows before that filling.
//!
//! **Training.** Every month that has `n_target_lags` months before it on the grid gives a row
//! (rows with a value still missing are dropped). A random forest ([`crate::forest`]) is fitted to
//! predict the month's cases from its row. The forecast spread is one number, `scale`: the mean
//! absolute difference between the forest's prediction for each training row and its true value
//! (skpro's `ResidualDouble` with its default `DummyRegressor(strategy="mean")` on absolute
//! residuals, never below 1e-10).
//!
//! **Forecast.** Each of the `n_samples` trajectories of a location starts from the location's
//! last `n_target_lags` observed months. For each future month in turn: build the row (covariate
//! lags from the history and the future covariates, the location's indicator, the trajectory's
//! last `n_target_lags` values), let the forest predict the mean `m`, draw the month's value from
//! the normal distribution with mean `m` and standard deviation `scale`, and append it to the
//! trajectory. The month's forecast samples are the trajectories' values for it. The samples are
//! not clipped: a trajectory can go below zero, as in the original. A location without any
//! observed cases starts from missing lags, which the trees route as scikit-learn does (see
//! [`crate::forest`]); its later months then lag on its own samples.
//!
//! Port of chap-models' `chapkit_simple_multistep_model` 0.1.5 (`multistep.py`,
//! `transformations.py`, skpro's `ResidualDouble`).

use chapkit::rng::Rng;

use crate::forest::Forest;

/// Covariates enter as their values 1..=3 months back (`lag_all_features` in the original).
pub const COVARIATE_LAGS: usize = 3;

/// Lower bound of the forecast spread (skpro's `min_scale`).
pub const MIN_SCALE: f64 = 1e-10;

/// Fill a series' gaps in place: forward from the last value seen, then backward from the first
/// (pandas' `ffill().bfill()`). A series without any value stays missing.
pub fn fill_gaps(values: &mut [f64]) {
    let mut last = f64::NAN;
    for v in values.iter_mut() {
        if v.is_nan() {
            *v = last;
        } else {
            last = *v;
        }
    }
    let mut next = f64::NAN;
    for v in values.iter_mut().rev() {
        if v.is_nan() {
            *v = next;
        } else {
            next = *v;
        }
    }
}

/// A covariate's lag features in one location's rows (in period order): for each row, the values
/// 1..=[`COVARIATE_LAGS`] rows back, missing where there is no such row.
pub fn covariate_lags(values: &[f64]) -> Vec<[f64; COVARIATE_LAGS]> {
    (0..values.len())
        .map(|i| {
            std::array::from_fn(|k| {
                i.checked_sub(k + 1)
                    .and_then(|j| values.get(j))
                    .copied()
                    .unwrap_or(f64::NAN)
            })
        })
        .collect()
}

/// One feature row: the covariate features, the location indicator (`location` is the
/// location's position among `n_locations` training locations; `None`, all zeros, for a location
/// not seen in training) and the target lags, oldest first.
pub fn feature_row(
    covariate_features: &[f64],
    location: Option<usize>,
    n_locations: usize,
    target_lags: &[f64],
) -> Vec<f64> {
    let mut row = Vec::with_capacity(covariate_features.len() + n_locations + target_lags.len());
    row.extend_from_slice(covariate_features);
    row.extend((0..n_locations).map(|j| if Some(j) == location { 1.0 } else { 0.0 }));
    row.extend_from_slice(target_lags);
    row
}

/// The training rows of one location: for each month `t >= n_target_lags` on the grid, `t`, its
/// feature row and target. `covariate_features[t]` holds the location's covariate features at
/// grid month `t` and `target` its gap-filled cases. Rows with a missing value are left out.
pub fn training_rows(
    covariate_features: &[Vec<f64>],
    target: &[f64],
    location: usize,
    n_locations: usize,
    n_target_lags: usize,
) -> Vec<(usize, Vec<f64>, f64)> {
    (n_target_lags..target.len())
        .filter_map(|t| {
            let lags = target.get(t - n_target_lags..t)?;
            let row = feature_row(
                covariate_features.get(t)?,
                Some(location),
                n_locations,
                lags,
            );
            let y = *target.get(t)?;
            (!y.is_nan() && row.iter().all(|v| !v.is_nan())).then_some((t, row, y))
        })
        .collect()
}

/// The forecast spread: the mean absolute in-sample residual of the forest, at least
/// [`MIN_SCALE`].
pub fn residual_scale(forest: &Forest, rows: &[Vec<f64>], y: &[f64]) -> f64 {
    let n = rows.len().min(y.len());
    if n == 0 {
        return MIN_SCALE;
    }
    let total: f64 = rows
        .iter()
        .zip(y)
        .map(|(x, yi)| (yi - forest.predict(x)).abs())
        .sum();
    (total / n as f64).max(MIN_SCALE)
}

/// Most feature values built at once while forecasting (8 MB): the trajectories' rows go to the
/// forest in blocks of this many values (at least one row), so the feature matrix does not grow
/// with `n_samples` or the number of locations.
pub const BLOCK_VALUES: usize = 1_000_000;

/// `n_samples` recursive trajectories for one location. `covariates[s]` holds the covariate
/// features of future month `s` (empty without covariates), `previous` the last `n_target_lags`
/// observed cases, oldest first. Returns, per future month, its `n_samples` samples.
///
/// Each trajectory keeps only its last `n_target_lags` values (a rolling window): the earlier ones
/// are in the returned samples already.
#[allow(clippy::too_many_arguments)]
pub fn forecast(
    forest: &Forest,
    scale: f64,
    covariates: &[Vec<f64>],
    location: Option<usize>,
    n_locations: usize,
    previous: &[f64],
    n_samples: usize,
    rng: &mut Rng,
) -> Vec<Vec<f64>> {
    forecast_in_blocks(
        forest,
        scale,
        covariates,
        (location, n_locations),
        previous,
        n_samples,
        BLOCK_VALUES,
        rng,
    )
}

/// [`forecast`], with the rows built `block_values` values at a time.
#[allow(clippy::too_many_arguments)]
fn forecast_in_blocks(
    forest: &Forest,
    scale: f64,
    covariates: &[Vec<f64>],
    (location, n_locations): (Option<usize>, usize),
    previous: &[f64],
    n_samples: usize,
    block_values: usize,
    rng: &mut Rng,
) -> Vec<Vec<f64>> {
    let lags = previous.len();
    // Trajectory `i`'s last `lags` values, oldest first, at `windows[i * lags..(i + 1) * lags]`.
    let mut windows = previous.repeat(n_samples);
    let window = |windows: &[f64], i: usize| -> std::ops::Range<usize> {
        let start = i.saturating_mul(lags);
        start..start.saturating_add(lags).min(windows.len())
    };
    let mut out = Vec::with_capacity(covariates.len());
    let (mut rows, mut block_means, mut means) = (Vec::new(), Vec::new(), Vec::new());
    for step_covariates in covariates {
        // Every trajectory's row for this month, a block at a time, then the forest's mean for each.
        let width = step_covariates.len() + n_locations + lags;
        let block = (block_values / width.max(1)).max(1);
        means.clear();
        let mut start = 0;
        while start < n_samples {
            let end = start.saturating_add(block).min(n_samples);
            rows.clear();
            for i in start..end {
                let w = windows.get(window(&windows, i)).unwrap_or_default();
                rows.extend(feature_row(step_covariates, location, n_locations, w));
            }
            forest.predict_rows(&rows, width, &mut block_means);
            means.extend_from_slice(&block_means);
            start = end;
        }
        // Draw each trajectory's value for the month and roll it into its window.
        let samples: Vec<f64> = means.iter().map(|&m| rng.gauss(m, scale)).collect();
        for (i, &value) in samples.iter().enumerate() {
            let range = window(&windows, i);
            if let Some(w) = windows.get_mut(range).filter(|w| !w.is_empty()) {
                w.copy_within(1.., 0);
                if let Some(last) = w.last_mut() {
                    *last = value;
                }
            }
        }
        out.push(samples);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forest::{Columns, ForestParams};

    #[test]
    fn gaps_fill_forward_then_backward() {
        let mut v = [f64::NAN, 1.0, f64::NAN, f64::NAN, 4.0, f64::NAN];
        fill_gaps(&mut v);
        assert_eq!(v, [1.0, 1.0, 1.0, 1.0, 4.0, 4.0]);
        let mut empty = [f64::NAN; 2];
        fill_gaps(&mut empty);
        assert!(empty.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn covariate_lags_count_rows_back() {
        let lags = covariate_lags(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(lags[3], [3.0, 2.0, 1.0]);
        assert_eq!(lags[1][0], 1.0);
        assert!(lags[1][1].is_nan());
    }

    #[test]
    fn training_rows_hold_covariates_indicator_and_lags() {
        let target = [1.0, 2.0, 3.0, 4.0];
        let covariates = vec![vec![10.0], vec![20.0], vec![30.0], vec![40.0]];
        let rows = training_rows(&covariates, &target, 1, 3, 2);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], (2, vec![30.0, 0.0, 1.0, 0.0, 1.0, 2.0], 3.0));
        assert_eq!(rows[1].1, vec![40.0, 0.0, 1.0, 0.0, 2.0, 3.0]);
    }

    #[test]
    fn trajectories_feed_back_their_own_samples() {
        // The forest returns its first target lag (x[1]) shifted by one: a random walk upwards.
        let rows: Vec<Vec<f64>> = (0..200).map(|i| vec![1.0, f64::from(i)]).collect();
        let y: Vec<f64> = (0..200).map(|i| f64::from(i) + 1.0).collect();
        let params = ForestParams {
            n_estimators: 5,
            max_depth: 20,
            min_samples_leaf: 1,
        };
        let forest = Forest::fit(&Columns::from_rows(&rows), &y, params, 1);
        let mut rng = Rng::new(2);
        let out = forecast(
            &forest,
            1e-10,
            &[vec![], vec![], vec![]],
            Some(0),
            1,
            &[50.0],
            4,
            &mut rng,
        );
        assert_eq!(out.len(), 3);
        for (step, samples) in out.iter().enumerate() {
            assert_eq!(samples.len(), 4);
            for s in samples {
                assert!((s - (51.0 + step as f64)).abs() < 1.5, "{s} at step {step}");
            }
        }
    }

    /// The forecast as it was first written: every trajectory keeps its whole history and all
    /// rows of a month are built at once.
    #[allow(clippy::too_many_arguments)]
    fn forecast_whole(
        forest: &Forest,
        scale: f64,
        covariates: &[Vec<f64>],
        location: Option<usize>,
        n_locations: usize,
        previous: &[f64],
        n_samples: usize,
        rng: &mut Rng,
    ) -> Vec<Vec<f64>> {
        let mut trajectories: Vec<Vec<f64>> = vec![previous.to_vec(); n_samples];
        let lags = previous.len();
        let mut out = Vec::new();
        let (mut rows, mut means) = (Vec::new(), Vec::new());
        for step_covariates in covariates {
            rows.clear();
            for t in &trajectories {
                rows.extend(feature_row(
                    step_covariates,
                    location,
                    n_locations,
                    &t[t.len() - lags..],
                ));
            }
            let width = step_covariates.len() + n_locations + lags;
            forest.predict_rows(&rows, width, &mut means);
            let samples: Vec<f64> = means.iter().map(|&m| rng.gauss(m, scale)).collect();
            for (t, &v) in trajectories.iter_mut().zip(&samples) {
                t.push(v);
            }
            out.push(samples);
        }
        out
    }

    #[test]
    fn rolling_windows_and_blocks_give_the_same_forecast() {
        // Two covariate features, three locations, three lags.
        let rows: Vec<Vec<f64>> = (0..300)
            .map(|i| {
                let f = f64::from(i);
                vec![f % 7.0, f % 5.0, 0.0, 1.0, 0.0, f, f + 1.0, f + 2.0]
            })
            .collect();
        let y: Vec<f64> = (0..300).map(|i| f64::from(i) + 3.0).collect();
        let params = ForestParams {
            n_estimators: 7,
            max_depth: 12,
            min_samples_leaf: 2,
        };
        let forest = Forest::fit(&Columns::from_rows(&rows), &y, params, 4);
        let covariates: Vec<Vec<f64>> = (0..6).map(|s| vec![f64::from(s), 2.0]).collect();
        let previous = [100.0, 101.0, 102.0];
        let whole = forecast_whole(
            &forest,
            2.0,
            &covariates,
            Some(1),
            3,
            &previous,
            37,
            &mut Rng::new(9),
        );
        for block_values in [1, 8, 9, 80, 1_000_000] {
            let rolled = forecast_in_blocks(
                &forest,
                2.0,
                &covariates,
                (Some(1), 3),
                &previous,
                37,
                block_values,
                &mut Rng::new(9),
            );
            assert_eq!(rolled, whole, "{block_values}");
        }
        // No lags at all, and no samples.
        let none = forecast(&forest, 2.0, &covariates, None, 3, &[], 4, &mut Rng::new(1));
        assert_eq!(none.len(), 6);
        assert!(none.iter().all(|s| s.len() == 4));
        let empty = forecast(
            &forest,
            2.0,
            &covariates,
            None,
            3,
            &previous,
            0,
            &mut Rng::new(1),
        );
        assert!(empty.iter().all(Vec::is_empty));
    }
}
