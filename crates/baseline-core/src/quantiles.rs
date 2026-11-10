//! # Forecast samples from observed values
//!
//! All three baselines forecast with the *empirical distribution* of values they have seen:
//! instead of fitting a distribution, they turn a set of observed numbers into 100 forecast
//! samples by taking quantiles. This file is that shared step. Port of chap-core's
//! `chap_core/models/builtin/empirical.py`.

use std::sync::LazyLock;

/// Number of samples in every forecast, as in chap-core.
pub const N_SAMPLES: usize = 100;

/// The quantile levels the samples are taken at, sorted:
///
/// - 99 levels evenly spaced and symmetric around 0.5: (i + 0.5) / 99 for i = 0..98,
///   that is 0.00505, 0.01515, ..., 0.5, ..., 0.98485, 0.99495;
/// - plus 0.5 a second time.
///
/// With 0.5 twice, the two middle samples are both the median, so the median of the 100
/// samples is exactly the median of the values, whichever median definition a metric uses.
static QUANTILE_LEVELS: LazyLock<[f64; N_SAMPLES]> = LazyLock::new(|| {
    let mut levels = [0.5; N_SAMPLES];
    for (i, level) in levels.iter_mut().take(N_SAMPLES - 1).enumerate() {
        *level = (i as f64 + 0.5) / (N_SAMPLES - 1) as f64;
    }
    levels.sort_by(f64::total_cmp);
    levels
});

/// The 100 forecast samples for a set of observed values: their quantiles at
/// [`QUANTILE_LEVELS`]. `None` when there are no values.
///
/// Quantiles use linear interpolation, numpy's default: sort the values, and for level q
/// take position q * (n - 1); between two values, interpolate linearly.
pub fn quantile_samples(values: &[f64]) -> Option<Vec<f64>> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let last = (sorted.len() - 1) as f64;
    let at = |i: usize| sorted.get(i).copied().unwrap_or(f64::NAN);
    Some(
        QUANTILE_LEVELS
            .iter()
            .map(|&q| {
                let position = q * last;
                let below = position.floor() as usize;
                let above = position.ceil() as usize;
                at(below) + (at(above) - at(below)) * (position - below as f64)
            })
            .collect(),
    )
}

/// The median across several sample sets, taken level by level: sample i of the result is
/// the median of sample i of every set (numpy `median(axis=0)`). Because every set is sorted,
/// so is the result. `None` when there are no sets or their lengths differ.
///
/// Used to forecast a location that has no observations of its own from all the others.
pub fn median_per_level<'a>(sets: impl IntoIterator<Item = &'a Vec<f64>>) -> Option<Vec<f64>> {
    let sets: Vec<&Vec<f64>> = sets.into_iter().collect();
    let n = sets.first()?.len();
    if sets.iter().any(|s| s.len() != n) {
        return None;
    }
    let mut level_values = Vec::with_capacity(sets.len());
    Some(
        (0..n)
            .map(|i| {
                level_values.clear();
                level_values.extend(sets.iter().filter_map(|s| s.get(i).copied()));
                median(&mut level_values)
            })
            .collect(),
    )
}

/// The median of a non-empty list: the middle value, or the mean of the two middle values
/// (numpy `median`). Sorts `values`.
pub fn median(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let n = values.len();
    let at = |i: usize| values.get(i).copied().unwrap_or(f64::NAN);
    if n % 2 == 1 {
        at(n / 2)
    } else {
        f64::midpoint(at(n / 2 - 1), at(n / 2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_match_chap_core() {
        // Sampling [0, 1] returns the levels themselves.
        let l = quantile_samples(&[0.0, 1.0]).unwrap_or_default();
        assert_eq!(l.len(), N_SAMPLES);
        assert!((l[0] - 0.5 / 99.0).abs() < 1e-15);
        assert_eq!((l[49], l[50]), (0.5, 0.5));
        assert!((l[99] - 98.5 / 99.0).abs() < 1e-15);
    }

    #[test]
    fn edge_cases() {
        assert_eq!(quantile_samples(&[3.0]), Some(vec![3.0; N_SAMPLES]));
        assert_eq!(quantile_samples(&[]), None);
        let (a, b, c) = (vec![1.0, 10.0], vec![3.0, 20.0], vec![2.0, 30.0]);
        assert_eq!(median_per_level([&a, &b]), Some(vec![2.0, 15.0]));
        assert_eq!(median_per_level([&a, &b, &c]), Some(vec![2.0, 20.0]));
        assert_eq!(median_per_level([&a, &vec![1.0]]), None);
    }
}
