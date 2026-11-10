//! # Configurable baseline: how the forecast is calculated
//!
//! **Idea.** Forecast with chap-core's seasonal shift persistence (the last observed value,
//! scaled by how cases changed over the same horizon and time of year before; see
//! `baseline_seasonal_shift_persistence::model`), then move and stretch its samples by a known
//! amount. The model is a yardstick for evaluation metrics: how do they react to a forecast that
//! is too high, too low, too narrow or too wide?
//!
//! **Adjustment.** Each forecast period's 100 samples `s` are adjusted in `log(1 + cases)`
//! space, where a shift is a factor on the cases:
//!
//! ```text
//!     z  = log(1 + s)
//!     m  = median(z)
//!     z' = m + spread_factor * (z - m) + log_bias
//!     samples = max(exp(z') - 1, 0)
//! ```
//!
//! - `spread_factor` (default 1, at least 0) multiplies each sample's distance from the median:
//!   1 keeps the samples, 0 gives a point forecast at the median, 0.5 halves and 2 doubles the
//!   spread.
//! - `log_bias` (default 0) is added to every sample: 0.3 multiplies `1 + cases` by
//!   `exp(0.3)`, about 1.35; -0.3 by about 0.74.
//!
//! The median is numpy's: the mean of the two middle values of 100. Seasonal shift persistence
//! puts the median twice in the middle, so it is the median sample itself.
//!
//! **Defaults.** With `spread_factor` 1 and `log_bias` 0 the samples are returned unchanged, so
//! they are exactly seasonal shift persistence's. (The original computes the round trip
//! `exp(log(1 + s)) - 1` anyway, which moves a sample by a few units in the last place.)
//!
//! Port of chap-models' `configurable_baseline_model` (`predict.py`, `adjust`).

use baseline_core::quantiles::median;

/// Adjust one forecast period's samples in place (see the module documentation). The operations
/// and their order are the original's, so the results are the same `f64`s where the C library's
/// `log1p` and `expm1` agree with numpy's.
#[allow(clippy::float_cmp)] // the defaults are exact values, not results of a calculation
pub fn adjust(samples: &mut [f64], spread_factor: f64, log_bias: f64) {
    if spread_factor == 1.0 && log_bias == 0.0 {
        return;
    }
    // z = log(1 + s)
    for s in samples.iter_mut() {
        *s = s.ln_1p();
    }
    let m = median(&mut samples.to_vec());
    for z in samples.iter_mut() {
        // z' = m + spread_factor * (z - m) + log_bias, evaluated left to right like numpy,
        // then back to cases, never below zero.
        let shifted = m + spread_factor * (*z - m) + log_bias;
        *z = shifted.exp_m1().max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_keep_the_samples() {
        let mut s = vec![0.0, 0.1, 3.0, 3.0, 7.25];
        adjust(&mut s, 1.0, 0.0);
        assert_eq!(s, vec![0.0, 0.1, 3.0, 3.0, 7.25]);
    }

    #[test]
    fn zero_spread_is_the_median() {
        let mut s = vec![1.0, 3.0, 3.0, 15.0];
        adjust(&mut s, 0.0, 0.0);
        assert!(s.iter().all(|&v| (v - 3.0).abs() < 1e-12), "{s:?}");
    }

    #[test]
    fn bias_scales_one_plus_cases() {
        let mut s = vec![1.0, 3.0, 3.0, 15.0];
        adjust(&mut s, 1.0, 2.0_f64.ln());
        for (got, want) in s.iter().zip([3.0, 7.0, 7.0, 31.0]) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }

    #[test]
    fn never_below_zero() {
        let mut s = vec![0.0, 1.0, 1.0, 2.0];
        adjust(&mut s, 3.0, -1.0);
        assert!(s.iter().all(|&v| v >= 0.0));
        assert_eq!(s[0], 0.0);
    }
}
