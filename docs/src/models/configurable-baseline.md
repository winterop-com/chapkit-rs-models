# Configurable baseline

**Package:** `baseline-configurable` · **Binary:** `baseline-configurable`
· **Service id:** `configurable-baseline` · **Port of:** chap-models
[`configurable_baseline_model`](https://github.com/chap-models/configurable_baseline_model) (Knut Rand)
· **Code:** `crates/baseline-configurable/src/model.rs`

## In one sentence

[Seasonal shift persistence](seasonal-shift-persistence.md), with its samples moved and stretched by a known
amount in `log(1 + cases)` space.

## The method

The model is a yardstick for evaluation metrics rather than a forecasting model: with the options it gives a
forecast that is too high, too low, too narrow or too wide by a known amount, so you can see how a metric
reacts.

**Forecast.** First the seasonal shift persistence forecast, unchanged: 100 samples per location and future
period ([how](seasonal-shift-persistence.md#the-method)).

**Adjustment.** Each period's samples *s* are then adjusted in `log(1 + cases)` space:

```text
z  = log(1 + s)
m  = median(z)
z' = m + spread_factor × (z − m) + log_bias
samples = max(exp(z') − 1, 0)
```

The median is the mean of the two middle values of the 100 (numpy's). Seasonal shift persistence puts its
median twice in the middle, so `m` is the log of the median sample, and the median forecast only moves with
`log_bias`.

## Options

| option | default | meaning |
|---|---|---|
| `spread_factor` | `1.0` | at least 0: multiplies each sample's distance from the median in log space. 1 keeps the samples, 0 gives a point forecast at the median, 0.5 halves and 2 doubles the spread |
| `log_bias` | `0.0` | added to every sample in log space: `1 + cases` is multiplied by `exp(log_bias)`, so 0.3 is about 35% too high and −0.3 about 26% too low |

With the default options the samples are exactly seasonal shift persistence's. The options are the model's
chapkit configuration (`POST /api/v1/configs`, `{"name": ..., "data": {"spread_factor": 2.0, "log_bias": 0.3}}`);
a negative or non-numeric `spread_factor` is rejected. For hyperparameter search, `/api/v1/info` declares
`spread_factor` in [0, 3] and `log_bias` in [−0.5, 0.5].

The original seeds six configurations into Chap: `baseline` (the defaults), `biased_over` (`log_bias` 0.3),
`biased_under` (−0.3), `point` (`spread_factor` 0), `narrow` (0.5) and `wide` (2).

## Worked example

Location A in the [example data](../verification.md), **May 2021** (one month after the last observation, 23
cases). Seasonal shift persistence forecasts a median of 24.0, with samples from 23.0 to 27.0 (see its
[worked example](seasonal-shift-persistence.md#worked-example)). In log space the median is log(25) = 3.219.

| configuration | sample 0 | sample 10 | median | sample 89 | sample 99 |
|---|---|---|---|---|---|
| defaults | 23.01 | 23.21 | 24.00 | 26.36 | 26.97 |
| `log_bias` 0.3 | 31.41 | 31.68 | 32.75 | 35.94 | 36.76 |
| `log_bias` −0.3 | 16.79 | 16.94 | 17.52 | 19.27 | 19.72 |
| `spread_factor` 0 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |
| `spread_factor` 0.5 | 23.50 | 23.60 | 24.00 | 25.16 | 25.44 |
| `spread_factor` 2 | 22.06 | 22.45 | 24.00 | 28.95 | 30.29 |
| both (2 and 0.3) | 30.13 | 30.65 | 32.75 | 39.43 | 41.24 |

With `log_bias` 0.3 the median becomes exp(3.219 + 0.3) − 1 = 25 × 1.350 − 1 = **32.75**. With
`spread_factor` 2 the highest sample, log(27.97) = 3.331, is 0.112 above the median in log space; doubled
that is exp(3.219 + 0.224) − 1 = **30.29**. Stretching in log space is not symmetric in cases: the samples
above the median move further than those below it.

## Inputs and outputs

| | |
|---|---|
| training data | `time_period` (monthly or weekly), `location`, `disease_cases` |
| historic data | the series up to the forecast start; its last observed value is the starting point |
| configuration | `spread_factor`, `log_bias` (see [Options](#options)) |
| output | 100 samples per location and future period, never negative |
| fails when | as seasonal shift persistence, or `spread_factor` is negative |

## Same output as the original

`tests/parity.rs` checks the port against the original's calculation on the [example data](../verification.md),
monthly and weekly, for the defaults and six other option sets. `scripts/gen_configurable_fixtures.py` makes
the expected output: it runs chap-core 2.7.0's built-in seasonal shift persistence (the original carries a copy
of it) and applies the original's `adjust` from `predict.py`, numpy operation for numpy operation.

- **Other options:** the original's operations in its order, so the same `f64`s, sample for sample, on the
  platform the fixtures were made on (macOS). `log1p` and `expm1` come from the platform's C library on both
  sides, so across platforms a sample can differ in its last bits. Measured over every fixture sample with
  Linux's glibc (amd64 and arm64): at most 1.8 × 10⁻¹⁵ relative to the sample (floored at one case); a spread
  of 2 doubles the last-bit difference of `log1p` before `expm1`. The test allows 10⁻¹⁴.
- **Defaults:** exactly seasonal shift persistence's samples (and chap-core's). The original computes
  `exp(log(1 + s)) − 1` even then, which moves a sample by up to 4 units in the last place (about 3 × 10⁻¹⁴
  here); the port skips the round trip, as the original's README promises the built-in model's samples.

## In code

```rust,ignore
pub fn adjust(samples: &mut [f64], spread_factor: f64, log_bias: f64) {
    if spread_factor == 1.0 && log_bias == 0.0 {
        return;
    }
    for s in samples.iter_mut() {
        *s = s.ln_1p();
    }
    let m = median(&mut samples.to_vec());
    for z in samples.iter_mut() {
        let shifted = m + spread_factor * (*z - m) + log_bias;
        *z = shifted.exp_m1().max(0.0);
    }
}
```

The forecast it adjusts is `forecast_location` in `crates/baseline-seasonal-shift-persistence`.
