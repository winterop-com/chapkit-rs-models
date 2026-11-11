# Auto ARIMA

**Package:** `auto-arima` · **Binary:** `auto-arima` · **Service id:** `auto-arima-crs` · **Port of:**
chap-models [`auto_arima_chapkit`](https://github.com/chap-models/auto_arima_chapkit) 1.0.4 (Harsha, Cardiff
University; adapted to chapkit by Halvard), an R model built on fable · **Code:**
`crates/auto-arima/src/model.rs`, the ARIMA in chapkit-rs's `chapkit::ts` (`crates/chapkit-ts`)

## In one sentence

For each location, an ARIMA model of the case counts alone, with its orders chosen automatically as R's
`fable::ARIMA()` chooses them, run over the latest history and sampled from its normal forecast.

## The method

The original is three R scripts. Per location:

**Training** (`train.R`): `model(ARIMA(disease_cases))` on the location's series, with no covariates. fable
chooses the model itself:

- `D`, seasonal differences (0 or 1): 1 when the STL seasonal strength of the series is at least 0.64;
- `d`, differences (0 to 2): one more while the KPSS test rejects stationarity (5%) of the series differenced
  so far;
- `p` and `q` (0 to 5), `P` and `Q` (0 to 2), and whether to add a constant: a stepwise search that starts from
  ARIMA(2,d,2)(1,D,1) and the null, AR and MA models, then moves to neighbouring models while the AICc
  improves, with `p + q + P + Q <= 6`, at most two of {constant, `d`, `D`}, and no model with an AR or MA
  root within 1.01 of the unit circle;
- the coefficients by maximum likelihood (R's `stats::arima`, started from conditional sums of squares). For
  a period above 12 or more than 150 values the search compares models by conditional sums of squares and only
  the winner is refitted by likelihood.

The season is 12 months. The port also takes weekly data, with a season of 52 weeks (a 53rd ISO week is one
more step of the series, not a season of its own, as in chap-core); the model info declares `period_type`
`any`.

**Forecast** (`predict.R`): `refit(model, historic)`, then `forecast(h)`. fable's `refit()` keeps the trained
orders **and coefficients** (`reestimate = FALSE`): it runs the model over the history it is given, so the
forecast starts from the latest observations, and recomputes the residual variance `sigma2` there (sum of
squared residuals over `n - d - D * season`; at training the divisor is that minus the number of
coefficients). Each step's forecast is a normal distribution with the ARIMA's mean and variance.

**Samples**: 100 independent draws per step from that normal distribution (`rnorm`). Steps are drawn
independently and samples are not clipped at zero, as in the original.

## Options

| option | default | meaning |
|---|---|---|
| `n_samples` | `100` | draws per forecast period, at most 100,000 (fixed at 100 in the original) |
| `seed` | not set | seeds the draws; not set, every run differs, as in the original |

The original has no options of its own. Forecasts may be 0 to 12 periods long, as the original declares.

## Worked example

Laos (`laos_subset.csv` from chap-core's example data, 3 provinces), trained on January 2010 to March 2012
(27 months, chap-core's training set for 7 splits of 3 months). fable and the port choose:

| province | model | coefficients | `sigma2` | AICc |
|---|---|---|---|---|
| Bokeo | ARIMA(1,0,0)(0,1,0)[12] | ar1 0.4783 | 35.61 | 100.38 |
| Savannakhet | ARIMA(1,0,0) | ar1 0.7767 | 20174.44 | 348.66 |
| Vientiane | ARIMA(2,0,0)(0,1,0)[12] | ar1 1.2958, ar2 -0.5829 | 39632.41 | 209.35 |

Bokeo's model says: this month is last year's month plus 0.478 times last month's difference from its own
last year. March 2012 had 1 case, as had March 2011, so the difference is 0 and the forecast for April to June
2012 is April to June 2011: means **1, 3 and 4** cases. Refitted on the same 27 months, `sigma2` is the same
sum of squares over 15 instead of 14 values, 33.24, so the sd of the first month is sqrt(33.24) = **5.77**,
then sqrt(33.24 × (1 + 0.478²)) = **6.39** and **6.53**. The 100 samples of each month are normal draws with
these means and sds; most months get some negative samples.

## Inputs and outputs

| | |
|---|---|
| training data | `time_period` (monthly or weekly), `location`, `disease_cases`; other columns are ignored |
| historic data | each location's series up to the forecast start (it may be shorter or longer than the training data) |
| future data | the periods to forecast; only their number and position after the history are used |
| configuration | see [Options](#options) |
| output | `n_samples` samples per location and future period |
| fails when | a location has no observed cases, no ARIMA can be fitted (where fable keeps a null model and forecasts missing values, for example "data follows a simple polynomial" on chapkit-rs's 50-week synthetic test data, which fable rejects too), a location to forecast was not trained or has no history, the data mixes monthly and weekly periods, `n_samples` is 0 or above 100,000, or a forecast would hold more than 250 million samples (future rows times `n_samples`) |

A gap inside a location's series counts as missing values, which the ARIMA steps over (R's `NA`), as chap-core
fills gaps before a model sees them.

**Weekly data.** The original fails on weekly data: with chap-core's weekly periods
(`2016-02-29/2016-03-06`) `yearmonth()` maps the weeks of a month to one month and `as_tsibble()` stops with
"A valid tsibble must have distinct rows identified by key and index"; with `2016W09` it stops with "cannot be
expressed as Date type". The port's weekly forecasts are checked against the same fable code run on a
`yearweek` index instead.

## How it is checked

`scripts/gen_auto_arima_fixtures.R` runs fable in the original's own image (`ghcr.io/chap-models/auto_arima_chapkit`,
R 4.5.0, fable 0.5.0) and writes `crates/auto-arima/fixtures/fable_monthly.json` and `fable_weekly.json`: for
each location the orders, coefficients, `sigma2`, AICc and 3-step forecast mean and sd of `ARIMA(disease_cases)`
fitted on the first `n` periods, and of the model trained on chap-core's training set refitted, as `predict.R`
does, on each split's history. `tests/fable.rs` checks the port against them.

**Monthly**: Laos, 3 provinces × 7 training lengths (24, 26, 27, 28, 30, 33 and 36 months).

| check | result |
|---|---|
| orders and constant | 21 of 21 as fable's |
| forecast mean and sd after training | within 3.0e-6 relative (most within 1e-9) |
| forecast mean and sd after `refit()` on 27 to 33 months, the backtest's 7 splits | within 1.9e-7 relative |
| a forecast from a history shorter than the training data (Bokeo, trained on 36 months, refitted on 20) | within 1e-6 |

**Weekly**: chap-core's `nicaragua_weekly_data.csv`, 4 departments (boaco, leon, managua, matagalpa), the 251
complete weeks from 2016-02-29 to 2020-12-20, trained on 200 and on 242 weeks.

| check | result |
|---|---|
| orders and constant | 8 of 8 as fable's |
| forecast mean and sd after training, and after `refit()` on 242 to 248 weeks | within 1e-9 relative, but for matagalpa at 242 weeks (below) |

matagalpa's ARIMA(3,1,1)(1,0,1)[52] at 242 weeks differs by up to 6.1e-3 relative in the mean and 1.4e-3 in
the sd. For this model R's initial state covariance (`getQ0`, Gardner et al. 1980) is inaccurate: it has a
negative eigenvalue (-0.32), so fable's likelihood, and with it its coefficients and forecast, rest on a
broken matrix. chapkit-ts uses Gardner's result only where it satisfies its own equation and the exact
covariance otherwise; the chosen orders and the AICc agree (within 1e-5), the coefficients differ by up to
2.6e-2. R itself gives coefficients 1e-2 apart for this model on arm64 and on x86.

**Other automatic ARIMAs choose differently.** statsforecast's `AutoARIMA` (2.1.1, the method chap-core's
`mstl_arima` weather provider uses), with season length 12 or 52 on the same series, agrees with fable on 10
of the 21 monthly orders and on none of the 8 weekly ones (for example managua at 242 weeks: fable
ARIMA(0,1,4), statsforecast ARIMA(4,1,0)). It follows R's `forecast::auto.arima`, which tests seasonality
differently and differences until KPSS passes; a port of "auto ARIMA" has to say which one it follows.

### Backtests

Both models on the same splits, scored with chap-core 2.7.0's metrics: 7 splits of 3 periods, one training,
the model refitted on each split's history. On the R side chap-core's own `Evaluation` runs the original's
`train.R` and `predict.R` in its image for every split; on the port's side `auto-arima eval --data FILE
--periods 3 --splits 7 --config '{"seed": N}'`. Since the draws are random (the original sets no seed), each
side runs 5 times; mean ± standard deviation over the runs:

| data | model | runs | CRPS | MAE | RMSE | coverage 10-90 | coverage 25-75 |
|---|---|---|---|---|---|---|---|
| Laos monthly (3 × 36 months, 63 forecasts) | original (R) | 5 | 68.53 ± 1.17 | 97.71 ± 3.88 | 142.12 ± 4.49 | 0.749 ± 0.026 | 0.435 ± 0.031 |
| | port | seeds 1-5 | 69.95 ± 3.03 | 99.56 ± 6.05 | 144.25 ± 6.52 | 0.756 ± 0.043 | 0.422 ± 0.052 |
| Nicaragua weekly (4 × 251 weeks, 84 forecasts) | fable's forecasts, 100 normal draws | 5 | 31.05 ± 0.29 | 30.44 ± 1.27 | 40.72 ± 1.58 | 0.974 ± 0.005 | 0.807 ± 0.031 |
| | port | seeds 1-5 | 30.96 ± 0.31 | 29.09 ± 1.69 | 40.08 ± 1.46 | 0.976 ± 0.008 | 0.810 ± 0.021 |

The forecast distributions of every split are fable's (the refit rows above), so the scores differ only by the
random draws: the differences are within the spread between runs of either model. The original cannot run
weekly, so the weekly reference scores fable's means and sds from the fixture, drawn the same way in Python.
The log1p metrics are left out: almost every forecast (59 of 63 monthly) has samples below -1, where `log1p`
is undefined, so they rest on a handful of rows.

## Speed

On an Apple M2 Max, 2026-11-10. The original's image is linux/amd64 only, so on this machine it runs
emulated; the port runs natively.

| | original (R, in its container) | port |
|---|---|---|
| train, Laos 3 × 27 months | 2.42 s (`Rscript train.R`; 2.69 s with the container) | 35 ms |
| forecast 3 months | 2.0 s (`Rscript predict.R`; 2.25 s with the container) | 0.25 ms |
| whole backtest, 1 training and 7 forecasts | 21 s through chap-core's evaluation | 28 to 48 ms, scoring included |
| train, Nicaragua 4 × 242 weeks | 7.4 s fable fits alone (inside R, without start-up) | 0.70 s |
| forecast 3 weeks | | 16 ms |
| image | 3.48 GB (989 MB compressed) | 26.9 MB (8.7 MB compressed) |

Most of the original's time per call is R start-up: loading dplyr, fable, tsibble, lubridate and
distributional takes 1.85 s of each `Rscript`; its fits of the 3 Laos provinces take 0.29 s. The port's times
are criterion's (`make bench BENCH=auto_arima`), measured while other builds were running. statsforecast's
`AutoARIMA` took 72 s for the 4 weekly series at 242 weeks (Python, native, first call included).

## In code

```rust,ignore
/// Step 1: choose and fit the ARIMA of `values` with a season of `season` periods.
pub fn fit(values: &[f64], season: usize) -> ts::Result<Fitted> {
    ts::auto_arima(values, &AutoArima::new(season))
}

/// Step 2: the trained model run over `history`, then the mean and variance of steps 1..=h.
pub fn forecast(fitted: &Fitted, history: &[f64], h: usize) -> ts::Result<Forecast> {
    fitted.filter(history)?.forecast(h)
}

/// Step 3: `n` draws from the normal distribution with `mean` and standard deviation `sd`.
pub fn samples(mean: f64, sd: f64, n: usize, rng: &mut Rng) -> Vec<f64> {
    (0..n).map(|_| rng.gauss(mean, sd)).collect()
}
```
