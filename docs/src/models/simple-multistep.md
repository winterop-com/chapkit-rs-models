# Simple multistep

**Package:** `simple-multistep` · **Binary:** `simple-multistep` · **Service id:** `simple-multistep-crs`
· **Port of:** chap-models
[`chapkit_simple_multistep_model`](https://github.com/chap-models/chapkit_simple_multistep_model) 0.1.5 (HISP
Oslo and HISP West and Central Africa), the Chap marketplace's reference model · **Code:**
`crates/simple-multistep/src/model.rs`, the forest in `src/forest.rs`

## In one sentence

A random forest predicts next month's cases from the last six months and the location; to forecast further
ahead, each sampled month is fed back in as if it had been observed.

## The method

Unlike the baselines, this model fits something: one regression model shared by all locations, which only
ever predicts **one month ahead**. Longer forecasts are built from it step by step (a *recursive* forecast).

**Features.** The row for location *l* and month *t* holds, in this order:

1. for each configured covariate (`additional_continuous_covariates`, none by default), its values 1, 2 and 3
   months before *t*;
2. one indicator per training location, in sorted order: 1 for *l*, 0 for the others;
3. the cases of the `n_target_lags` months before *t* (6 by default), oldest first.

Before the lags are read, gaps are filled: on the grid of all months in the data, a missing month takes the
location's last value before it, and missing months at the start take its first value.

**Training.** Every month with `n_target_lags` months before it gives one row and its cases as the target
(17 locations with cases × 134 months = 2,278 rows in the original's example data). A random forest is fitted
to them, exactly as the original configures scikit-learn's `RandomForestRegressor`: 100 trees, each grown on
a bootstrap sample of the rows, splitting on the best of √(number of features) randomly drawn features, at
most `rf_max_depth` deep and with at least `rf_min_samples_leaf` rows per leaf. A forest predicts the mean of
its trees.

**Spread.** One number for all forecasts: the mean absolute difference between the forest's prediction for
each training row and the row's true cases,

```text
scale = mean(|y_i − forest(x_i)|)
```

This is what the original's skpro `ResidualDouble` computes with its defaults (a "residual model" that
predicts the mean absolute residual, whatever the row).

**Forecast.** Each location gets `n_samples` (100) trajectories, all starting from its last `n_target_lags`
observed months. For each future month in turn, every trajectory:

1. builds its row: the covariate lags (from the history and the future covariates), the location indicator,
   and its own last `n_target_lags` values;
2. asks the forest for the mean *m*;
3. draws the month's value from a normal distribution with mean *m* and standard deviation `scale`, and
   appends it.

The month's 100 samples are the trajectories' values. Uncertainty grows with the horizon because later
months start from sampled, not observed, values. Samples are not clipped at zero, as in the original.

## Options

| option | default | meaning |
|---|---|---|
| `n_target_lags` | `6` | months of cases in each row, at most 1,200 |
| `n_samples` | `100` | trajectories per location, at most 100,000 |
| `rf_max_depth` | `10` | maximum depth of each tree |
| `rf_min_samples_leaf` | `5` | minimum rows in a leaf |
| `rf_n_estimators` | `100` | trees in the forest, at most 10,000 (not an option in the original, which uses scikit-learn's 100) |
| `seed` | not set | seeds the forest and the sampling; not set, every run differs, as in the original |
| `additional_continuous_covariates` | `[]` | chapkit's own key: the covariates to lag, in order |

The names and defaults are the original's, so its configurations carry over: the marketplace's
`monthly_climate` (rainfall, mean temperature and relative humidity) and `monthly_selfhistory` (lagged cases
only) work unchanged. For hyperparameter search, `/api/v1/info` declares `n_target_lags` in [1, 12],
`rf_n_estimators` in [25, 300] (steps of 25), `rf_max_depth` in [2, 20] and `rf_min_samples_leaf` in [1, 20].

## Worked example

The original's example data (`example_data/historic_data.csv`, Laos, 18 provinces, January 1998 to August
2009), default options, seed 1. One province, LA-VI, has no cases at all, so 17 provinces give training rows:
2,278 rows of 24 features (18 location indicators and 6 lags). The forest has 13,414 nodes and
`scale` is **17.70** cases.

Savannakhet (LA-SV) had 0, 0, 0, 1, 13 and 34 cases from March to August 2009, the start of its season. For
September the forest predicts a mean of **61.4**, and the 100 September samples are drawn around it with
standard deviation 17.70. October's rows then hold each trajectory's own September value as the newest lag:

| month | lags of a typical trajectory | 10% | median | 90% |
|---|---|---|---|---|
| September 2009 | 0, 0, 0, 1, 13, 34 | 42.6 | 61.4 | 84.9 |
| October 2009 | 0, 0, 1, 13, 34, *≈61* | 72.8 | 101.3 | 135.8 |
| November 2009 | 0, 1, 13, 34, *≈61*, *≈101* | 106.7 | 141.4 | 172.8 |

With seeds 2 and 3 the scale is 18.00 and 17.48, September's mean 63.4 and 59.4, November's median 144.9
and 142.8. Three runs of the original on the same data give a scale of 17.58 to 17.91, September means of 58.0
to 67.0 and November medians of 135.2 to 153.4: the same model, drawn with other random numbers.

## Inputs and outputs

| | |
|---|---|
| training data | `time_period` (monthly), `location`, `disease_cases`, and the configured covariates |
| historic data | the series up to the forecast start, with the covariates (their lags reach back into it) |
| future data | the months to forecast, with the covariates |
| configuration | see [Options](#options) |
| output | `n_samples` samples per location and future month |
| fails when | a configured covariate column is missing (the message names it), a location to forecast has no history or fewer than `n_target_lags` months of it, an option is 0, `n_samples` is above 100,000, `n_target_lags` above 1,200 or `rf_n_estimators` above 10,000, a location's trajectories would hold more than 10 million values (`n_samples` times the trained `n_target_lags`), or a forecast would hold more than 250 million samples (future rows times `n_samples`) |

A location with no observed cases at all is still forecast, as in the original: its missing lags go down the
larger branch at every split, which is scikit-learn's rule when training had no missing values. A location
that was not in the training data gets all location indicators 0 (the original would fail or misalign its
columns).

## How it is checked

The baselines are checked number for number. This model cannot be: the forest's bootstrap and feature draws
and the normal draws come from random numbers, and scikit-learn's and numpy's generators are not reproduced.
So the port is checked in three parts.

**1. The features, exactly.** Everything before the random numbers is deterministic, and `tests/features.rs`
checks it against the original's own code. `scripts/gen_simple_multistep_fixtures.py` runs the original's
`transform_data`, `target_to_xarray`, `features_to_xarray` and `MultistepModel.fit_multi` on a slice of its
example data (3 provinces, 40 months, with one missing case count and one missing rainfall value) and records
the training matrix (102 rows × 18 features, with all three climate covariates) and what each forecast starts
from (the last six months of cases and the covariate features of the three future months). The port's are the
same `f64`s, gap filling included.

**2. The forest, by construction.** `src/forest.rs` follows scikit-learn's tree builder rule by rule: bootstrap
counts as weights, rows never drawn left out, the stopping rules, the order and number of features tried (a
constant feature counts as tried, but at least one varying feature is), midpoint thresholds between values more
than 1e-7 apart, the first strictly best split winning, missing values to the larger child. On the example
data its forests have 13,292 to 13,442 nodes over three seeds, the original's 13,180 to 13,434.

**3. The forecasts, by their scores.** The `compare` example backtests both models on the same splits with
chapkit's port of chap-core's evaluation (7 splits of 3 months, one training, climatology for the future
weather), the original through its own `train` and `predict` commands, and scores both alike. Each side runs
several times, the port with seeds 1 to 10 and the original unseeded, so the spread between runs of one model
shows how close two runs of the *same* model are. Mean ± standard deviation over the runs, on the original's
example data:

| configuration | model | runs | CRPS | MAE | RMSE | coverage 10-90 | coverage 25-75 |
|---|---|---|---|---|---|---|---|
| lagged cases only (default) | port | 10 seeds | 22.77 ± 0.59 | 29.31 ± 0.89 | 45.14 ± 0.80 | 0.655 ± 0.023 | 0.417 ± 0.022 |
| | original | 6 runs | 22.67 ± 0.30 | 29.11 ± 0.44 | 45.30 ± 0.39 | 0.671 ± 0.017 | 0.425 ± 0.019 |
| climate covariates (`monthly_climate`) | port | 10 seeds | 20.74 ± 0.50 | 25.02 ± 0.57 | 45.62 ± 1.06 | 0.714 ± 0.017 | 0.493 ± 0.025 |
| | original | 6 runs | 20.70 ± 0.45 | 24.98 ± 0.54 | 45.84 ± 0.72 | 0.718 ± 0.010 | 0.499 ± 0.023 |

Each run scores 334 forecasts; LA-VI, without any cases, is left out, as chap-core's backtest does. A single run of the port takes 40 to 80 ms, one
of the original about 15 s, most of it Python start-up for its 8 commands.

The differences between the two models are smaller than the spread between seeds of either. To rerun:

```sh
git clone https://github.com/chap-models/chapkit_simple_multistep_model ORIGINAL && (cd ORIGINAL && uv sync)
cargo run --release -p simple-multistep --example compare -- \
    --data ORIGINAL/example_data/historic_data.csv --original ORIGINAL --runs 10 --python-runs 6 \
    [--covariates rainfall,mean_temperature,mean_relative_humidity]
```

The binary's own backtest gives the same numbers for a seed:
`simple-multistep eval --data historic_data.csv --config '{"seed": 1}'`.

## Speed

Training on the example data (18 provinces × 140 months) and forecasting its 12 future months, best of 5, in
milliseconds (`compare --future example_data/future_data.csv`; the original in-process, without the Python
start-up):

| configuration | | train | predict |
|---|---|---|---|
| lagged cases only | original (1 core) | 101.9 | 624.9 |
| | port, 1 core | 50.6 | 74.2 |
| | port, 12 cores | 17.6 | 10.6 |
| climate covariates | original (1 core) | 235.2 | 654.5 |
| | port, 1 core | 85.9 | 69.5 |
| | port, 12 cores | 24.4 | 12.7 |

Apple M2 Max, 2026-11-10. On one core the port trains 2 to 3 times and forecasts 8 to 9 times as fast; on all
cores 6 to 10 times and 50 to 60 times.

The original uses one core (scikit-learn fits the forest with `n_jobs=None`). The port grows the trees in
parallel, and forecasts locations in parallel. The forecast is where the original is slowest: it asks the
forest for one month of 100 trajectories per call, 216 calls through pandas and skpro; the port predicts the
100 rows tree by tree in a plain loop. `make bench BENCH=simple_multistep` times train and predict on
synthetic data of the same size and of 100 locations.

## In code

```rust,ignore
for step_covariates in covariates {
    // Every trajectory's row for this month, then the forest's mean for each.
    rows.clear();
    for trajectory in &trajectories {
        let window = trajectory
            .get(trajectory.len().saturating_sub(lags)..)
            .unwrap_or_default();
        rows.extend(feature_row(step_covariates, location, n_locations, window));
    }
    let width = step_covariates.len() + n_locations + lags;
    forest.predict_rows(&rows, width, &mut means);
    // Draw each trajectory's value for the month and append it.
    let samples: Vec<f64> = means.iter().map(|&m| rng.gauss(m, scale)).collect();
    for (trajectory, &value) in trajectories.iter_mut().zip(&samples) {
        trajectory.push(value);
    }
    out.push(samples);
}
```

That is `forecast` in `src/model.rs`; the features are `training_rows` and `feature_row` in the same file, and
the forest `Forest::fit` in `src/forest.rs`.
