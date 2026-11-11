//! How the forecast is calculated, for one location. Plain functions over numbers; the tables,
//! locations and threads are in `lib.rs`.
//!
//! The original (chap-models' `auto_arima_chapkit`, R) does, per location:
//!
//! 1. **train** (`train.R`): `model(ARIMA(disease_cases))` on the location's monthly series. fable
//!    chooses the orders itself and estimates the coefficients:
//!    - `D` (seasonal differences, 0 or 1): 1 when the STL seasonal strength of the series is at
//!      least 0.64;
//!    - `d` (differences, 0 to 2): one more for as long as the KPSS test rejects stationarity of
//!      the series differenced so far (at 5%);
//!    - `p, q` (0 to 5), `P, Q` (0 to 2) and whether to include a constant: a stepwise search from
//!      ARIMA(2,d,2)(1,D,1), moving to neighbouring models while the AICc improves, with
//!      `p + q + P + Q <= 6`, at most two of {constant, `d`, `D`}, and no model whose AR or MA
//!      roots come within 1.01 of the unit circle;
//!    - the coefficients by maximum likelihood (started from conditional sums of squares; for
//!      long seasonal series, over 150 values or a period above 12, the search itself uses
//!      conditional sums of squares and only the chosen model is refitted by likelihood).
//!
//!    The season is 12 months; this port also takes weekly data, with a season of 52 weeks.
//!
//! 2. **predict** (`predict.R`): `refit(model, historic)` and `forecast(h)`. fable's `refit()`
//!    keeps the trained orders **and coefficients** (`reestimate = FALSE`): it only runs the model
//!    over the latest history, so the forecast starts from the last observations, and recomputes
//!    the residual variance `sigma2` on it (sum of squared residuals over `n - d - D * season`).
//!    The forecast of each step `h` is a normal distribution with the ARIMA's mean and variance
//!    (`sigma2` times the sum of the squared moving-average weights up to `h`).
//!
//! 3. **samples**: for each step, 100 independent draws `rnorm(100, mean, sd)`. Samples are not
//!    clipped at zero and steps are drawn independently, as in the original.
//!
//! The ARIMA itself (R's `stats::arima`, fable's search and forecast) is chapkit's
//! [`chapkit::ts`], checked against R there.

use chapkit::rng::Rng;
use chapkit::ts::{self, Arima, AutoArima, Forecast};

/// One location's trained model: the ARIMA fable chose for it, with its coefficients.
pub type Fitted = Arima;

/// Step 1: choose and fit the ARIMA of `values` (one per period, oldest first, `NaN` for a missing
/// period) with a season of `season` periods (12 monthly, 52 weekly), with fable's defaults.
pub fn fit(values: &[f64], season: usize) -> ts::Result<Fitted> {
    ts::auto_arima(values, &AutoArima::new(season))
}

/// Mean and standard deviation of `forecast` `h` steps ahead (`h >= 1`).
pub fn step(forecast: &Forecast, h: usize) -> Option<(f64, f64)> {
    let i = h.checked_sub(1)?;
    Some((*forecast.mean.get(i)?, forecast.variance.get(i)?.sqrt()))
}

/// Step 2: the trained model run over `history` (same orders, same coefficients, new state and
/// `sigma2`), then the mean and variance of steps `1..=h` past its end.
pub fn forecast(fitted: &Fitted, history: &[f64], h: usize) -> ts::Result<Forecast> {
    fitted.filter(history)?.forecast(h)
}

/// Step 3: `n` draws from the normal distribution with `mean` and standard deviation `sd`.
pub fn samples(mean: f64, sd: f64, n: usize, rng: &mut Rng) -> Vec<f64> {
    (0..n).map(|_| rng.gauss(mean, sd)).collect()
}
