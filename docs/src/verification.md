# Same output as chap-core

The ports must produce chap-core's numbers, not just similar ones. Three checks make sure they do.

## 1. Parity tests on chap-core's own output

`scripts/gen_baseline_fixtures.py` runs chap-core's built-in baselines on two small datasets and stores the
data and chap-core's predictions in `crates/baseline-core/fixtures/`:

| fixture | data | locations | to forecast |
|---|---|---|---|
| `baselines_monthly.json` | January 2018 to April 2021, with a few missing months | A, B, C in training, plus D only in the historic data | 6 months, May to October 2021 |
| `baselines_weekly.json` | weekly from 2019 week 40 | A, B, C | the following weeks |

This is the **example data** used throughout this book. Each package's `tests/parity.rs` trains and predicts on
both fixtures and compares every one of the 100 samples of every forecast row with chap-core's, bit for bit
(the same `f64`, not just a close one). The quantile step does numpy's arithmetic in numpy's order, so the samples
are chap-core's exactly. Persistence is the one exception: it calls `exp` and `ln`, which numpy and Rust both take
from the platform's C library, and their last bit differs between macOS and Linux, so chap-core's own output does
too. Its test allows two units in the last place, the gap between chap-core on macOS and the fixtures made on
Linux; on the same platform it is exact. They run with `make test`. When chap-core's baselines change,
`make fixtures` regenerates the fixtures from a chap-core checkout.

## 2. The same output at scale

`make bench-compare` generates larger datasets (100 x 120 to 5000 x 120 and 1000 x 1000 locations x months), runs
chap-core's baselines and the chapkit-rs ports on the same data, and compares all samples of all forecast rows, bit for
bit: 2.2 million samples per model, all identical. A second check used weekly data built to provoke rounding
differences: 200 locations x 300 weeks, with many zero counts (ties), missing values and locations whose last
value is missing, 160,000 samples per model:

| model | not bit-identical before 1.0.3 | largest difference before | not bit-identical since |
|---|---|---|---|
| global median | 0 | 0 | 0 |
| seasonal median | 349 | 1.4e-14 (2.2e-16 relative) | 0 |
| seasonal shift persistence | 2,328 | 2.8e-14 (5.7e-16 relative) | 0 |
| persistence | 98 | 2.8e-14 (5.0e-16 relative) | 0 |

Before 1.0.3 the ports interpolated between two values as `a + (b - a) * t`; numpy computes `b - (b - a) * (1 - t)`
when `t` is at least 0.5. The two can round differently in the last bit. In a backtest through chap-core that
flipped one tie between a seasonal shift persistence sample and the observation in about 350,000 samples. The ports now
interpolate as numpy does.

## 3. Through chap-core itself

The packages have also been run as services registered with a real chap-core (2.6.0, through chap-devops), with
`cdx models test --backtest`; all three pass (persistence, added later for chap-core 2.7.0, has not been run
this way yet). For global median and seasonal median the backtest scores (CRPS,
MAE, RMSE, coverage) are identical to those of chap-core's built-in baselines on the same datasets (chap-core
2.6.0 has no built-in seasonal shift persistence to compare with through chap-core; the checks above cover it).
See the chapkit-rs book, *End to end with chap-core*.
