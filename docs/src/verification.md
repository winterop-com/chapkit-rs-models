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
both fixtures and compares every one of the 100 samples of every forecast row with chap-core's, to a relative
tolerance of 1e-9. They run with `make test`. When chap-core's baselines change, `make fixtures` regenerates the
fixtures from a chap-core checkout.

## 2. The same output at scale

`make bench-compare` generates larger datasets, runs chap-core's baselines and the Rust ports on the same data,
and compares all samples of all forecast rows:

| model | sizes (locations x months) | samples compared | largest difference |
|---|---|---|---|
| global median | 100 x 120 to 5000 x 120, 100 x 1000, 1000 x 1000 | 2.2 million | 0 (bit-for-bit identical) |
| seasonal median | same | 2.2 million | 0 (bit-for-bit identical) |
| seasonal shift persistence | same | 2.2 million | 3.5e-16 relative |

The shift persistence difference is in the last bit of a floating-point number: multiplying the ratios in a
different order rounds differently. It is far below anything a metric can see.

## 3. Through chap-core itself

The packages have also been run as services registered with a real chap-core (2.6.0, through chap-devops), with
`cdx models test --backtest`; all three pass. For global median and seasonal median the backtest scores (CRPS,
MAE, RMSE, coverage) are identical to those of chap-core's built-in baselines on the same datasets (chap-core
2.6.0 has no built-in seasonal shift persistence to compare with through chap-core; the checks above cover it).
See the chapkit-rs book, *End to end with chap-core*.
