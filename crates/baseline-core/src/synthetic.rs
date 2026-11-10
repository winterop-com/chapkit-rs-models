//! Deterministic synthetic data for benchmarks: monthly periods from 1850-01 and a seasonal
//! signal per location. `bench/scale_builtin.py` builds the identical data for chap-core.

use chapkit::{Column, DataFrame, FrameError};

const START_YEAR: usize = 1850;

/// `locations` x `periods` rows starting at period index `start`; with `disease_cases` when
/// `with_target`.
pub fn frame(
    locations: usize,
    start: usize,
    periods: usize,
    with_target: bool,
) -> Result<DataFrame, FrameError> {
    let rows = locations * periods;
    let (mut tp, mut loc) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
    let (mut rain, mut temp, mut y) = (
        Vec::with_capacity(rows),
        Vec::with_capacity(rows),
        Vec::with_capacity(rows),
    );
    for l in 0..locations {
        for p in start..start + periods {
            tp.push(format!("{}-{:02}", START_YEAR + p / 12, p % 12 + 1));
            loc.push(format!("loc{l}"));
            let season = ((p % 12) as f64 / 12.0 * std::f64::consts::TAU).sin();
            rain.push(100.0 + 50.0 * season + l as f64);
            temp.push(25.0 + 3.0 * season);
            y.push((20.0 + 10.0 * season + 2.0 * (l % 50) as f64 + (p % 7) as f64).round());
        }
    }
    let mut cols = vec![
        ("time_period", Column::from_strs(tp)),
        ("location", Column::from_strs(loc)),
        ("rainfall", Column::from_f64(rain)),
        ("mean_temperature", Column::from_f64(temp)),
    ];
    if with_target {
        cols.push(("disease_cases", Column::from_f64(y)));
    }
    DataFrame::from_columns(cols)
}

/// Training data (`locations` x `periods`) and the `horizon` future periods after it.
pub fn train_and_future(
    locations: usize,
    periods: usize,
    horizon: usize,
) -> Result<(DataFrame, DataFrame), FrameError> {
    Ok((
        frame(locations, 0, periods, true)?,
        frame(locations, periods, horizon, false)?,
    ))
}
