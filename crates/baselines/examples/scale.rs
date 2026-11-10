//! Time one baseline in-process on a generated dataset of a given size. Used by bench/scale.py.
//!
//!     cargo run --release -p baselines --example scale -- <model> <locations> <periods>
//!
//! The data is deterministic and identical to bench/scale_builtin.py (chap-core's side): monthly
//! periods from 1850-01, a seasonal signal per location, 3 future periods. Prints one JSON line
//! with median milliseconds for decoding the request JSON, train and predict.

use std::time::Instant;

use baselines::{GlobalMedian, SeasonalMedian, SeasonalShiftPersistence};
use chapkit::{Column, DataFrame, Model, NoConfig, PredictInput, RunInfo, TrainInput};

const START_YEAR: usize = 1850;
const HORIZON: usize = 3;
const HORIZON_U32: u32 = 3;

fn frame(
    locs: usize,
    start: usize,
    n: usize,
    with_y: bool,
) -> Result<DataFrame, chapkit::FrameError> {
    let rows = locs * n;
    let (mut tp, mut loc) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
    let (mut rain, mut temp, mut y) = (
        Vec::with_capacity(rows),
        Vec::with_capacity(rows),
        Vec::with_capacity(rows),
    );
    for l in 0..locs {
        for p in start..start + n {
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
    if with_y {
        cols.push(("disease_cases", Column::from_f64(y)));
    }
    DataFrame::from_columns(cols)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
}

fn time<M: Model>(
    model: &M,
    config: &M::Config,
    locs: usize,
    periods: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let train = frame(locs, 0, periods, true)?;
    let future = frame(locs, periods, HORIZON, false)?;
    let run = RunInfo::with_periods(HORIZON_U32);
    let rows = train.n_rows();
    let reps = if rows <= 100_000 { 5 } else { 1 };

    // What chapkit-rs pays on top when chap-core sends the data over HTTP.
    // SCALE_NO_DECODE=1 skips the JSON step, so peak memory is the model's alone.
    let skip_decode = std::env::var_os("SCALE_NO_DECODE").is_some();
    let wire = if skip_decode {
        Vec::new()
    } else {
        serde_json::to_vec(&train)?
    };
    let (mut decode, mut fit, mut pred) = (vec![], vec![], vec![]);
    let mut out_rows = 0;
    for _ in 0..reps {
        if !skip_decode {
            let t = Instant::now();
            let decoded: DataFrame = serde_json::from_slice(&wire)?;
            decode.push(t.elapsed().as_secs_f64() * 1e3);
            drop(decoded);
        }

        let t = Instant::now();
        let trained = model.train(config, TrainInput::new(&train, &run, None))?;
        fit.push(t.elapsed().as_secs_f64() * 1e3);

        let t = Instant::now();
        let out = model.predict(
            &trained,
            config,
            PredictInput::new(&train, &future, &run, None),
        )?;
        pred.push(t.elapsed().as_secs_f64() * 1e3);
        out_rows = out.n_rows();
    }
    Ok(serde_json::json!({
        "rows": rows,
        "json_mb": wire.len() as f64 / 1e6,
        "decode_ms": median(decode),
        "train_ms": median(fit),
        "predict_ms": median(pred),
        "out_rows": out_rows,
    })
    .to_string())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (Some(model), Some(locs), Some(periods)) = (args.get(1), args.get(2), args.get(3)) else {
        return Err("usage: scale <model> <locations> <periods>".into());
    };
    let (locs, periods): (usize, usize) = (locs.parse()?, periods.parse()?);
    let line = match model.as_str() {
        "global_median" => time(&GlobalMedian, &NoConfig::default(), locs, periods)?,
        "seasonal_median" => time(&SeasonalMedian, &NoConfig::default(), locs, periods)?,
        "seasonal_shift_persistence" => time(
            &SeasonalShiftPersistence,
            &NoConfig::default(),
            locs,
            periods,
        )?,
        other => return Err(format!("unknown model {other}").into()),
    };
    println!("{line}");
    Ok(())
}
