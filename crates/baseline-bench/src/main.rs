//! Time one baseline in-process on a generated dataset of a given size. Used by bench/scale.py.
//!
//!     cargo run --release -p baseline-bench -- <model> <locations> <periods>
//!
//! The data is deterministic and identical to bench/scale_builtin.py (chap-core's side), from
//! `baseline_core::synthetic`; 3 future periods. Prints one JSON line
//! with median milliseconds for decoding the request JSON, train and predict.
//!
//! With `SCALE_DUMP=<file>`, also writes every forecast as JSON `{"location|period": [samples]}`
//! for bench/compare_outputs.py, which checks it against chap-core's.

use std::time::Instant;

use cpu_time::ProcessTime;

use baseline_core::synthetic::train_and_future;
use baseline_global_median::GlobalMedian;
use baseline_seasonal_median::SeasonalMedian;
use baseline_seasonal_shift_persistence::SeasonalShiftPersistence;
use chapkit::{DataFrame, Model, NoConfig, PredictInput, RunInfo, TrainInput};

const HORIZON: usize = 3;
const HORIZON_U32: u32 = 3;

/// Every forecast row as `"location|period" -> samples`.
fn dump(
    pred: &DataFrame,
) -> Result<serde_json::Map<String, serde_json::Value>, Box<dyn std::error::Error>> {
    let (tp, loc) = (pred.require("time_period")?, pred.require("location")?);
    let samples: Vec<Vec<f64>> = pred
        .names()
        .iter()
        .filter(|n| n.starts_with("sample_"))
        .map(|n| pred.f64_column(n))
        .collect::<Result<_, _>>()?;
    let mut out = serde_json::Map::new();
    for row in 0..pred.n_rows() {
        let key = format!(
            "{}|{}",
            loc.str_at(row).unwrap_or_default(),
            tp.str_at(row).unwrap_or_default()
        );
        let values: Vec<f64> = samples.iter().filter_map(|s| s.get(row).copied()).collect();
        out.insert(key, serde_json::json!(values));
    }
    Ok(out)
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
    let (train, future) = train_and_future(locs, periods, HORIZON)?;
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
    // CPU time of the whole process (all threads) during train and predict.
    let (mut fit_cpu, mut pred_cpu) = (vec![], vec![]);
    let mut out_rows = 0;
    let mut last: Option<DataFrame> = None;
    for _ in 0..reps {
        if !skip_decode {
            let t = Instant::now();
            let decoded: DataFrame = serde_json::from_slice(&wire)?;
            decode.push(t.elapsed().as_secs_f64() * 1e3);
            drop(decoded);
        }

        let (t, c) = (Instant::now(), ProcessTime::now());
        let trained = model.train(config, TrainInput::new(&train, &run, None))?;
        fit.push(t.elapsed().as_secs_f64() * 1e3);
        fit_cpu.push(c.elapsed().as_secs_f64() * 1e3);

        let (t, c) = (Instant::now(), ProcessTime::now());
        let out = model.predict(
            &trained,
            config,
            PredictInput::new(&train, &future, &run, None),
        )?;
        pred.push(t.elapsed().as_secs_f64() * 1e3);
        pred_cpu.push(c.elapsed().as_secs_f64() * 1e3);
        out_rows = out.n_rows();
        last = Some(out);
    }
    if let (Some(path), Some(pred)) = (std::env::var_os("SCALE_DUMP"), &last) {
        std::fs::write(path, serde_json::to_vec(&dump(pred)?)?)?;
    }
    Ok(serde_json::json!({
        "rows": rows,
        "json_mb": wire.len() as f64 / 1e6,
        "decode_ms": median(decode),
        "train_ms": median(fit),
        "predict_ms": median(pred),
        "train_cpu_ms": median(fit_cpu),
        "predict_cpu_ms": median(pred_cpu),
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
