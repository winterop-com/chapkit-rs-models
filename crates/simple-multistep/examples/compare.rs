//! Backtest this port and the original side by side on the same splits, and score both the same
//! way. The original is driven through its own `train` and `predict` commands (its uv
//! environment), on the training, historic and future tables chapkit's backtest hands each model.
//! Neither model's sampling can be matched draw for draw, so each side runs several times (the
//! port with seeds 1..=runs, the original unseeded) to show the seed-to-seed spread.
//!
//!     cargo run --release -p simple-multistep --example compare -- \
//!         --data ORIGINAL/example_data/historic_data.csv --original ORIGINAL \
//!         [--covariates rainfall,mean_temperature,mean_relative_humidity] [--runs 3] [--python-runs 3]
//!
//! `ORIGINAL` is a checkout of chap-models/chapkit_simple_multistep_model after `uv sync`.
//!
//! With `--future FILE` it times one training on `--data` and one forecast of `FILE` instead (best
//! of five; the original in-process, without its interpreter's start-up), on one core and on all.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::Mutex;
use std::time::Instant;

use chapkit::eval::{Climatology, Metric, Options, Scores, backtest};
use chapkit::{DataFrame, Error, Model, PredictInput, RunInfo, TrainInput};
use serde_json::{Value, json};
use simple_multistep::{Config, SimpleMultistep};

type Res<T> = Result<T, String>;

const SHOWN: [Metric; 5] = [
    Metric::Crps,
    Metric::Mae,
    Metric::Rmse,
    Metric::Coverage10_90,
    Metric::Coverage25_75,
];

/// The original's training and forecast, timed in its own process: best of `reps`, in seconds.
const PYTHON_TIMING: &str = r"
import asyncio, json, sys, time
import pandas as pd
from chapkit.data import DataFrame
from chapkit_simple_multistep_model.config import MultistepConfig
from chapkit_simple_multistep_model.predict import on_predict
from chapkit_simple_multistep_model.train import on_train

config = MultistepConfig.model_validate(json.loads(sys.argv[1]))
historic = DataFrame.from_pandas(pd.read_csv(sys.argv[2]))
future = DataFrame.from_pandas(pd.read_csv(sys.argv[3]))
train, predict = [], []
for _ in range(int(sys.argv[4])):
    t = time.perf_counter(); model = asyncio.run(on_train(config, historic)); train.append(time.perf_counter() - t)
    t = time.perf_counter(); asyncio.run(on_predict(config, model, historic, future)); predict.append(time.perf_counter() - t)
print(min(train), min(predict))
";

struct Args {
    data: PathBuf,
    original: PathBuf,
    future: Option<PathBuf>,
    covariates: Vec<String>,
    runs: u64,
    python_runs: u64,
}

impl Args {
    fn parse() -> Res<Self> {
        let args: Vec<String> = std::env::args().collect();
        let get = |name: &str| {
            args.iter()
                .position(|a| a == name)
                .and_then(|i| args.get(i + 1).cloned())
        };
        let count =
            |name: &str| get(name).map_or(Ok(3), |r| r.parse().map_err(|e| format!("{name}: {e}")));
        Ok(Self {
            data: get("--data").ok_or("--data FILE is required")?.into(),
            original: get("--original")
                .ok_or("--original DIR is required")?
                .into(),
            future: get("--future").map(PathBuf::from),
            covariates: get("--covariates")
                .map(|c| c.split(',').map(str::to_owned).collect())
                .unwrap_or_default(),
            runs: count("--runs")?,
            python_runs: count("--python-runs")?,
        })
    }
}

fn read(path: &Path) -> Res<DataFrame> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    DataFrame::from_csv(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn write_csv(df: &DataFrame, path: &Path) -> chapkit::Result<()> {
    let mut out = df.names().join(",");
    for row in 0..df.n_rows() {
        out.push('\n');
        let cells: Vec<String> = df
            .iter()
            .map(|(_, c)| match c.json_at(row) {
                Value::Null => String::new(),
                Value::String(s) => s,
                v => v.to_string(),
            })
            .collect();
        out.push_str(&cells.join(","));
    }
    std::fs::write(path, out).map_err(|e| Error::model(e.to_string()))
}

/// Run one of the original's commands; fails with its stderr.
fn run_original(command: &Path, args: &[&Path]) -> chapkit::Result<String> {
    let flags = match command.file_name().and_then(|n| n.to_str()) {
        Some("train") => &["--data", "--model", "--config"][..],
        _ => &["--historic", "--future", "--output", "--model", "--config"][..],
    };
    let mut cmd = Command::new(command);
    for (flag, value) in flags.iter().zip(args) {
        cmd.arg(flag).arg(value);
    }
    let out = cmd.output().map_err(|e| Error::model(e.to_string()))?;
    if !out.status.success() {
        return Err(Error::model(
            String::from_utf8_lossy(&out.stderr).into_owned(),
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn line(label: &str, scores: &Scores, seconds: f64) -> String {
    let mut s = format!("{label:<12}");
    for m in SHOWN {
        let _ = write!(s, "{:>16.3}", scores.get(m));
    }
    let _ = write!(s, "{seconds:>12.2}");
    s
}

fn summary(label: &str, runs: &[Scores]) -> String {
    let mut s = format!("{label:<12}");
    for m in SHOWN {
        let v: Vec<f64> = runs.iter().map(|r| r.get(m)).collect();
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len().max(2) - 1) as f64;
        let _ = write!(s, "{:>16}", format!("{mean:.3} +- {:.3}", var.sqrt()));
    }
    s
}

/// The port's backtests, seeds 1..=runs.
fn port_runs(args: &Args, data: &DataFrame, run: &RunInfo) -> Res<Vec<Scores>> {
    let model = SimpleMultistep;
    let mut all = Vec::new();
    for seed in 1..=args.runs {
        let config = Config {
            seed: Some(seed),
            ..Config::default()
        };
        let started = Instant::now();
        let bt = backtest(
            data,
            &Options::default(),
            &Climatology,
            |d| model.train(&config, TrainInput::new(d, run, None)),
            |t, h, f| model.predict(t, &config, PredictInput::new(h, f, run, None)),
        )
        .map_err(|e| e.to_string())?;
        let scores = bt.score(data).map_err(|e| e.to_string())?;
        let label = format!("crs seed {seed}");
        println!("{}", line(&label, &scores, started.elapsed().as_secs_f64()));
        all.push(scores);
    }
    Ok(all)
}

/// The original's backtests, through its `train` and `predict` commands, one process at a time.
fn python_runs(args: &Args, data: &DataFrame, dir: &Path) -> Res<Vec<Scores>> {
    let config_path = dir.join("config.yml");
    // JSON is YAML: the original's options at their defaults, and the covariates.
    let config = json!({
        "prediction_periods": Options::default().periods,
        "n_target_lags": 6, "n_samples": 100, "rf_max_depth": 10, "rf_min_samples_leaf": 5,
        "additional_continuous_covariates": args.covariates,
    });
    std::fs::write(&config_path, config.to_string()).map_err(|e| e.to_string())?;
    let bin = |name: &str| args.original.join(".venv/bin").join(name);
    let lock = Mutex::new(0usize);
    let mut all = Vec::new();
    for r in 1..=args.python_runs {
        let started = Instant::now();
        let bt = backtest(
            data,
            &Options::default(),
            &Climatology,
            |d| {
                let _guard = lock.lock().map_err(|e| Error::model(e.to_string()))?;
                let (train, model) = (dir.join("train.csv"), dir.join(format!("model{r}.pickle")));
                write_csv(d, &train)?;
                run_original(&bin("train"), &[&train, &model, &config_path])?;
                Ok(model)
            },
            |model, h, f| {
                let mut n = lock.lock().map_err(|e| Error::model(e.to_string()))?;
                *n += 1;
                let paths = ["historic", "future", "out"].map(|k| dir.join(format!("{k}{n}.csv")));
                let [historic, future, output] = &paths;
                write_csv(h, historic)?;
                write_csv(f, future)?;
                run_original(
                    &bin("predict"),
                    &[historic, future, output, model, &config_path],
                )?;
                read(output).map_err(Error::model)
            },
        )
        .map_err(|e| e.to_string())?;
        let scores = bt.score(data).map_err(|e| e.to_string())?;
        let label = format!("python {r}");
        println!("{}", line(&label, &scores, started.elapsed().as_secs_f64()));
        all.push(scores);
    }
    Ok(all)
}

/// Best of `reps` trainings and forecasts of the port on `threads` threads, in seconds.
fn time_port(
    data: &DataFrame,
    future: &DataFrame,
    run: &RunInfo,
    threads: usize,
    reps: usize,
) -> Res<(f64, f64)> {
    let model = SimpleMultistep;
    let config = Config::default();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| e.to_string())?;
    pool.install(|| {
        let (mut train, mut predict) = (f64::MAX, f64::MAX);
        for _ in 0..reps {
            let t = Instant::now();
            let trained = model
                .train(&config, TrainInput::new(data, run, None))
                .map_err(|e| e.to_string())?;
            train = train.min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            model
                .predict(
                    &trained,
                    &config,
                    PredictInput::new(data, future, run, None),
                )
                .map_err(|e| e.to_string())?;
            predict = predict.min(t.elapsed().as_secs_f64());
        }
        Ok((train, predict))
    })
}

fn timing(args: &Args, data: &DataFrame, future_path: &Path, run: &RunInfo) -> Res<()> {
    const REPS: usize = 5;
    let future = read(future_path)?;
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    let one = time_port(data, &future, run, 1, REPS)?;
    let all = time_port(data, &future, run, cores, REPS)?;
    let config =
        json!({ "additional_continuous_covariates": run.additional_continuous_covariates });
    let out = Command::new(args.original.join(".venv/bin/python"))
        .args(["-c", PYTHON_TIMING, &config.to_string()])
        .arg(&args.data)
        .arg(future_path)
        .arg(REPS.to_string())
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let python: Vec<f64> = text
        .split_whitespace()
        .filter_map(|v| v.parse().ok())
        .collect();
    let [py_train, py_predict] = python[..] else {
        return Err(format!("unexpected timing output: {text}"));
    };
    println!("{:<24}{:>12}{:>12}", "best of 5, ms", "train", "predict");
    let row = |label: &str, (train, predict): (f64, f64)| {
        println!("{label:<24}{:>12.1}{:>12.1}", train * 1e3, predict * 1e3);
    };
    row("original (1 core)", (py_train, py_predict));
    row("crs, 1 core", one);
    row(&format!("crs, {cores} cores"), all);
    Ok(())
}

fn compare(args: &Args) -> Res<()> {
    let data = read(&args.data)?;
    let options = Options::default();
    let run = RunInfo {
        additional_continuous_covariates: args.covariates.clone(),
        ..RunInfo::with_periods(u32::try_from(options.periods).unwrap_or(u32::MAX))
    };
    if let Some(future) = &args.future {
        return timing(args, &data, future, &run);
    }
    let mut header = format!("{:<12}", "run");
    for m in SHOWN {
        let _ = write!(header, "{:>16}", m.name());
    }
    let covariates = if args.covariates.is_empty() {
        "none".to_owned()
    } else {
        args.covariates.join(", ")
    };
    println!(
        "{} splits of {} months, {} training, climatology weather; covariates: {covariates}\n\
         {header}{:>12}",
        options.splits, options.periods, options.retrain, "seconds"
    );
    let port = port_runs(args, &data, &run)?;
    let dir = std::env::temp_dir().join(format!("simple-multistep-compare-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let python = python_runs(args, &data, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    let python = python?;
    println!(
        "\n{header}\n{}\n{}",
        summary("crs", &port),
        summary("python", &python)
    );
    Ok(())
}

fn main() -> ExitCode {
    match Args::parse().and_then(|args| compare(&args)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
