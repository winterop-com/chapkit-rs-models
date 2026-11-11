#![allow(clippy::unwrap_used)]
//! Train and predict in-process (`make bench BENCH=auto_arima`) on the data the port is checked
//! on: Laos, 3 locations x 27 months, and Nicaragua, 4 locations x 242 weeks; 3 periods ahead.

use std::hint::black_box;

use auto_arima::{AutoArima, Config};
use chapkit::{DataFrame, Model, PredictInput, RunInfo, TrainInput};
use criterion::{Criterion, criterion_group, criterion_main};
use serde_json::{Value, json};

/// The first `n` periods of each location of a fixture as history, the next `h` as future.
fn frames(fixture: &str, n: usize, h: usize) -> (DataFrame, DataFrame) {
    let f: Value = serde_json::from_str(fixture).unwrap();
    let (mut historic, mut future) = (Vec::new(), Vec::new());
    for (location, loc) in f.as_object().unwrap() {
        let periods = loc["periods"].as_array().unwrap();
        let values = loc["values"].as_array().unwrap();
        for i in 0..n {
            historic.push(json!([periods[i], location, values[i]]));
        }
        for p in &periods[n..n + h] {
            future.push(json!([p, location]));
        }
    }
    let frame = |columns: Value, data: Vec<Value>| {
        serde_json::from_value(json!({"columns": columns, "data": data})).unwrap()
    };
    (
        frame(
            json!(["time_period", "location", "disease_cases"]),
            historic,
        ),
        frame(json!(["time_period", "location"]), future),
    )
}

fn bench(c: &mut Criterion) {
    let model = AutoArima;
    let config = Config {
        seed: Some(1),
        ..Config::default()
    };
    let run = RunInfo::with_periods(3);
    let mut group = c.benchmark_group("auto_arima");
    group.sample_size(10);
    for (label, fixture, n) in [
        (
            "monthly/3x27",
            include_str!("../fixtures/fable_monthly.json"),
            27,
        ),
        (
            "weekly/4x242",
            include_str!("../fixtures/fable_weekly.json"),
            242,
        ),
    ] {
        let (train, future) = frames(fixture, n, 3);
        let trained = model
            .train(&config, TrainInput::new(&train, &run, None))
            .unwrap();
        group.bench_function(format!("train/{label}"), |b| {
            b.iter(|| {
                model
                    .train(&config, TrainInput::new(black_box(&train), &run, None))
                    .unwrap()
            });
        });
        group.bench_function(format!("predict/{label}"), |b| {
            b.iter(|| {
                let input = PredictInput::new(&train, black_box(&future), &run, None);
                model.predict(&trained, &config, input).unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
