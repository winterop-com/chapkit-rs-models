#![allow(clippy::unwrap_used)]
//! Train and predict in-process (`make bench BENCH=simple_multistep`): the size of the original's
//! example data (18 locations x 140 months) and a larger one, without and with covariates.

use std::hint::black_box;

use baseline_core::synthetic::train_and_future;
use chapkit::{Model, PredictInput, RunInfo, TrainInput};
use criterion::{Criterion, criterion_group, criterion_main};
use simple_multistep::{Config, SimpleMultistep};

fn bench(c: &mut Criterion) {
    let model = SimpleMultistep;
    let config = Config {
        seed: Some(1),
        ..Config::default()
    };
    let mut group = c.benchmark_group("simple_multistep");
    group.sample_size(10);
    for (locations, periods) in [(18, 140), (100, 140)] {
        let (train, future) = train_and_future(locations, periods, 3).unwrap();
        for (label, covariates) in [
            ("lags", vec![]),
            (
                "covariates",
                vec!["rainfall".to_owned(), "mean_temperature".to_owned()],
            ),
        ] {
            let run = RunInfo {
                additional_continuous_covariates: covariates,
                ..RunInfo::with_periods(3)
            };
            let trained = model
                .train(&config, TrainInput::new(&train, &run, None))
                .unwrap();
            let size = format!("{label}/{locations}x{periods}");
            group.bench_function(format!("train/{size}"), |b| {
                b.iter(|| {
                    model
                        .train(&config, TrainInput::new(black_box(&train), &run, None))
                        .unwrap()
                });
            });
            group.bench_function(format!("predict/{size}"), |b| {
                b.iter(|| {
                    let input = PredictInput::new(&train, black_box(&future), &run, None);
                    model.predict(&trained, &config, input).unwrap()
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
