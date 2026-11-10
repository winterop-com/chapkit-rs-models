#![allow(clippy::unwrap_used)]
//! Train and predict in-process at a few sizes (`make bench`).

use std::hint::black_box;

use baseline_core::synthetic::train_and_future;
use baseline_global_median::GlobalMedian;
use chapkit::{Model, NoConfig, PredictInput, RunInfo, TrainInput};
use criterion::{Criterion, criterion_group, criterion_main};

fn bench(c: &mut Criterion) {
    let (model, config, run) = (GlobalMedian, NoConfig {}, RunInfo::with_periods(3));
    for (locations, periods) in [(5, 36), (20, 120), (500, 120)] {
        let (train, future) = train_and_future(locations, periods, 3).unwrap();
        let trained = model
            .train(&config, TrainInput::new(&train, &run, None))
            .unwrap();
        c.bench_function(&format!("global_median/train/{locations}x{periods}"), |b| {
            b.iter(|| {
                model
                    .train(&config, TrainInput::new(black_box(&train), &run, None))
                    .unwrap()
            });
        });
        c.bench_function(
            &format!("global_median/predict/{locations}x{periods}"),
            |b| {
                b.iter(|| {
                    let input = PredictInput::new(&train, black_box(&future), &run, None);
                    model.predict(&trained, &config, input).unwrap()
                });
            },
        );
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
