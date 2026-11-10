#![allow(clippy::unwrap_used)]
//! Train and predict in-process at a few sizes (`make bench`), with the default options (seasonal
//! shift persistence unchanged) and with both options set (the log-space adjustment).

use std::hint::black_box;

use baseline_configurable::{ConfigurableBaseline, Options};
use baseline_core::synthetic::train_and_future;
use chapkit::{Model, PredictInput, RunInfo, TrainInput};
use criterion::{Criterion, criterion_group, criterion_main};

fn bench(c: &mut Criterion) {
    let (model, run) = (ConfigurableBaseline, RunInfo::with_periods(3));
    let adjusted = Options {
        spread_factor: 2.0,
        log_bias: 0.3,
    };
    for (locations, periods) in [(5, 36), (20, 120), (500, 120)] {
        let (train, future) = train_and_future(locations, periods, 3).unwrap();
        let trained = model
            .train(&Options::default(), TrainInput::new(&train, &run, None))
            .unwrap();
        c.bench_function(
            &format!("configurable_baseline/train/{locations}x{periods}"),
            |b| {
                b.iter(|| {
                    model
                        .train(
                            &Options::default(),
                            TrainInput::new(black_box(&train), &run, None),
                        )
                        .unwrap()
                });
            },
        );
        for (label, config) in [("default", Options::default()), ("adjusted", adjusted)] {
            c.bench_function(
                &format!("configurable_baseline/predict_{label}/{locations}x{periods}"),
                |b| {
                    b.iter(|| {
                        let input = PredictInput::new(&train, black_box(&future), &run, None);
                        model.predict(&trained, &config, input).unwrap()
                    });
                },
            );
        }
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
