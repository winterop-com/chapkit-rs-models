#![allow(clippy::unwrap_used)]
//! Train and predict end to end, the checkpoint round trip, and the configuration.

use chapkit::{Checkpoint, DataFrame, Model, PredictInput, RunInfo, TrainInput};
use serde_json::{Value, json};
use simple_multistep::{Config, SimpleMultistep};

/// The fixture's slice of the original's example data: 3 locations, 40 months, 3 future months.
fn data() -> (DataFrame, DataFrame) {
    let f: Value = serde_json::from_str(include_str!("../fixtures/features.json")).unwrap();
    let frame = |k: &str| serde_json::from_value(f[k].clone()).unwrap();
    (frame("historic"), frame("future"))
}

fn run(covariates: &[&str]) -> RunInfo {
    RunInfo {
        additional_continuous_covariates: covariates.iter().map(|c| (*c).to_owned()).collect(),
        ..RunInfo::with_periods(3)
    }
}

fn seeded(seed: u64) -> Config {
    Config {
        seed: Some(seed),
        rf_n_estimators: 30,
        ..Config::default()
    }
}

fn forecast(config: &Config, run: &RunInfo) -> (DataFrame, Checkpoint) {
    let (historic, future) = data();
    let model = SimpleMultistep;
    let trained = model
        .train(config, TrainInput::new(&historic, run, None))
        .unwrap();
    let out = model
        .predict(
            &trained,
            config,
            PredictInput::new(&historic, &future, run, None),
        )
        .unwrap();
    (out, model.save(&trained).unwrap())
}

fn samples(out: &DataFrame, row: usize) -> Vec<f64> {
    (0..100)
        .map(|i| {
            out.column(&format!("sample_{i}"))
                .unwrap()
                .f64_at(row)
                .unwrap()
        })
        .collect()
}

#[test]
fn forecasts_every_future_row_with_spread() {
    for covariates in [
        &[][..],
        &["rainfall", "mean_temperature", "mean_relative_humidity"],
    ] {
        let (out, _) = forecast(&seeded(1), &run(covariates));
        assert_eq!(out.n_rows(), 9);
        assert_eq!(out.n_cols(), 2 + 100);
        for row in 0..out.n_rows() {
            let s = samples(&out, row);
            assert!(s.iter().all(|v| v.is_finite()));
            let (lo, hi) = s
                .iter()
                .fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
            assert!(hi > lo, "no spread in row {row}");
        }
    }
}

#[test]
fn a_seed_fixes_the_forecast() {
    let (a, _) = forecast(&seeded(7), &run(&[]));
    let (b, _) = forecast(&seeded(7), &run(&[]));
    let (c, _) = forecast(&seeded(8), &run(&[]));
    assert_eq!(samples(&a, 4), samples(&b, 4));
    assert_ne!(samples(&a, 4), samples(&c, 4));
}

#[test]
fn checkpoint_round_trip_keeps_predictions() {
    let config = seeded(3);
    let run = run(&["rainfall"]);
    let (historic, future) = data();
    let model = SimpleMultistep;
    let trained = model
        .train(&config, TrainInput::new(&historic, &run, None))
        .unwrap();
    let ck = model.save(&trained).unwrap();
    assert_eq!(ck.tensors.len(), 7, "the trees are stored as tensors");
    let loaded = model
        .load(&Checkpoint::from_bytes(&ck.to_bytes().unwrap()).unwrap())
        .unwrap();
    assert_eq!(loaded, trained);
    let predict = |t| {
        model
            .predict(
                t,
                &config,
                PredictInput::new(&historic, &future, &run, None),
            )
            .unwrap()
    };
    assert_eq!(predict(&loaded), predict(&trained));
}

#[test]
fn a_damaged_checkpoint_is_refused() {
    let (_, mut ck) = forecast(&seeded(1), &run(&[]));
    // A child pointing back at its parent would loop forever.
    let left = ck.tensors.get_mut("forest.left").unwrap();
    if let chapkit::TensorData::I64(v) = &mut left.data {
        v[0] = 0;
    }
    assert!(SimpleMultistep.load(&ck).is_err());
}

#[test]
fn a_location_without_cases_is_still_forecast() {
    // As in the original: its missing lags go down the trees' larger branches.
    let (historic, future) = data();
    let location = historic.column("location").unwrap().clone();
    let cases = historic.column("disease_cases").unwrap();
    let blanked: Vec<Option<f64>> = (0..historic.n_rows())
        .map(|r| {
            (location.str_at(r) != Some("LA-BK"))
                .then(|| cases.f64_at(r))
                .flatten()
        })
        .collect();
    let mut historic = historic.clone();
    historic
        .set_column("disease_cases", chapkit::Column::Float(blanked))
        .unwrap();
    let (config, run) = (seeded(1), run(&[]));
    let model = SimpleMultistep;
    let trained = model
        .train(&config, TrainInput::new(&historic, &run, None))
        .unwrap();
    assert_eq!(trained.state.locations.len(), 3);
    let out = model
        .predict(
            &trained,
            &config,
            PredictInput::new(&historic, &future, &run, None),
        )
        .unwrap();
    for row in 0..out.n_rows() {
        assert!(samples(&out, row).iter().all(|v| v.is_finite()));
    }
}

#[test]
fn a_missing_covariate_is_named() {
    let (historic, _) = data();
    let err = SimpleMultistep
        .train(
            &seeded(1),
            TrainInput::new(&historic, &run(&["population"]), None),
        )
        .unwrap_err();
    assert!(err.to_string().contains("population"), "{err}");
}

#[test]
fn config_keeps_the_original_names_and_defaults() {
    // A chap-core configuration of the original (with chapkit's own keys) reads unchanged.
    let config: Config = serde_json::from_value(json!({
        "prediction_periods": 3,
        "n_target_lags": 6,
        "n_samples": 100,
        "rf_max_depth": 10,
        "rf_min_samples_leaf": 5,
        "additional_continuous_covariates": ["rainfall"],
    }))
    .unwrap();
    assert_eq!(config, Config::default());
    let defaults: Config = serde_json::from_value(json!({})).unwrap();
    assert_eq!(defaults.n_target_lags, 6);
    assert_eq!(defaults.rf_n_estimators, 100);
    assert_eq!(defaults.seed, None);
    let zero = Config {
        n_target_lags: 0,
        ..seeded(1)
    };
    let (historic, _) = data();
    assert!(
        SimpleMultistep
            .train(&zero, TrainInput::new(&historic, &run(&[]), None))
            .is_err()
    );
}

#[test]
fn search_space_is_valid() {
    let space = SimpleMultistep.search_space().unwrap();
    space.validate().unwrap();
    let json = serde_json::to_value(&space).unwrap();
    for key in [
        "n_target_lags",
        "rf_n_estimators",
        "rf_max_depth",
        "rf_min_samples_leaf",
    ] {
        assert!(json.get(key).is_some(), "{key}");
    }
}

/// A configuration cannot ask for unbounded memory or time: `n_samples` is at most 100,000 and
/// `rf_n_estimators` at most 10,000 (where the configuration is read, and in `$validate`).
#[test]
fn bounds_the_samples_and_the_forest() {
    for (key, n) in [
        ("n_samples", 0),
        ("n_samples", 100_001),
        ("n_samples", 1_000_000_000),
        ("rf_n_estimators", 0),
        ("rf_n_estimators", 10_001),
    ] {
        assert!(
            serde_json::from_value::<Config>(json!({key: n})).is_err(),
            "{key} {n}"
        );
    }
    let most: Config =
        serde_json::from_value(json!({"n_samples": 100_000, "rf_n_estimators": 10_000})).unwrap();
    assert_eq!((most.n_samples, most.rf_n_estimators), (100_000, 10_000));

    let (historic, future) = data();
    let run = run(&[]);
    let train = TrainInput::new(&historic, &run, None);
    let predict = PredictInput::new(&historic, &future, &run, None);
    let model = SimpleMultistep;
    let huge = Config {
        n_samples: 1_000_000_000,
        ..seeded(1)
    };
    let forest = Config {
        rf_n_estimators: 1_000_000_000,
        ..seeded(1)
    };
    assert_eq!(model.validate_train(&huge, train).len(), 1);
    assert_eq!(model.validate_train(&forest, train).len(), 1);
    assert_eq!(model.validate_predict(&huge, predict).len(), 1);
    assert!(model.train(&forest, train).is_err());
    let trained = model.train(&seeded(1), train).unwrap();
    assert!(model.predict(&trained, &huge, predict).is_err());
    assert_eq!(model.validate_predict(&seeded(1), predict).len(), 0);
}

/// One location's `n` months of cases from 1900-01, and the month after as the future.
fn long_series(n: usize) -> (DataFrame, DataFrame) {
    let month = |i: usize| format!("{}-{:02}", 1900 + i / 12, i % 12 + 1);
    let historic = (0..n)
        .map(|i| json!([month(i), "A", (i % 12) as f64 + i as f64 / 100.0]))
        .collect::<Vec<_>>();
    let frame = |columns: Value, data: Vec<Value>| {
        serde_json::from_value(json!({"columns": columns, "data": data})).unwrap()
    };
    (
        frame(
            json!(["time_period", "location", "disease_cases"]),
            historic,
        ),
        frame(
            json!(["time_period", "location"]),
            vec![json!([month(n), "A"])],
        ),
    )
}

/// The trajectories a forecast keeps are bounded before forecasting: `n_target_lags` is at most
/// 1,200 (a century of months), and `n_samples` times the trained model's `n_target_lags` at
/// most 10 million values per location.
#[test]
fn bounds_the_lags_and_the_trajectories() {
    for n in [0, 1_201, 1_000_000_000] {
        assert!(
            serde_json::from_value::<Config>(json!({"n_target_lags": n})).is_err(),
            "{n}"
        );
    }
    let most: Config = serde_json::from_value(json!({"n_target_lags": 1_200})).unwrap();
    assert_eq!(most.n_target_lags, 1_200);

    let (historic, future) = long_series(1_010);
    let run = run(&[]);
    let train = TrainInput::new(&historic, &run, None);
    let predict = PredictInput::new(&historic, &future, &run, None);
    let model = SimpleMultistep;
    let lags = |n_target_lags, n_samples| Config {
        n_target_lags,
        n_samples,
        rf_n_estimators: 2,
        ..seeded(1)
    };
    assert_eq!(model.validate_train(&lags(1_201, 100), train).len(), 1);
    assert_eq!(
        model.validate_predict(&lags(1_000, 10_001), predict).len(),
        1
    );
    assert_eq!(
        model.validate_predict(&lags(1_000, 10_000), predict).len(),
        0
    );

    // The trained model's lags count, whatever the configuration says.
    let trained = model.train(&lags(1_000, 100), train).unwrap();
    assert_eq!(trained.state.n_target_lags, 1_000);
    let err = model
        .predict(&trained, &lags(6, 10_001), predict)
        .unwrap_err();
    assert!(err.to_string().contains("10000000"), "{err}");
    let out = model.predict(&trained, &lags(6, 10_000), predict).unwrap();
    assert_eq!(out.n_rows(), 1);
    assert!(out.has("sample_9999"));
}

/// `rf_min_samples_leaf` up to `usize::MAX` trains (trees of a single leaf), without overflow.
#[test]
fn a_huge_min_samples_leaf_trains() {
    let (historic, future) = data();
    let run = run(&[]);
    let config: Config =
        serde_json::from_value(json!({"rf_min_samples_leaf": 9_223_372_036_854_775_808u64, "seed": 1, "rf_n_estimators": 3}))
            .unwrap();
    let model = SimpleMultistep;
    let trained = model
        .train(&config, TrainInput::new(&historic, &run, None))
        .unwrap();
    assert!(trained.forest.trees.iter().all(|t| t.nodes.len() == 1));
    let out = model
        .predict(
            &trained,
            &config,
            PredictInput::new(&historic, &future, &run, None),
        )
        .unwrap();
    assert_eq!(out.n_rows(), 9);
}
