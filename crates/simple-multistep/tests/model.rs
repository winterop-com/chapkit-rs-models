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
