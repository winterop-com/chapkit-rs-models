#![allow(clippy::unwrap_used)]
//! The chapkit-rs port must reproduce chap-models' `configurable_baseline_model`: chap-core's
//! seasonal shift persistence with the original's `adjust` applied, for the default options and
//! the model's seeded configurations (fixtures made by `scripts/gen_configurable_fixtures.py`).

use baseline_configurable::{ConfigurableBaseline, Options};
use baseline_core::fixture::{
    Fixture, Match, check_parity, check_parity_on, checkpoint_round_trip,
};
use chapkit::Model;

fn fixture(case: &str) -> Fixture {
    Fixture::from_file(format!(
        "{}/fixtures/configurable_{case}.json",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

/// The option sets in the fixtures, by label (the model's `chap_seed.yaml`, plus both together).
const OPTIONS: [(&str, f64, f64); 6] = [
    ("biased_over", 1.0, 0.3),
    ("biased_under", 1.0, -0.3),
    ("point", 0.0, 0.0),
    ("narrow", 0.5, 0.0),
    ("wide", 2.0, 0.0),
    ("wide_biased", 2.0, 0.3),
];

#[test]
fn defaults_are_seasonal_shift_persistence_bit_for_bit() {
    for case in ["monthly", "weekly"] {
        check_parity(
            case,
            "seasonal_shift_persistence",
            &ConfigurableBaseline,
            &Options::default(),
            Match::Exact,
        )
        .unwrap();
        // The original computes exp(log(1 + s)) - 1 even with the defaults, which moves samples
        // by up to 4 units in the last place; this port keeps them unchanged.
        check_parity_on(
            &fixture(case),
            case,
            "default",
            &ConfigurableBaseline,
            &Options::default(),
            Match::Ulps(4),
        )
        .unwrap();
    }
}

#[test]
fn options_match_the_original() {
    for case in ["monthly", "weekly"] {
        let fx = fixture(case);
        for (label, spread_factor, log_bias) in OPTIONS {
            let config = Options {
                spread_factor,
                log_bias,
            };
            check_parity_on(
                &fx,
                case,
                label,
                &ConfigurableBaseline,
                &config,
                // log1p and expm1 come from the platform's C library: bit for bit on the platform
                // that made the fixtures (macOS), up to 3 ULP apart on Linux.
                Match::Ulps(4),
            )
            .unwrap();
        }
    }
}

#[test]
fn config_accepts_reserved_keys_and_rejects_bad_options() {
    let c: Options =
        serde_json::from_value(serde_json::json!({"prediction_periods": 3, "log_bias": 0.3}))
            .unwrap();
    assert_eq!(
        c,
        Options {
            spread_factor: 1.0,
            log_bias: 0.3
        }
    );
    let bad = Options {
        spread_factor: -1.0,
        log_bias: 0.0,
    };
    let fx = Fixture::monthly().unwrap();
    let run = chapkit::RunInfo::with_periods(3);
    let input = chapkit::TrainInput::new(&fx.train, &run, None);
    assert!(ConfigurableBaseline.train(&bad, input).is_err());
    assert_eq!(ConfigurableBaseline.validate_train(&bad, input).len(), 1);
    assert_eq!(
        ConfigurableBaseline
            .validate_train(&Options::default(), input)
            .len(),
        0
    );
}

#[test]
fn search_space_is_valid() {
    let space = ConfigurableBaseline.search_space().unwrap();
    space.validate().unwrap();
    let json = serde_json::to_value(&space).unwrap();
    assert_eq!(json["spread_factor"]["low"], 0.0);
    assert_eq!(json["log_bias"]["high"], 0.5);
}

#[test]
fn checkpoint_round_trip_keeps_predictions() {
    let config = Options {
        spread_factor: 2.0,
        log_bias: 0.3,
    };
    let ck = checkpoint_round_trip(&ConfigurableBaseline, &config).unwrap();
    assert!(ck.tensors.is_empty(), "state is stored as JSON");
}
