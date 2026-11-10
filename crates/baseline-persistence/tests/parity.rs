#![allow(clippy::unwrap_used)]
//! The Rust port must reproduce chap-core's `persistence` number for number, and survive a
//! checkpoint round trip.

use baseline_core::fixture::{check_parity, checkpoint_round_trip};
use baseline_persistence::Persistence;
use chapkit::NoConfig;

#[test]
fn matches_chap_core_monthly_and_weekly() {
    check_parity("monthly", "persistence", &Persistence, &NoConfig {}).unwrap();
    check_parity("weekly", "persistence", &Persistence, &NoConfig {}).unwrap();
}

#[test]
fn checkpoint_round_trip_keeps_predictions() {
    let ck = checkpoint_round_trip(&Persistence, &NoConfig {}).unwrap();
    assert!(ck.tensors.is_empty(), "state is stored as JSON");
    assert_eq!(ck.provenance.trained_until.as_deref(), Some("2021-04"));
}
