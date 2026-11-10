#![allow(clippy::unwrap_used)]
//! The chapkit-rs port must reproduce chap-core's `global_median` number for number, and survive a
//! checkpoint round trip.

use baseline_core::fixture::{Match, check_parity, checkpoint_round_trip};
use baseline_global_median::GlobalMedian;
use chapkit::NoConfig;

#[test]
fn matches_chap_core_monthly_and_weekly() {
    check_parity(
        "monthly",
        "global_median",
        &GlobalMedian,
        &NoConfig {},
        Match::Exact,
    )
    .unwrap();
    check_parity(
        "weekly",
        "global_median",
        &GlobalMedian,
        &NoConfig {},
        Match::Exact,
    )
    .unwrap();
}

#[test]
fn checkpoint_round_trip_keeps_predictions() {
    let ck = checkpoint_round_trip(&GlobalMedian, &NoConfig {}).unwrap();
    assert!(ck.tensors.is_empty(), "state is stored as JSON");
    assert_eq!(ck.provenance.trained_until.as_deref(), Some("2021-04"));
}
