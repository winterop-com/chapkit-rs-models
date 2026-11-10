#![allow(clippy::unwrap_used)]
//! The Rust port must reproduce chap-core's `seasonal_median` number for number, and survive a
//! checkpoint round trip.

use baseline_core::fixture::{check_parity, checkpoint_round_trip};
use baseline_seasonal_median::SeasonalMedian;
use chapkit::NoConfig;

#[test]
fn matches_chap_core_monthly_and_weekly() {
    check_parity("monthly", "seasonal_median", &SeasonalMedian, &NoConfig {}).unwrap();
    check_parity("weekly", "seasonal_median", &SeasonalMedian, &NoConfig {}).unwrap();
}

#[test]
fn checkpoint_round_trip_keeps_predictions() {
    let ck = checkpoint_round_trip(&SeasonalMedian, &NoConfig {}).unwrap();
    assert!(ck.tensors.is_empty(), "state is stored as JSON");
    assert_eq!(ck.provenance.trained_until.as_deref(), Some("2021-04"));
}
