#![allow(clippy::unwrap_used)]
//! The chapkit-rs port must reproduce chap-core's `seasonal_shift_persistence` number for number, and survive a
//! checkpoint round trip.

use baseline_core::fixture::{Match, check_parity, checkpoint_round_trip};
use baseline_seasonal_shift_persistence::SeasonalShiftPersistence;
use chapkit::NoConfig;

#[test]
fn matches_chap_core_monthly_and_weekly() {
    check_parity(
        "monthly",
        "seasonal_shift_persistence",
        &SeasonalShiftPersistence,
        &NoConfig {},
        Match::Exact,
    )
    .unwrap();
    check_parity(
        "weekly",
        "seasonal_shift_persistence",
        &SeasonalShiftPersistence,
        &NoConfig {},
        Match::Exact,
    )
    .unwrap();
}

/// Short series where ratios run out, so the fallbacks for missing ratios decide the forecast:
/// `single` is one location with January and February [1, 3], forecast two months (the second
/// has no ratio at horizon 2 and must use the single ratio 2 at horizon 1, giving 7, not 3).
#[test]
fn matches_chap_core_when_ratios_run_out() {
    for case in ["single", "fallbacks"] {
        check_parity(
            case,
            "seasonal_shift_persistence",
            &SeasonalShiftPersistence,
            &NoConfig {},
            Match::Exact,
        )
        .unwrap();
    }
}

#[test]
fn checkpoint_round_trip_keeps_predictions() {
    let ck = checkpoint_round_trip(&SeasonalShiftPersistence, &NoConfig {}).unwrap();
    assert!(ck.tensors.is_empty(), "state is stored as JSON");
    assert_eq!(ck.provenance.trained_until.as_deref(), Some("2021-04"));
}
