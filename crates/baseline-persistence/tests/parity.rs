#![allow(clippy::unwrap_used)]
//! The chapkit-rs port must reproduce chap-core's `persistence` number for number, and survive a
//! checkpoint round trip.

use baseline_core::fixture::{Match, check_parity, checkpoint_round_trip};
use baseline_persistence::Persistence;
use chapkit::NoConfig;

/// `exp` and `ln` come from the platform's C library, for numpy as for Rust, and their last bit
/// differs between platforms: the fixtures were made on Linux, and on macOS chap-core itself is a
/// unit or two in the last place off them in 8 of the 2400 monthly samples. On the same platform
/// the port is bit for bit chap-core's (`make bench-compare`).
const LIBM_ULPS: Match = Match::Ulps(2);

#[test]
fn matches_chap_core_monthly_and_weekly() {
    check_parity(
        "monthly",
        "persistence",
        &Persistence,
        &NoConfig {},
        LIBM_ULPS,
    )
    .unwrap();
    check_parity(
        "weekly",
        "persistence",
        &Persistence,
        &NoConfig {},
        LIBM_ULPS,
    )
    .unwrap();
}

#[test]
fn checkpoint_round_trip_keeps_predictions() {
    let ck = checkpoint_round_trip(&Persistence, &NoConfig {}).unwrap();
    assert!(ck.tensors.is_empty(), "state is stored as JSON");
    assert_eq!(ck.provenance.trained_until.as_deref(), Some("2021-04"));
}
