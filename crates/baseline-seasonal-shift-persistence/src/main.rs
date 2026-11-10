//! chap-core's `seasonal_shift_persistence` baseline as a chapkit-compatible service.
//!
//!     cargo run --release -p baseline-seasonal-shift-persistence            # serve on HOST:PORT (127.0.0.1:8000)
//!     cargo run --release -p baseline-seasonal-shift-persistence -- demo    # run once in-process on the chap-core fixture

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    baseline_core::run(
        "seasonal-shift-persistence",
        "Seasonal shift persistence",
        baseline_seasonal_shift_persistence::SeasonalShiftPersistence,
        chapkit::NoConfig {},
    )
    .await
}
