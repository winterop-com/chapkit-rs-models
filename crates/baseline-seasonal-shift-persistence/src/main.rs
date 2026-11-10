//! chap-core's `seasonal_shift_persistence` baseline as a chapkit-compatible service.
//!
//!     cargo run --release -p baseline-seasonal-shift-persistence -- serve  # serve on 127.0.0.1:8000 (--host, --port)
//!     cargo run --release -p baseline-seasonal-shift-persistence -- demo   # run once in-process on the chap-core fixture
//!     cargo run --release -p baseline-seasonal-shift-persistence -- --help # all commands

#[tokio::main]
async fn main() -> std::process::ExitCode {
    baseline_core::run(
        "seasonal-shift-persistence",
        env!("CARGO_PKG_VERSION"),
        "Seasonal shift persistence (crs)",
        baseline_seasonal_shift_persistence::SeasonalShiftPersistence,
        chapkit::NoConfig {},
    )
    .await
}
