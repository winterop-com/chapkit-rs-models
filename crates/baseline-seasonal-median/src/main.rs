//! chap-core's `seasonal_median` baseline as a chapkit-compatible service.
//!
//!     cargo run --release -p baseline-seasonal-median -- serve  # serve on 127.0.0.1:8000 (--host, --port)
//!     cargo run --release -p baseline-seasonal-median -- demo   # run once in-process on the chap-core fixture
//!     cargo run --release -p baseline-seasonal-median -- --help # all commands

#[tokio::main]
async fn main() -> std::process::ExitCode {
    baseline_core::run(
        "seasonal-median",
        env!("CARGO_PKG_VERSION"),
        "Seasonal median (crs)",
        baseline_seasonal_median::SeasonalMedian,
        chapkit::NoConfig {},
    )
    .await
}
