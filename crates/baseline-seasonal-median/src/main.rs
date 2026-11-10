//! chap-core's `seasonal_median` baseline as a chapkit-compatible service.
//!
//!     cargo run --release -p baseline-seasonal-median            # serve on HOST:PORT (127.0.0.1:8000)
//!     cargo run --release -p baseline-seasonal-median -- demo    # run once in-process on the chap-core fixture

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    baseline_core::run(
        "seasonal-median",
        "Seasonal median (crs)",
        baseline_seasonal_median::SeasonalMedian,
        chapkit::NoConfig {},
    )
    .await
}
