//! chap-core's `global_median` baseline as a chapkit-compatible service.
//!
//!     cargo run --release -p baseline-global-median            # serve on HOST:PORT (127.0.0.1:8000)
//!     cargo run --release -p baseline-global-median -- demo    # run once in-process on the chap-core fixture

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    baseline_core::run(
        "global-median",
        "Global median",
        baseline_global_median::GlobalMedian,
        chapkit::NoConfig {},
    )
    .await
}
