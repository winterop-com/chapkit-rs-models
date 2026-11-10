//! chap-core's `persistence` baseline as a chapkit-compatible service.
//!
//!     cargo run --release -p baseline-persistence            # serve on HOST:PORT (127.0.0.1:8000)
//!     cargo run --release -p baseline-persistence -- demo    # run once in-process on the chap-core fixture

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    baseline_core::run(
        "persistence",
        env!("CARGO_PKG_VERSION"),
        "Persistence (crs)",
        baseline_persistence::Persistence,
        chapkit::NoConfig {},
    )
    .await
}
