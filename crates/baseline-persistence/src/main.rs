//! chap-core's `persistence` baseline as a chapkit-compatible service.
//!
//!     cargo run --release -p baseline-persistence -- serve  # serve on 127.0.0.1:8000 (--host, --port)
//!     cargo run --release -p baseline-persistence -- demo   # run once in-process on the chap-core fixture
//!     cargo run --release -p baseline-persistence -- --help # all commands

#[tokio::main]
async fn main() -> std::process::ExitCode {
    baseline_core::run(
        "persistence",
        env!("CARGO_PKG_VERSION"),
        "Persistence (crs)",
        baseline_persistence::Persistence,
        chapkit::NoConfig {},
    )
    .await
}
