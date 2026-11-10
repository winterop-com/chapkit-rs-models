//! chap-core's `global_median` baseline as a chapkit-compatible service.
//!
//!     cargo run --release -p baseline-global-median -- serve  # serve on 127.0.0.1:8000 (--host, --port)
//!     cargo run --release -p baseline-global-median -- demo   # run once in-process on the chap-core fixture
//!     cargo run --release -p baseline-global-median -- --help # all commands

#[tokio::main]
async fn main() -> std::process::ExitCode {
    baseline_core::run(
        "global-median",
        env!("CARGO_PKG_VERSION"),
        "Global median (crs)",
        baseline_global_median::GlobalMedian,
        chapkit::NoConfig {},
    )
    .await
}
