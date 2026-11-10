//! chap-models' `configurable_baseline_model` as a chapkit-compatible service.
//!
//!     cargo run --release -p baseline-configurable -- serve  # serve on 127.0.0.1:8000 (--host, --port)
//!     cargo run --release -p baseline-configurable -- demo   # run once in-process on the chap-core fixture
//!     cargo run --release -p baseline-configurable -- --help # all commands

#[tokio::main]
async fn main() -> std::process::ExitCode {
    baseline_core::run_described(
        "configurable-baseline",
        env!("CARGO_PKG_VERSION"),
        "Configurable baseline (crs)",
        "chapkit-rs port of chap-models' configurable_baseline_model: seasonal shift persistence \
         with a configurable spread and bias in log(1 + cases) space",
        baseline_configurable::ConfigurableBaseline,
        baseline_configurable::Options::default(),
    )
    .await
}
