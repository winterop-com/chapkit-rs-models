//! chap-models' `auto_arima_chapkit` as a chapkit-compatible service.
//!
//!     cargo run --release -p auto-arima -- serve  # serve on 127.0.0.1:8000 (--host, --port)
//!     cargo run --release -p auto-arima -- eval   # backtest in-process (--data FILE)
//!     cargo run --release -p auto-arima -- --help # all commands

#[tokio::main]
async fn main() -> std::process::ExitCode {
    chapkit::Service::new("auto-arima-crs", "Auto ARIMA (crs)", auto_arima::AutoArima)
        .version(env!("CARGO_PKG_VERSION"))
        .description(
            "chapkit-rs port of chap-models' auto_arima_chapkit: an ARIMA per location with its \
             orders chosen automatically, as fable's ARIMA() chooses them, for monthly and weekly data",
        )
        .serve()
        .await
}
