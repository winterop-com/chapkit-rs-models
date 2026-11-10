//! chap-models' `chapkit_simple_multistep_model` as a chapkit-compatible service.
//!
//!     cargo run --release -p simple-multistep -- serve  # serve on 127.0.0.1:8000 (--host, --port)
//!     cargo run --release -p simple-multistep -- eval   # backtest in-process (--data FILE)
//!     cargo run --release -p simple-multistep -- --help # all commands

#[tokio::main]
async fn main() -> std::process::ExitCode {
    chapkit::Service::new(
        "simple-multistep-crs",
        "Simple Multistep (crs)",
        simple_multistep::SimpleMultistep,
    )
    .version(env!("CARGO_PKG_VERSION"))
    .description(
        "chapkit-rs port of chap-models' chapkit_simple_multistep_model: a random forest on \
         per-location lag features with a normal spread, rolled forward one month at a time",
    )
    .serve()
    .await
}
