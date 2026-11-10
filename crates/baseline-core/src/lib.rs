//! Shared by the chap-core baseline ports (`baseline-*` crates).
//!
//! - [`quantiles`]: the calculation all baselines share, turning observed values into forecast
//!   samples (chap-core's `empirical.py`).
//! - [`fixture`]: chap-core's own baseline outputs, and the parity check against them.
//! - [`run`]: serve a model, or run it once in-process with `demo`.
//! - [`synthetic`]: deterministic data of any size, for benchmarks.
//!
//! Splitting data per location and building the prediction table are not baseline-specific;
//! they live in chapkit (`chapkit::locations`).

pub mod fixture;
pub mod quantiles;
pub mod synthetic;

use chapkit::{AssessedStatus, Model, ModelInfo, ModelMetadata};

pub use quantiles::N_SAMPLES;

/// The models book, where each baseline has a page.
const BOOK: &str = "https://winterop-com.github.io/chapkit-rs-models";

/// Model info shared by the baselines. `page` is the model's page in the book (`global-median`).
///
/// The status is `red`, as chap-core rates its own built-in baselines: they are yardsticks to
/// compare real models with, not forecasting models.
pub fn info(note: &str, page: &str) -> ModelInfo {
    ModelInfo {
        model_metadata: ModelMetadata {
            author: Some("Morten Hansen".into()),
            author_note: Some(format!(
                "Rust port (chapkit-rs) of chap-core's built-in baseline. {note}"
            )),
            author_assessed_status: Some(AssessedStatus::Red),
            contact_email: Some("morten@winterop.com".into()),
            repository_url: Some("https://github.com/winterop-com/chapkit-rs-models".into()),
            documentation_url: Some(format!("{BOOK}/models/{page}.html")),
            ..ModelMetadata::default()
        },
        ..ModelInfo::default()
    }
}

/// A baseline binary's command line: chapkit's subcommands (`serve`, `test`, `healthcheck`) and
/// `demo`.
#[derive(clap::Parser)]
#[command(arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    #[command(flatten)]
    Chapkit(chapkit::cli::Command),
    /// Run once on the chap-core fixture
    Demo,
}

/// A baseline binary's `main`: chapkit's `serve`, `test` and `healthcheck`, or `demo`, which trains
/// and predicts once in-process on the chap-core fixture and prints it. Errors are one line
/// `error: ...` and exit code 1, as for every chapkit binary.
///
/// `version` is the binary's crate version. chap-core stores a model template once per version,
/// so every published change to a model needs a new version.
pub async fn run<M: Model + Clone>(
    id: &str,
    version: &str,
    title: &str,
    model: M,
    config: M::Config,
) -> std::process::ExitCode {
    let service = chapkit::Service::new(id, title, model.clone())
        .version(version)
        .description(format!(
            "Rust port of chap-core's {} baseline",
            id.replace('-', "_")
        ));
    let cli: Cli = service.parse_args();
    let result = match cli.command {
        Command::Demo => fixture::demo(id, &model, &config),
        Command::Chapkit(command) => service.run(command).await.map_err(Into::into),
    };
    chapkit::cli::exit_code(result)
}
