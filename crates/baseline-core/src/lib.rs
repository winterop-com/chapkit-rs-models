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

/// A baseline binary's `main`: serve `model` as a chapkit service on `HOST:PORT`, or, with the
/// argument `demo`, train and predict once in-process on the chap-core fixture and print it.
pub async fn run<M: Model>(
    id: &str,
    title: &str,
    model: M,
    config: M::Config,
) -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().nth(1).as_deref() == Some("demo") {
        return fixture::demo(id, &model, &config);
    }
    chapkit::Service::new(id, title, model)
        .description(format!(
            "Rust port of chap-core's {} baseline",
            id.replace('-', "_")
        ))
        .serve()
        .await?;
    Ok(())
}
