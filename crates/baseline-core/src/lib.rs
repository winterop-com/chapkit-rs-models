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

/// Model info shared by the baselines.
pub fn info(note: &str) -> ModelInfo {
    ModelInfo {
        model_metadata: ModelMetadata {
            author: Some("Morten Hansen".into()),
            author_note: Some(note.into()),
            author_assessed_status: Some(AssessedStatus::Green),
            contact_email: Some("morten@winterop.com".into()),
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
