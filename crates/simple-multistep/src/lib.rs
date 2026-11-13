//! chap-models' `chapkit_simple_multistep_model` as a chapkit [`Model`].
//!
//! **The calculation is in [`model`]** (`src/model.rs`) and the random forest in [`forest`]
//! (`src/forest.rs`). This file connects them to chapkit: the configuration, reading the tables
//! onto a period grid per location, seeding, and the checkpoint (the trees as tensors).

pub mod forest;
pub mod model;

use std::collections::{BTreeMap, BTreeSet};

use chapkit::rng::Rng;
use chapkit::{
    AssessedStatus, Checkpoint, Column, DataFrame, Diagnostic, Error, Model, ModelInfo,
    ModelMetadata, Param, Period, PeriodType, PredictInput, Result, SearchSpace, Tensor,
    TensorData, TrainInput, cols, locations::predict_by_location,
};
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use forest::{Columns, Forest, ForestParams, Node, Split, Tree};
use model::COVARIATE_LAGS;

/// The model's options: the original's names and defaults, so a chap-core configuration of the
/// original works unchanged, plus `rf_n_estimators` and `seed`. `prediction_periods` and
/// `additional_continuous_covariates` are chapkit's own keys (the covariates come from the latter).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Config {
    /// Number of lagged target values to feed the one-step regressor.
    #[serde(default = "default_n_target_lags", deserialize_with = "n_target_lags")]
    #[schemars(range(min = 1, max = MAX_N_TARGET_LAGS))]
    pub n_target_lags: usize,
    /// Number of trajectory samples drawn per location at predict time.
    #[serde(default = "default_n_samples", deserialize_with = "n_samples")]
    #[schemars(range(min = 1, max = MAX_N_SAMPLES))]
    pub n_samples: usize,
    /// Max depth of the underlying RandomForest regressor.
    #[serde(default = "default_rf_max_depth")]
    pub rf_max_depth: usize,
    /// Minimum samples per leaf for the underlying RandomForest regressor.
    #[serde(default = "default_rf_min_samples_leaf")]
    pub rf_min_samples_leaf: usize,
    /// Number of trees in the RandomForest regressor (fixed at scikit-learn's 100 in the original).
    #[serde(
        default = "default_rf_n_estimators",
        deserialize_with = "rf_n_estimators"
    )]
    #[schemars(range(min = 1, max = MAX_RF_N_ESTIMATORS))]
    pub rf_n_estimators: usize,
    /// Seed for the forest and the sampling; random when not set, as in the original.
    #[serde(default)]
    pub seed: Option<u64>,
}

fn default_n_target_lags() -> usize {
    6
}
fn default_n_samples() -> usize {
    100
}
fn default_rf_max_depth() -> usize {
    10
}
fn default_rf_min_samples_leaf() -> usize {
    5
}
fn default_rf_n_estimators() -> usize {
    100
}

/// Most trajectories per location: a thousand times the default (the original's 100), well past
/// any use of the samples, so that a configuration cannot ask for unbounded memory.
pub const MAX_N_SAMPLES: usize = 100_000;

/// Most trees: a hundred times scikit-learn's 100 and over thirty times the top of the search
/// space (300). Time and memory grow with each tree; depth and leaf size need no bound, since a
/// tree never has more nodes than twice its training rows.
pub const MAX_RF_N_ESTIMATORS: usize = 10_000;

/// Most target lags: a century of months, a hundred times the top of the search space (12).
/// Training rows and forecast trajectories grow with it.
pub const MAX_N_TARGET_LAGS: usize = 1_200;

/// Most values one location's trajectories may hold (`n_samples` times the trained model's
/// `n_target_lags`): 80 MB per location being forecast. The default 100 samples take 600; the
/// most samples (100,000) take up to 100 lags, the most lags (1,200) up to 8,333 samples.
pub const MAX_TRAJECTORY_VALUES: usize = 10_000_000;

/// Why `n_samples` trajectories of `n_target_lags` values are too many, if they are.
fn trajectory_problem(n_samples: usize, n_target_lags: usize) -> Option<String> {
    (n_samples.saturating_mul(n_target_lags) > MAX_TRAJECTORY_VALUES).then(|| {
        format!(
            "{n_samples} samples of n_target_lags = {n_target_lags} are more than \
             {MAX_TRAJECTORY_VALUES} trajectory values per location"
        )
    })
}

/// Most samples one forecast may hold (future rows times `n_samples`): 2 GB of numbers. 100,000
/// locations at 12 months and the default 100 samples take 120 million.
pub const MAX_FORECAST_SAMPLES: usize = 250_000_000;

/// Why `value` of option `name` is not in `1..=max`, if it is not.
fn out_of_bounds(name: &str, value: usize, max: usize) -> Option<String> {
    (!(1..=max).contains(&value))
        .then(|| format!("{name} must be between 1 and {max}, got {value}"))
}

/// Reads a bounded option, rejecting it out of bounds as the configuration is stored.
fn bounded<'de, D: serde::Deserializer<'de>>(
    d: D,
    name: &str,
    max: usize,
) -> std::result::Result<usize, D::Error> {
    let value = usize::deserialize(d)?;
    out_of_bounds(name, value, max).map_or(Ok(value), |p| Err(serde::de::Error::custom(p)))
}

fn n_samples<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<usize, D::Error> {
    bounded(d, "n_samples", MAX_N_SAMPLES)
}

fn n_target_lags<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<usize, D::Error> {
    bounded(d, "n_target_lags", MAX_N_TARGET_LAGS)
}

fn rf_n_estimators<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<usize, D::Error> {
    bounded(d, "rf_n_estimators", MAX_RF_N_ESTIMATORS)
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_target_lags: default_n_target_lags(),
            n_samples: default_n_samples(),
            rf_max_depth: default_rf_max_depth(),
            rf_min_samples_leaf: default_rf_min_samples_leaf(),
            rf_n_estimators: default_rf_n_estimators(),
            seed: None,
        }
    }
}

impl Config {
    /// Why the configuration cannot be used, if it cannot; with `future_rows`, also for a
    /// forecast of that many rows.
    fn problem(&self, future_rows: Option<usize>) -> Option<String> {
        let bounds = [
            ("n_target_lags", self.n_target_lags, MAX_N_TARGET_LAGS),
            ("n_samples", self.n_samples, MAX_N_SAMPLES),
            ("rf_max_depth", self.rf_max_depth, usize::MAX),
            ("rf_min_samples_leaf", self.rf_min_samples_leaf, usize::MAX),
            ("rf_n_estimators", self.rf_n_estimators, MAX_RF_N_ESTIMATORS),
        ];
        bounds
            .into_iter()
            .find_map(|(name, value, max)| out_of_bounds(name, value, max))
            .or_else(|| trajectory_problem(self.n_samples, self.n_target_lags))
            .or_else(|| {
                let rows = future_rows?;
                (rows.saturating_mul(self.n_samples) > MAX_FORECAST_SAMPLES).then(|| {
                    format!(
                        "{rows} future rows of {} samples are more than {MAX_FORECAST_SAMPLES} samples",
                        self.n_samples
                    )
                })
            })
    }

    fn diagnostics(&self, future_rows: Option<usize>) -> Vec<Diagnostic> {
        self.problem(future_rows)
            .map(|p| Diagnostic::error("invalid_option", p))
            .into_iter()
            .collect()
    }

    fn check(&self, future_rows: Option<usize>) -> Result<()> {
        self.problem(future_rows)
            .map_or(Ok(()), |p| Err(Error::Config(p)))
    }

    /// The seed of this call: the configured one, else a fresh one.
    fn seed(&self) -> u64 {
        self.seed.unwrap_or_else(|| Rng::from_time().next_u64())
    }
}

/// The trained model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trained {
    /// Everything but the trees.
    pub state: State,
    /// The forest predicting next month's cases.
    pub forest: Forest,
}

/// The trained model's settings and spread (the checkpoint's JSON state).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct State {
    /// Training locations, sorted: the order of the location indicators.
    pub locations: Vec<String>,
    /// Covariates, in the order of their features.
    pub covariates: Vec<String>,
    /// Target lags per row.
    pub n_target_lags: usize,
    /// Features per row.
    pub n_features: usize,
    /// Standard deviation of each forecast step (the mean absolute training residual).
    pub scale: f64,
    /// Training rows.
    pub n_rows: usize,
}

/// The model.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleMultistep;

impl Model for SimpleMultistep {
    type Config = Config;
    type Trained = Trained;

    fn info(&self) -> ModelInfo {
        ModelInfo {
            period_type: PeriodType::Monthly,
            min_prediction_periods: 1,
            max_prediction_periods: 100,
            allow_free_additional_continuous_covariates: true,
            required_covariates: Vec::new(),
            model_metadata: ModelMetadata {
                author: Some("Morten Hansen".into()),
                author_note: Some(
                    "chapkit-rs port of chap-models' chapkit_simple_multistep_model (HISP Oslo and \
                     HISP West and Central Africa): a random forest on per-location lag features, \
                     a normal spread from its mean absolute residual, rolled forward one month at a \
                     time. Fits on lagged cases alone unless covariates are configured."
                        .into(),
                ),
                author_assessed_status: Some(AssessedStatus::Orange),
                contact_email: Some("morten@winterop.com".into()),
                repository_url: Some("https://github.com/winterop-com/chapkit-rs-models".into()),
                documentation_url: Some(
                    "https://winterop-com.github.io/chapkit-rs-models/models/simple-multistep.html"
                        .into(),
                ),
                ..ModelMetadata::default()
            },
            ..ModelInfo::default()
        }
    }

    fn search_space(&self) -> Option<SearchSpace> {
        let int = |low, high, step| Param::Int {
            low,
            high,
            step,
            log: false,
        };
        Some(
            SearchSpace::new()
                .with("n_target_lags", int(1, 12, 1))
                .with("rf_n_estimators", int(25, 300, 25))
                .with("rf_max_depth", int(2, 20, 1))
                .with("rf_min_samples_leaf", int(1, 20, 1)),
        )
    }

    fn validate_train(&self, config: &Config, _: TrainInput<'_>) -> Vec<Diagnostic> {
        config.diagnostics(None)
    }

    fn validate_predict(&self, config: &Config, input: PredictInput<'_>) -> Vec<Diagnostic> {
        config.diagnostics(Some(input.future.n_rows()))
    }

    fn train(&self, config: &Config, input: TrainInput<'_>) -> Result<Trained> {
        config.check(None)?;
        let covariates = input.run.additional_continuous_covariates.clone();
        let target = self.info().target;
        let (names, rows) = training_set(input.data, &target, &covariates, config.n_target_lags)?;
        let (x, y): (Vec<Vec<f64>>, Vec<f64>) = rows.into_iter().map(|r| (r.x, r.y)).unzip();
        if x.is_empty() {
            return Err(Error::invalid(format!(
                "no location has more than n_target_lags = {} months of data",
                config.n_target_lags
            )));
        }
        let params = ForestParams {
            n_estimators: config.rf_n_estimators,
            max_depth: config.rf_max_depth,
            min_samples_leaf: config.rf_min_samples_leaf,
        };
        let forest = Forest::fit(&Columns::from_rows(&x), &y, params, config.seed());
        let scale = model::residual_scale(&forest, &x, &y);
        Ok(Trained {
            state: State {
                n_features: covariates.len() * COVARIATE_LAGS + names.len() + config.n_target_lags,
                locations: names,
                covariates,
                n_target_lags: config.n_target_lags,
                scale,
                n_rows: x.len(),
            },
            forest,
        })
    }

    fn predict(
        &self,
        trained: &Trained,
        config: &Config,
        input: PredictInput<'_>,
    ) -> Result<DataFrame> {
        config.check(Some(input.future.n_rows()))?;
        let state = &trained.state;
        // The trained model's lags, which a checkpoint may hold beyond the configuration's.
        if let Some(p) = trajectory_problem(config.n_samples, state.n_target_lags) {
            return Err(Error::Config(p));
        }
        let target = self.info().target;
        let tables = PredictTables::read(input.historic, input.future, &target, &state.covariates)?;
        let seed = config.seed();
        predict_by_location(input.future, config.n_samples, |location, rows| {
            let periods: Vec<Period> = rows.iter().map(|r| r.1).collect();
            let (previous, covariates) = tables.inputs(location, &periods, state.n_target_lags)?;
            let position = state
                .locations
                .binary_search_by(|l| l.as_str().cmp(location))
                .ok();
            let mut rng = Rng::new(seed ^ fnv1a(location));
            let by_step = model::forecast(
                &trained.forest,
                state.scale,
                &covariates,
                position,
                state.locations.len(),
                &previous,
                config.n_samples,
                &mut rng,
            );
            Ok(by_step)
        })
    }

    /// The trees as tensors, the rest as JSON state. Node `i` of the forest (trees one after the
    /// other) has `feature[i]` (-1 for a leaf), `threshold[i]`, `left[i]`/`right[i]` (node indices
    /// within its tree, -1 for a leaf), `missing_left[i]` (1 when a missing value goes left) and
    /// `value[i]`; tree `t` holds nodes
    /// `tree_start[t]..tree_start[t + 1]`.
    fn save(&self, trained: &Trained) -> Result<Checkpoint> {
        let nodes: Vec<&Node> = trained.forest.trees.iter().flat_map(|t| &t.nodes).collect();
        let ints = |f: fn(&Node) -> i64| {
            Tensor::new(
                vec![nodes.len()],
                TensorData::I64(nodes.iter().map(|n| f(n)).collect()),
            )
        };
        let floats = |f: fn(&Node) -> f64| {
            Tensor::new(
                vec![nodes.len()],
                TensorData::F64(nodes.iter().map(|n| f(n)).collect()),
            )
        };
        let starts: Vec<i64> = std::iter::once(0)
            .chain(trained.forest.trees.iter().scan(0, |total, tree| {
                *total += tree.nodes.len();
                Some(index(*total))
            }))
            .collect();
        Ok(Checkpoint::from_state(&trained.state)?
            .with_tensor(
                "forest.feature",
                ints(|n| n.split.map_or(-1, |s| index(s.feature)))?,
            )
            .with_tensor(
                "forest.threshold",
                floats(|n| n.split.map_or(f64::NAN, |s| s.threshold))?,
            )
            .with_tensor(
                "forest.left",
                ints(|n| n.split.map_or(-1, |s| index(s.left)))?,
            )
            .with_tensor(
                "forest.right",
                ints(|n| n.split.map_or(-1, |s| index(s.right)))?,
            )
            .with_tensor(
                "forest.missing_left",
                ints(|n| n.split.map_or(0, |s| i64::from(s.missing_left)))?,
            )
            .with_tensor("forest.value", floats(|n| n.value)?)
            .with_tensor(
                "forest.tree_start",
                Tensor::new(vec![starts.len()], TensorData::I64(starts))?,
            ))
    }

    fn load(&self, checkpoint: &Checkpoint) -> Result<Trained> {
        let state: State = checkpoint.state()?;
        let ints = |name: &str| -> Result<Vec<i64>> {
            match &checkpoint.tensor(name)?.data {
                TensorData::I64(v) => Ok(v.clone()),
                _ => Err(Error::Serde(format!("tensor '{name}' is not i64"))),
            }
        };
        let (feature, left, right, missing_left, starts) = (
            ints("forest.feature")?,
            ints("forest.left")?,
            ints("forest.right")?,
            ints("forest.missing_left")?,
            ints("forest.tree_start")?,
        );
        let threshold = checkpoint.tensor("forest.threshold")?.to_f64();
        let value = checkpoint.tensor("forest.value")?.to_f64();
        let bad = || Error::Serde("malformed forest in checkpoint".into());
        let node = |i: usize| -> Result<Node> {
            let f = *feature.get(i).ok_or_else(bad)?;
            let v = *value.get(i).ok_or_else(bad)?;
            if f < 0 {
                return Ok(Node {
                    split: None,
                    value: v,
                });
            }
            let to = |x: Option<&i64>| x.and_then(|x| usize::try_from(*x).ok()).ok_or_else(bad);
            Ok(Node {
                split: Some(Split {
                    feature: to(Some(&f))?,
                    threshold: *threshold.get(i).ok_or_else(bad)?,
                    left: to(left.get(i))?,
                    right: to(right.get(i))?,
                    missing_left: *missing_left.get(i).ok_or_else(bad)? != 0,
                }),
                value: v,
            })
        };
        let trees = starts
            .windows(2)
            .map(|w| {
                let (a, b) = match w {
                    [a, b] => (
                        usize::try_from(*a).map_err(|_| bad())?,
                        usize::try_from(*b).map_err(|_| bad())?,
                    ),
                    _ => return Err(bad()),
                };
                let tree = Tree {
                    nodes: (a..b).map(node).collect::<Result<_>>()?,
                };
                if tree.is_well_formed(state.n_features) {
                    Ok(tree)
                } else {
                    Err(bad())
                }
            })
            .collect::<Result<Vec<_>>>()?;
        if trees.is_empty() {
            return Err(bad());
        }
        Ok(Trained {
            state,
            forest: Forest { trees },
        })
    }
}

/// A training row: its location and month, features and cases.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingRow {
    pub location: String,
    pub period: Period,
    pub x: Vec<f64>,
    pub y: f64,
}

/// The training locations (sorted) and the rows the forest is fitted on: every location's rows in
/// location order, months in order, as the original stacks them.
pub fn training_set(
    data: &DataFrame,
    target: &str,
    covariates: &[String],
    n_target_lags: usize,
) -> Result<(Vec<String>, Vec<TrainingRow>)> {
    let table = Table::read(data, Some(target), covariates)?;
    if table.locations.is_empty() {
        return Err(Error::invalid("no training data"));
    }
    let names: Vec<String> = table.locations.keys().cloned().collect();
    let located: Vec<(&String, &LocationRows)> = table.locations.iter().collect();
    let per_location: Vec<Vec<TrainingRow>> = located
        .par_iter()
        .enumerate()
        .map(|(j, (name, rows))| {
            let (cases, features) = rows.on_grid(&table.grid);
            model::training_rows(&features, &cases, j, names.len(), n_target_lags)
                .into_iter()
                .filter_map(|(t, x, y)| {
                    Some(TrainingRow {
                        location: (*name).clone(),
                        period: *table.grid.get(t)?,
                        x,
                        y,
                    })
                })
                .collect()
        })
        .collect();
    Ok((names, per_location.into_iter().flatten().collect()))
}

/// The tables a forecast reads: the history (for the cases) and the history followed by the
/// future (for the covariate features, which reach back across the boundary).
#[derive(Debug, Clone)]
pub struct PredictTables {
    historic: Table,
    both: Table,
}

impl PredictTables {
    /// Read the historic and future tables.
    pub fn read(
        historic: &DataFrame,
        future: &DataFrame,
        target: &str,
        covariates: &[String],
    ) -> Result<Self> {
        let historic = Table::read(historic, Some(target), covariates)?;
        let future = Table::read(future, None, covariates)?;
        let both = historic.joined(&future);
        Ok(Self { historic, both })
    }

    /// What a location's forecast starts from: its last `n_target_lags` months of cases (gaps
    /// filled, oldest first), and the covariate features of each of `periods`.
    pub fn inputs(
        &self,
        location: &str,
        periods: &[Period],
        n_target_lags: usize,
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
        let history =
            self.historic.locations.get(location).ok_or_else(|| {
                Error::invalid(format!("location '{location}' has no historic data"))
            })?;
        let (cases, _) = history.on_grid(&self.historic.grid);
        let previous = cases
            .get(cases.len().saturating_sub(n_target_lags)..)
            .unwrap_or_default()
            .to_vec();
        // A location without any observed cases keeps missing lags; the trees send them down
        // their larger branches, as in the original.
        if previous.len() < n_target_lags {
            return Err(Error::invalid(format!(
                "location '{location}' needs n_target_lags = {n_target_lags} months of history"
            )));
        }
        let (_, features) = self
            .both
            .locations
            .get(location)
            .map_or_default(|r| r.on_grid(&self.both.grid));
        let covariates = periods
            .iter()
            .map(|period| {
                let t = self.both.grid.binary_search(period).ok();
                t.and_then(|t| features.get(t)).cloned().ok_or_else(|| {
                    Error::invalid(format!("no covariates for '{location}' at {period}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((previous, covariates))
    }
}

/// A node index as a tensor element.
fn index(i: usize) -> i64 {
    i64::try_from(i).unwrap_or(-1)
}

/// A 64-bit FNV-1a hash: gives each location its own random stream, whatever the order or
/// thread it is forecast in.
fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// One location's rows, sorted by period.
#[derive(Debug, Clone, Default)]
pub struct LocationRows {
    pub periods: Vec<Period>,
    /// Cases per row (missing when the table has no target).
    pub cases: Vec<f64>,
    /// Per covariate, its value per row.
    pub covariates: Vec<Vec<f64>>,
}

impl LocationRows {
    /// The location on the period `grid`: its gap-filled cases, and per grid month its covariate
    /// features (for each covariate, the values 1..=3 of the location's rows back; gap-filled).
    pub fn on_grid(&self, grid: &[Period]) -> (Vec<f64>, Vec<Vec<f64>>) {
        let at: Vec<Option<usize>> = self
            .periods
            .iter()
            .map(|p| grid.binary_search(p).ok())
            .collect();
        let place = |values: &mut dyn Iterator<Item = f64>| {
            let mut out = vec![f64::NAN; grid.len()];
            for (pos, v) in at.iter().zip(values) {
                if let Some(slot) = pos.and_then(|t| out.get_mut(t)) {
                    *slot = v;
                }
            }
            model::fill_gaps(&mut out);
            out
        };
        let cases = place(&mut self.cases.iter().copied());
        let mut features =
            vec![Vec::with_capacity(self.covariates.len() * COVARIATE_LAGS); grid.len()];
        for values in &self.covariates {
            let lags = model::covariate_lags(values);
            for k in 0..COVARIATE_LAGS {
                let column = place(&mut lags.iter().map(|l| l.get(k).copied().unwrap_or(f64::NAN)));
                for (row, v) in features.iter_mut().zip(column) {
                    row.push(v);
                }
            }
        }
        (cases, features)
    }
}

/// A table split by location, with the grid of all its periods.
#[derive(Debug, Clone, Default)]
pub struct Table {
    pub grid: Vec<Period>,
    pub locations: BTreeMap<String, LocationRows>,
}

impl Table {
    /// Read `target` (if any) and `covariates` from `df`. Fails, naming them, on missing
    /// covariate columns.
    pub fn read(df: &DataFrame, target: Option<&str>, covariates: &[String]) -> Result<Self> {
        let missing: Vec<&str> = covariates
            .iter()
            .filter(|c| !df.has(c))
            .map(String::as_str)
            .collect();
        if !missing.is_empty() {
            return Err(Error::invalid(format!(
                "data is missing configured covariate columns: {}; set \
                 `additional_continuous_covariates` to covariates the data carries, or leave it \
                 empty to fit on lagged case history alone",
                missing.join(", ")
            )));
        }
        let cases = match target {
            Some(t) => Some(df.f64_column(t)?),
            None => None,
        };
        let values = covariates
            .iter()
            .map(|c| df.f64_column(c))
            .collect::<Result<Vec<_>, _>>()?;
        let time = df.require(cols::TIME_PERIOD)?;
        let mut grid = BTreeSet::new();
        let mut locations = BTreeMap::new();
        for (location, idx) in df.group_indices(cols::LOCATION)? {
            let mut rows = idx
                .into_iter()
                .map(|i| Ok((period_at(time, i)?, i)))
                .collect::<Result<Vec<_>>>()?;
            rows.sort_by_key(|r| r.0);
            let get = |v: &[f64], i: usize| v.get(i).copied().unwrap_or(f64::NAN);
            let loc = LocationRows {
                periods: rows.iter().map(|r| r.0).collect(),
                cases: rows
                    .iter()
                    .map(|&(_, i)| cases.as_deref().map_or(f64::NAN, |c| get(c, i)))
                    .collect(),
                covariates: values
                    .iter()
                    .map(|v| rows.iter().map(|&(_, i)| get(v, i)).collect())
                    .collect(),
            };
            grid.extend(loc.periods.iter().copied());
            locations.insert(location, loc);
        }
        Ok(Self {
            grid: grid.into_iter().collect(),
            locations,
        })
    }

    /// This table's rows followed by `other`'s, per location (the history and the future, for the
    /// covariate features).
    fn joined(&self, other: &Table) -> Table {
        let mut locations = self.locations.clone();
        for (name, rows) in &other.locations {
            let entry = locations.entry(name.clone()).or_default();
            let mut merged: Vec<(Period, f64, Vec<f64>)> = Vec::new();
            for src in [&*entry, rows] {
                for (i, p) in src.periods.iter().enumerate() {
                    let values = src
                        .covariates
                        .iter()
                        .map(|c| c.get(i).copied().unwrap_or(f64::NAN))
                        .collect();
                    merged.push((*p, src.cases.get(i).copied().unwrap_or(f64::NAN), values));
                }
            }
            merged.sort_by_key(|m| m.0);
            let n_cov = merged.first().map_or(0, |m| m.2.len());
            *entry = LocationRows {
                periods: merged.iter().map(|m| m.0).collect(),
                cases: merged.iter().map(|m| m.1).collect(),
                covariates: (0..n_cov)
                    .map(|c| {
                        merged
                            .iter()
                            .map(|m| m.2.get(c).copied().unwrap_or(f64::NAN))
                            .collect()
                    })
                    .collect(),
            };
        }
        let grid: BTreeSet<Period> = self.grid.iter().chain(&other.grid).copied().collect();
        Table {
            grid: grid.into_iter().collect(),
            locations,
        }
    }
}

fn period_at(col: &Column, row: usize) -> Result<Period> {
    let raw = col.str_at(row).unwrap_or_default();
    Period::parse(raw)
        .ok_or_else(|| Error::invalid(format!("expected monthly periods, got '{raw}'")))
}
