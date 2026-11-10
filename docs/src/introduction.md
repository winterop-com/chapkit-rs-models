# Introduction

This book describes the model packages in `chapkit-rs-models`: what each model does, how its forecast is
calculated, worked examples with real numbers, and how the Rust versions are checked against the originals.

Every model here is a **model package**: a Rust crate that implements one forecasting model on top of
[chapkit-rs](https://github.com/winterop-com/chapkit-rs) and builds into one binary. The binary serves the
chapkit HTTP API, so chap-core registers and runs it like any other chapkit model service.

| package | model | in short |
|---|---|---|
| `baseline-global-median` | [Global median](models/global-median.md) | every future period gets all of the location's past observations |
| `baseline-seasonal-median` | [Seasonal median](models/seasonal-median.md) | a future month gets the same month in earlier years |
| `baseline-seasonal-shift-persistence` | [Seasonal shift persistence](models/seasonal-shift-persistence.md) | the last value, scaled by how cases changed at that time of year before |
| `baseline-persistence` | [Persistence](models/persistence.md) | the last value, with a spread from how cases changed over the same distance before |
| `baseline-configurable` | [Configurable baseline](models/configurable-baseline.md) | seasonal shift persistence, made too high, too low, too narrow or too wide by a known amount (chap-models' `configurable_baseline_model`) |
| `simple-multistep` | [Simple multistep](models/simple-multistep.md) | a random forest predicts next month from the last months and the location; sampled predictions are fed back for the months after (chap-models' `chapkit_simple_multistep_model`) |

The first four are ports of chap-core's built-in **baselines** (`chap_core/models/builtin`); the configurable
baseline is a port of a Chap model built on one of them, for testing evaluation metrics. Baselines are the
yardstick for real models: a model that cannot beat "the same month last years" is not adding anything. They
make no assumptions beyond the data and fit nothing, which also makes them good for checking a setup end to end.

Simple multistep is a forecasting model proper, the Chap marketplace's reference model. It fits a random
forest and draws random numbers, so it cannot match its original sample for sample: its features are checked
against the original's exactly, and its forecasts by backtest scores, which agree within the spread between
two runs of either model ([details](models/simple-multistep.md#how-it-is-checked)).

**Same numbers as chap-core.** Each port is checked against chap-core's own output, sample by sample, both on
chap-core's test data and at scale up to millions of forecast samples ([details](verification.md)).

**Faster, and less memory.** The ports are 2 to 24 times faster than chap-core's Python versions on one core,
more on several, and use less memory at every size ([details](performance.md)).

## Who this book is for

- **New here?** Start with the tutorials: [set up your machine](tutorials/setup.md), [run a baseline and read
  its forecast](tutorials/first-forecast.md), [run them through chap-core](tutorials/chap-core.md), and [run them
  in Docker](tutorials/docker.md).
- **To know how a forecast is calculated**, read [How a forecast is made](forecasts.md) and the page of the
  model. Each page explains the method in words and formulas, with a worked example; the code in each
  package's `src/model.rs` follows the same steps.
- **To run a model**, see [Running a model](running.md).
- **To write a new model package**, see [How a model package is laid out](layout.md) and the chapkit-rs book.
