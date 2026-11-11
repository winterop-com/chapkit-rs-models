# chapkit-rs-models

Model packages for [chapkit-rs](https://github.com/winterop-com/chapkit-rs). Each crate is laid out the way
a real model project would be: it depends on `chapkit`, implements the `Model` trait, and gets the full chapkit-compatible HTTP
API (so chap-core can use it unchanged) from `chapkit::Service`.

The library is expected as a sibling checkout: clone `chapkit-rs` next to this repository
(`chap-workspace/rs/chapkit-rs` and `chap-workspace/rs/chapkit-rs-models`).

Each model package is a crate under `crates/`; the root holds the workspace, the HTTP benchmark (`bench/`)
and helper scripts. The ports of chap-core's baselines are prefixed `baseline-`, and each is checked against
chap-core number for number.

**The book**, online at <https://winterop-com.github.io/chapkit-rs-models/> (source in `docs/`, `make docs`),
describes every model in full: the method, worked examples with
real numbers, figures, fallbacks, and how the ports are checked against chap-core.

**How a forecast is calculated** is in each package's `src/model.rs`: plain, commented functions over numbers,
with no tables, threads or HTTP (for example `crates/baseline-seasonal-shift-persistence/src/model.rs`).
`src/lib.rs` connects that calculation to chapkit, and `src/main.rs` is the binary.

| crate | what |
|---|---|
| `crates/baseline-global-median` | chap-core's `global_median`: every period forecast from all of a location's observations. |
| `crates/baseline-seasonal-median` | chap-core's `seasonal_median`: each period forecast from the same month or week in earlier years. |
| `crates/baseline-seasonal-shift-persistence` | chap-core's `seasonal_shift_persistence`: the last value, scaled by how cases changed over the same horizon and season before. |
| `crates/baseline-persistence` | chap-core's `persistence` (version 2): the last value, with a spread from how cases changed over the same horizon before. |
| `crates/baseline-configurable` | chap-models' `configurable_baseline_model`: seasonal shift persistence with a configurable spread and bias in log space, a yardstick for metrics. |
| `crates/simple-multistep` | chap-models' `chapkit_simple_multistep_model`, the marketplace's reference model: a random forest on per-location lag features with a normal spread, rolled forward one month at a time. Checked against the original's features exactly and its backtest scores within seed-to-seed noise. |
| `crates/auto-arima` | chap-models' `auto_arima_chapkit` (R, fable): per location an automatic ARIMA (orders chosen as `fable::ARIMA()` chooses them, through chapkit-rs's `chapkit::ts`), refitted on the history, normal samples; monthly and weekly. Checked against fable's orders and forecasts (`tests/fable.rs`) and the original's backtest scores. |
| `crates/baseline-core` | Shared by the baselines: the quantile step, chap-core's outputs for the parity tests, the demo runner, synthetic data. |
| `crates/baseline-bench` | Times the baselines at any data size, for `make bench-scale`. |

```sh
make release                     # binaries into ./bin
bin/baseline-global-median serve # serve on 127.0.0.1:8000 (--host, --port, --database)
bin/baseline-global-median test  # in another terminal: chap-core's flow against it
bin/baseline-seasonal-median demo  # run once in-process and time it
make check                       # lint + tests (parity with chap-core, checkpoints, HTTP flow)
make fixtures                    # regenerate the golden outputs from a chap-core checkout
make bench                       # criterion: the models in-process
make bench-http                  # the same model through Python chapkit and chapkit-rs over HTTP
make bench-scale                 # chap-core's built-in baselines vs the chapkit-rs ports as data grows
make release FEATURES=turso      # binaries that also accept turso: database URLs
cargo build --features polars    # DataFrame::to_polars / from_polars for pandas-style models
```

## Images

Published for linux/amd64 and linux/arm64:
`ghcr.io/winterop-com/chapkit-rs-models/baseline-global-median`, `.../baseline-seasonal-median`,
`.../baseline-seasonal-shift-persistence`, `.../baseline-persistence`, `.../baseline-configurable`,
`.../simple-multistep` and `.../auto-arima` (tags `<version>`, `latest`,
`sha-<commit>`). `make docker-up` builds and runs them locally with `compose.yml`.

An image is published once per version, the workspace `version` in `Cargo.toml`. chap-core treats a model
version as write-once and refuses a new build under a version it has already seen, so a push to `main` only
publishes images whose version is not yet published: **bump the version to publish a change** (the workflow
warns when there is nothing new to publish).

## Running a service

Every binary serves the full chapkit API plus the chapkit-rs extensions `$predict-sync` (a forecast returned
directly, no job), `$explain` (LIME or KernelSHAP for one forecast cell, in-process), `$backtest` (a backtest
job, stored as an artifact) and `$hpo` (a hyperparameter search job). Besides `serve`, each binary has `test`
(chap-core's flow against a running service), `eval` (chap-core's backtest and metrics, in-process, no server;
`--json` for every score at full precision, `--config` for model options), `hpo` (random or grid search over
the model's search space, each trial a backtest) and `demo`. The main environment variables:

| variable | default | |
|---|---|---|
| `HOST`, `PORT` | 127.0.0.1, 8000 | listen address (also `serve --host`, `--port`) |
| `DATABASE_URL` | in memory | (also `serve --database`) set a file (`data/chapkit.db`, or `turso:///data/chapkit.db` with `FEATURES=turso`) for anything beyond a test: in memory, every stored prediction stays in RAM and a restart loses it |
| `SERVICEKIT_ORCHESTRATOR_URL` | unset | register with chap-core, e.g. `http://localhost:8700/v2/services/$register`; also set `SERVICEKIT_HOST`/`SERVICEKIT_PORT` to the address chap-core calls back on and `GIT_REVISION` |
| `CHAPKIT_MAX_BODY_MB` | 1024 | largest accepted request body (chap-core sends whole datasets as JSON) |
| `CHAPKIT_EAGER_WAIT_MS` | 1000 | hold a job submission until a fast job finishes, so chap-core's first poll sees it done |

The full list and the design behind it are in the chapkit-rs book (`make docs` in `../chapkit-rs`).

## Benchmarks

`bench/run.py` serves the same `global_median` model as a Python chapkit service
(`bench/python_global_median`) and as chapkit-rs, with and without eager completion, and measures startup,
the chap-core job cycle, CPU, cores used, memory and `$predict-sync` under load. `bench/results.json` holds
the last run; the numbers are written up in the book's Benchmarks chapter.
