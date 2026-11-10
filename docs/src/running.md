# Running a model

## Build

From the repository root, with `chapkit-rs` checked out next to it (`chap-workspace/rs/chapkit-rs`):

```sh
make release        # all binaries into ./bin
ls bin              # baseline-global-median  baseline-persistence  baseline-seasonal-median  baseline-seasonal-shift-persistence
```

## Serve

Each binary serves the full chapkit HTTP API with `serve`, so chap-core (or anything that speaks chapkit) can
use it. Without arguments a binary prints its commands (`serve`, `test`, `healthcheck`, `demo`); `--help` after
any of them lists its options:

```sh
bin/baseline-seasonal-median serve                 # http://127.0.0.1:8000, docs at /docs
bin/baseline-seasonal-median serve --host 0.0.0.0 --port 8001 --database data/chapkit.db
```

| option | variable | default | |
|---|---|---|---|
| `--host`, `--port` | `HOST`, `PORT` | 127.0.0.1, 8000 | listen address |
| `--database` | `DATABASE_URL` | in memory | a file (`data/chapkit.db`) for anything beyond a test, so results survive a restart and stay out of RAM |
| `--max-concurrency` | `CHAPKIT_MAX_CONCURRENCY` | 1 | jobs running at once |

A flag wins over its variable. The other settings are variables only:

| variable | default | |
|---|---|---|
| `SERVICEKIT_ORCHESTRATOR_URL` | unset | register with chap-core, for example `http://localhost:8700/v2/services/$register`; also set `SERVICEKIT_HOST`/`SERVICEKIT_PORT` to the address chap-core calls back on |
| `RAYON_NUM_THREADS` | all cores | cores a single train or predict may use; 1 gives the most work per core when many calls run at once |

The full list is in the chapkit-rs book (*Environment variables*). Besides chap-core's flow (`$train`,
`$predict`, jobs, artifacts), every service has `$predict-sync` (a forecast returned directly) and `$explain`
(feature importances for one forecast).

## Test a running service

With the service running, in another terminal:

```sh
bin/baseline-persistence test --locations 10
```

runs chap-core's flow against it (the same test as `chapkit-rs test`): configurations from its schema,
`$validate`, training and forecasting jobs, and a check of every stored forecast, on generated data. It finds the
service on `--host`/`--port` (the same defaults and variables as `serve`), or `--url`.

## Try once, without a server

```sh
bin/baseline-seasonal-shift-persistence demo
bin/baseline-persistence demo
```

trains and predicts once on the [example data](verification.md) and prints the timing and the median forecast
of each row.

## In Docker

Published images, for linux/amd64 and linux/arm64:

```sh
docker run -p 8000:8000 -v gm-data:/work/data ghcr.io/winterop-com/chapkit-rs-models/baseline-global-median
```

They are built from chapkit-rs's shared Dockerfile (see the chapkit-rs book, *Docker*): one binary on a minimal
base image, uid 1000, the database in `/work/data`, a read-only root filesystem and a built-in health check.
`make docker-up` builds and starts all three locally with `compose.yml`, on ports 8001 to 8003.

## With chap-core

Start the service with `SERVICEKIT_ORCHESTRATOR_URL` pointing at chap-core; it registers itself and keeps the
registration alive. With chap-devops (`cdx`):

```sh
cdx models configs sync seasonal-median          # a configured model with the defaults
cdx models test seasonal-median --backtest       # dataset, backtest and scores through chap-core
```
