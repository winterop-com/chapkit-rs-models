# chapkit-rs-examples

Example model projects for [chapkit-rs](../chapkit-rs). Each crate is laid out the way a real model project
would be: it depends on `chapkit`, implements the `Model` trait, and gets the full chapkit-compatible HTTP
API (so chap-core can use it unchanged) from `chapkit::Service`.

The library is expected as a sibling checkout: `chap-workspace/rs/chapkit-rs` next to this repository.

| crate | what |
|---|---|
| `baselines` | chap-core's built-in baselines (`global_median`, `seasonal_median`, `seasonal_shift_persistence`), ported and checked against chap-core number for number. One binary per model. |
| `ridge` | Ridge regression on covariates with a search space and tensor checkpoints; the reference model for xAI and HPO. |

```sh
make release                     # binaries into ./bin
bin/global-median                # serve on 127.0.0.1:8000 (HOST, PORT to change)
bin/seasonal-median demo         # run once in-process and time it
make check                       # lint + tests (parity with chap-core, checkpoints, HTTP flow)
make fixtures                    # regenerate the golden outputs from a chap-core checkout
make bench                       # criterion: the models in-process
make bench-http                  # the same model through Python chapkit and chapkit-rs over HTTP
make release FEATURES=turso      # binaries that also accept turso: database URLs
```

## Running a service

Every binary serves the full chapkit API plus the chapkit-rs extensions `$predict-sync` (a forecast returned
directly, no job) and `$explain` (LIME or KernelSHAP for one forecast cell, in-process). The main
environment variables:

| variable | default | |
|---|---|---|
| `HOST`, `PORT` | 127.0.0.1, 8000 | listen address |
| `DATABASE_URL` | in memory | set a file (`data/chapkit.db`, or `turso:///data/chapkit.db` with `FEATURES=turso`) for anything beyond a test: in memory, every stored prediction stays in RAM and a restart loses it |
| `SERVICEKIT_ORCHESTRATOR_URL` | unset | register with chap-core, e.g. `http://localhost:8700/v2/services/$register`; also set `SERVICEKIT_HOST`/`SERVICEKIT_PORT` to the address chap-core calls back on and `GIT_REVISION` |
| `CHAPKIT_MAX_BODY_MB` | 1024 | largest accepted request body (chap-core sends whole datasets as JSON) |
| `CHAPKIT_EAGER_WAIT_MS` | 1000 | hold a job submission until a fast job finishes, so chap-core's first poll sees it done |

The full list and the design behind it are in the chapkit-rs book (`make docs` in `../chapkit-rs`).

## Benchmarks

`bench/run.py` serves the same `global_median` model as a Python chapkit service
(`bench/python_global_median`) and as chapkit-rs, with and without eager completion, and measures startup,
the chap-core job cycle, CPU, cores used, memory and `$predict-sync` under load. `bench/results.json` holds
the last run; the numbers are written up in the book's Benchmarks chapter.
