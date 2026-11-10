# How a model package is laid out

Every model is its own crate under `crates/`, named after the model; ports of chap-core's baselines are
prefixed `baseline-`. A package has three source files, each with one job:

| file | contents | for whom |
|---|---|---|
| `src/model.rs` | **the calculation**: plain, commented functions over numbers, with no tables, threads or HTTP | anyone who wants to know how the forecast is made |
| `src/lib.rs` | the chapkit `Model` implementation: splits the data per location, runs locations in parallel, builds the prediction table, turns problems into clear errors | developers |
| `src/main.rs` | the binary: serve, or `demo` | operators |

plus `tests/parity.rs` (same output as the original, checkpoint round trip) and `benches/model.rs` (timings at
a few sizes, `make bench`).

Shared code lives in two places:

- **`chapkit` (the library)** for anything any model needs: splitting a table into one series per location
  (`chapkit::locations`), running locations in parallel and building the prediction table, checkpoints, the HTTP
  service, test helpers.
- **`crates/baseline-core`** for what only the baselines share: the quantile step, chap-core's fixtures and the
  parity check, the demo runner and synthetic benchmark data.

## Adding a model

1. Create `crates/<model>/` with `Cargo.toml`, `src/model.rs`, `src/lib.rs`, `src/main.rs` (copy a baseline).
2. Write the calculation in `model.rs` first, as functions a reader can follow; keep data handling out of it.
3. In `lib.rs`, implement `chapkit::Model` using `chapkit::locations` for the per-location work.
4. If it ports an existing model, add its outputs as a fixture and a parity test, and run it through chap-core.
5. Add the binary to `BINS` in the `Makefile` and a page to this book.
