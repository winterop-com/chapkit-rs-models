# Set up your machine

By the end of this page you have both repositories checked out, every model built, and the test suite
passing, which shows the models produce chap-core's numbers on your machine.

## Before you start

- macOS or Linux (on Windows, use WSL).
- About 2 GB of free disk space for the build.

**What you do not need:** Python, R or chap-core. The models are self-contained Rust programs; chap-core is only
needed for [running them through chap-core](chap-core.md), and Docker only for [running them in
containers](docker.md).

## 1. Install Rust

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Open a new terminal and check:

```sh
cargo --version        # cargo 1.99 or newer
```

## 2. Check out both repositories side by side

The models use the chapkit-rs library from the folder next to them:

```sh
mkdir -p chap-workspace/rs && cd chap-workspace/rs
git clone https://github.com/winterop-com/chapkit-rs
git clone https://github.com/winterop-com/chapkit-rs-models
cd chapkit-rs-models
```

## 3. Build the models

```sh
make release
```

The first build compiles all dependencies and takes a few minutes. On macOS each binary holds both Apple Silicon and
Intel code. It ends with the six binaries:

```text
28M bin/baseline-configurable
28M bin/baseline-global-median
28M bin/baseline-persistence
28M bin/baseline-seasonal-median
28M bin/baseline-seasonal-shift-persistence
28M bin/simple-multistep
```

## 4. Verify

```sh
make check
```

runs the linter and every test, including the parity tests that compare each model with chap-core's own output
sample by sample. Every `test result` line ends in `ok`.

For a first look at a forecast, run one model once on the example data:

```sh
bin/baseline-seasonal-median demo
```

```text
seasonal-median: train 263µs, predict 41µs, 24 rows
  A   2021-05  median   27.00
  A   2021-06  median   26.00
  ...
```

These are the numbers the [seasonal median page](../models/seasonal-median.md) works through by hand.

## If something goes wrong

- **`failed to load manifest ... ../chapkit-rs`**: chapkit-rs is not next to chapkit-rs-models. Both must be
  in the same folder.
- **`cargo: command not found`**: open a new terminal after installing Rust, or run `source ~/.cargo/env`.

## Next steps

- [Run a baseline and read its forecast](first-forecast.md)
