.PHONY: help check build release lint fmt test bench bench-http bench-scale bench-compare demo fixtures docs docs-serve docs-figures docker-build docker-up docker-down clean

BINS := baseline-global-median baseline-seasonal-median baseline-seasonal-shift-persistence baseline-persistence
CHAP_CORE ?= ../../chap-core
# FEATURES=turso builds the binaries with the turso store backend.
FEATURES ?=
CARGO_FEATURES := $(if $(FEATURES),--features $(FEATURES),)

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  check     lint + test; run before every commit"
	@echo "  build     Build all example crates (debug)"
	@echo "  release   Build release binaries into ./bin (FEATURES=turso for the turso backend)"
	@echo "  lint      Format and run clippy (warnings are errors)"
	@echo "  test      Run all tests (parity with chap-core, checkpoints, HTTP flow)"
	@echo "  bench     Run criterion benchmarks (BENCH=<filter> to narrow)"
	@echo "  bench-http  Compare against Python chapkit over HTTP (bench/run.py; needs ../../chapkit/.venv)"
	@echo "  bench-scale  chap-core's built-in baselines vs the Rust ports, 100 to 5000 locations and periods"
	@echo "  bench-compare  Check every forecast sample against chap-core's on the benchmark data"
	@echo "  demo      Run every model once in-process and time it"
	@echo "  fixtures  Regenerate baseline golden outputs from chap-core (CHAP_CORE=$(CHAP_CORE))"
	@echo "  docs      Build the book (docs/book)"
	@echo "  docs-serve  Serve the book with live reload"
	@echo "  docs-figures  Redraw the book's figures from the models' output (after make release)"
	@echo "  docker-build  Build an image per model (chapkit-rs's shared Dockerfile; ../chapkit-rs needed)"
	@echo "  docker-up     Build and start all models with docker compose (ports 8001-8003)"
	@echo "  docker-down   Stop them"
	@echo "  clean     Remove build output"

check: lint test

build:
	@cargo build --workspace --all-targets

release:
	@cargo build --workspace --release $(CARGO_FEATURES)
	@rm -rf bin && mkdir -p bin
	@for b in $(BINS); do cp target/release/$$b bin/$$b; done
	@ls -lh bin | awk 'NR>1 {print $$5, "bin/"$$9}'

lint:
	@cargo fmt --all
	@cargo clippy --workspace --all-targets -- -D warnings

fmt:
	@cargo fmt --all

test:
	@cargo test --workspace

bench:
	@cargo bench --workspace -- $(BENCH)

bench-http: release
	@python3 bench/run.py

bench-scale:
	@python3 bench/scale.py

bench-compare:
	@python3 bench/compare_outputs.py

demo:
	@for b in $(BINS); do cargo run -q --release --bin $$b -- demo || exit 1; echo; done

fixtures:
	@$(CHAP_CORE)/.venv/bin/python scripts/gen_baseline_fixtures.py

docs:
	@mdbook build docs

docs-serve:
	@mdbook serve docs --open

docs-figures:
	@$(CHAP_CORE)/.venv/bin/python scripts/make_figures.py

docker-build:
	@GIT_REVISION=$$(git rev-parse HEAD) docker compose build

docker-up:
	@GIT_REVISION=$$(git rev-parse HEAD) docker compose up --build -d
	@docker compose ps

docker-down:
	@docker compose down

clean:
	@cargo clean
	@rm -rf bin
