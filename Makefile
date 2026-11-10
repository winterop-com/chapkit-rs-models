.PHONY: help check build release lint fmt test bench bench-http bench-scale bench-compare demo fixtures docs docs-serve docs-figures docker-build docker-up docker-down clean

BINS := baseline-global-median baseline-seasonal-median baseline-seasonal-shift-persistence baseline-persistence baseline-configurable simple-multistep
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
	@echo "  release   Release binaries into ./bin, universal on macOS (FEATURES=turso for turso)"
	@echo "  lint      Format and run clippy (warnings are errors)"
	@echo "  test      Run all tests (parity with chap-core, checkpoints, HTTP flow)"
	@echo "  bench     Run criterion benchmarks (BENCH=<filter> to narrow)"
	@echo "  bench-http  Compare against Python chapkit over HTTP (bench/run.py; needs ../../chapkit/.venv)"
	@echo "  bench-scale  chap-core's built-in baselines vs the chapkit-rs ports, up to 100,000 locations"
	@echo "  bench-compare  Check every forecast sample against chap-core's on the benchmark data"
	@echo "  demo      Run every model once in-process and time it"
	@echo "  fixtures  Regenerate baseline golden outputs from chap-core (CHAP_CORE=$(CHAP_CORE))"
	@echo "  docs      Build the book (docs/book)"
	@echo "  docs-serve  Serve the book with live reload"
	@echo "  docs-figures  Redraw the book's figures from the models' output (after make release)"
	@echo "  docker-build  Build an image per model (chapkit-rs's shared Dockerfile; ../chapkit-rs needed)"
	@echo "  docker-up     Build and start all models with docker compose (ports 8001-8006)"
	@echo "  docker-down   Stop them"
	@echo "  clean     Remove build output"

check: lint test

build:
	@cargo build --workspace --all-targets

# On macOS, one binary per model for both architectures (joined with lipo, as chap-devops ships
# its macOS build); elsewhere a native build.
release:
ifeq ($(shell uname -s),Darwin)
	@for t in aarch64-apple-darwin x86_64-apple-darwin; do cargo build --release $(CARGO_FEATURES) --target $$t $(addprefix -p ,$(BINS)) || exit 1; done
	@rm -rf bin && mkdir -p bin
	@for b in $(BINS); do lipo -create -output bin/$$b target/aarch64-apple-darwin/release/$$b target/x86_64-apple-darwin/release/$$b; done
else
	@cargo build --release $(CARGO_FEATURES) $(addprefix -p ,$(BINS))
	@rm -rf bin && mkdir -p bin
	@for b in $(BINS); do cp target/release/$$b bin/$$b; done
endif
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
	@for b in $(filter baseline-%,$(BINS)); do cargo run -q --release --bin $$b -- demo || exit 1; echo; done

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
