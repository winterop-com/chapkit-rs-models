.PHONY: help check build release lint fmt test bench demo fixtures clean

BINS := global-median seasonal-median seasonal-shift-persistence ridge
CHAP_CORE ?= ../../chap-core

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  check     lint + test; run before every commit"
	@echo "  build     Build all example crates (debug)"
	@echo "  release   Build release binaries into ./bin"
	@echo "  lint      Format and run clippy (warnings are errors)"
	@echo "  test      Run all tests (parity with chap-core, checkpoints, HTTP flow)"
	@echo "  bench     Run criterion benchmarks (BENCH=<filter> to narrow)"
	@echo "  demo      Run every model once in-process and time it"
	@echo "  fixtures  Regenerate baseline golden outputs from chap-core (CHAP_CORE=$(CHAP_CORE))"
	@echo "  clean     Remove build output"

check: lint test

build:
	@cargo build --workspace --all-targets

release:
	@cargo build --workspace --release
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

demo:
	@for b in $(BINS); do cargo run -q --release --bin $$b -- demo || exit 1; echo; done

fixtures:
	@$(CHAP_CORE)/.venv/bin/python scripts/gen_baseline_fixtures.py

clean:
	@cargo clean
	@rm -rf bin
