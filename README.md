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
```
