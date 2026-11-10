# Run the baselines through chap-core

You start chap-core, register the four baselines with it, and let chap-core backtest them: build a dataset,
train and forecast repeatedly, and score the forecasts. This is the same path a model takes in a real Chap
deployment.

## Before you start

- The models built ([Set up your machine](setup.md)).
- Docker running (Docker Desktop, or Docker Engine on Linux).
- [chap-devops](https://github.com/dhis2-chap/chap-devops), which runs chap-core in Docker:

```sh
curl -fsSL https://raw.githubusercontent.com/dhis2-chap/chap-devops/main/install.sh | sh
chap-devops --help
```

## 1. Start chap-core

In a folder of your choice (not inside the repositories):

```sh
chap-devops init mychap --models none
cd mychap
chap-devops up --wait
```

`--models none` starts chap-core without its own model services; the baselines join from your machine.

## 2. Start the baselines and register them

From `chapkit-rs-models`, in four terminals (or with `&`), one port each. Persistence registers as `persistence-crs`, since
`persistence` is the name of chap-core's own built-in:

```sh
REV=$(git rev-parse HEAD)
for model in global-median:8001 seasonal-median:8002 seasonal-shift-persistence:8003 persistence:8004; do
  name=${model%:*}; port=${model#*:}
  SERVICEKIT_ORCHESTRATOR_URL='http://localhost:8700/v2/services/$register' \
  SERVICEKIT_HOST=host.docker.internal SERVICEKIT_PORT=$port \
  HOST=0.0.0.0 PORT=$port GIT_REVISION=$REV \
  bin/baseline-$name serve &
done
```

- `SERVICEKIT_ORCHESTRATOR_URL` is chap-core's registration endpoint; each service registers and keeps its
  registration alive.
- `HOST=0.0.0.0` and `SERVICEKIT_HOST=host.docker.internal` let chap-core, inside Docker, reach the services on
  your machine.

## 3. Check that chap-core sees them

Back in the `mychap` folder:

```sh
chap-devops status
```

```text
MODEL                       STATE      REACH                             LAST PING
global-median               unmanaged  http://host.docker.internal:8001  4s ago
seasonal-median             unmanaged  http://host.docker.internal:8002  4s ago
seasonal-shift-persistence  unmanaged  http://host.docker.internal:8003  4s ago
persistence-crs             unmanaged  http://host.docker.internal:8004  4s ago
```

## 4. Configure and backtest

Still in `mychap`:

```sh
for m in global-median seasonal-median seasonal-shift-persistence persistence-crs; do chap-devops models configs sync $m; done
chap-devops models test global-median seasonal-median seasonal-shift-persistence persistence-crs --backtest --seed 7
```

```text
testing 3 models (through chap-core: a dataset, a backtest and its scores)
global-median                 pass       8s   crps 16.3  mae 25.6  rmse 28.2
seasonal-median               pass       8s   crps 3.0  mae 3.5  rmse 4.2
seasonal-shift-persistence    pass       8s   crps 6.5  mae 8.1  rmse 11.3

3 of 3 models pass
```

(This output is from a run before persistence was added; the command now tests four models and prints a
line for `persistence-crs` too.)

For each model chap-core created a dataset from the model's sample data, ran a backtest (train, forecast,
compare with what happened, several times over) and computed the scores. Lower is better: here the seasonal
models clearly beat the global median, because the data is seasonal.

## 5. Clean up

```sh
kill %1 %2 %3 %4       # or stop the four services in their terminals
chap-devops down
```

## If something goes wrong

- **A model is missing from `chap-devops status`**: its service did not start or cannot reach chap-core; check
  its terminal for `registration.success`.
- **`chap-core ... is not responding`** right after `up`: chap-core takes a few seconds to start; wait and
  run `chap-devops status` again.

## Next steps

- [Run the models in Docker](docker.md), as a deployment would.
- [Same output as chap-core](../verification.md): how the scores above compare with chap-core's own baselines.
