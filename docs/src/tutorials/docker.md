# Run the models in Docker

You run the models as containers, the way they are deployed next to chap-core: from the published images, or
built from source.

## Before you start

- Docker running.
- For building from source: both repositories checked out side by side ([Set up your machine](setup.md)).

## 1. Run a published image

```sh
docker run -d --name gm -p 8001:8000 -v gm-data:/work/data \
  ghcr.io/winterop-com/chapkit-rs-models/baseline-global-median
docker ps --filter name=gm --format '{{.Names}} {{.Status}}'
```

```text
gm Up 6 seconds (healthy)
```

The image holds just the model binary (about 70 MB in all). It listens on port 8000, keeps its database in the
`/work/data` volume, and reports its health to Docker. Everything from [Run a baseline and read its
forecast](first-forecast.md) works against it on port 8001. Stop it with `docker rm -f gm`.

Or with chap-devops, which picks a free port and keeps track of it:

```sh
chap-devops run ghcr.io/winterop-com/chapkit-rs-models/baseline-global-median:latest
chap-devops ps
```

(chap-devops wants an explicit tag such as `:latest`.) To check a running model the way chap-core uses it, the
`chapkit-rs` command from the chapkit-rs book runs the whole flow against it:

```sh
chapkit-rs test --url http://localhost:5001      # the URL chap-devops printed
```

## 2. Or build and run all seven with compose

```sh
cd chapkit-rs-models
make docker-up
```

The services start with `health: starting`; a few seconds later:

```sh
docker compose ps --format 'table {{.Service}}\t{{.Status}}\t{{.Ports}}'
```

lists the services, each `Up` and `(healthy)` once its health check passes: `global-median` on port 8001,
`seasonal-median` 8002, `seasonal-shift-persistence` 8003, `persistence` 8004, `configurable-baseline` 8005 and
`simple-multistep` 8006 and `auto-arima` 8007.

They run on ports 8001 to 8007, each with its own volume, a read-only filesystem and a health check. The images are
built with chapkit-rs's shared Dockerfile (see the chapkit-rs book, *Docker*). `make docker-down` stops them;
the data stays in the volumes.

## 3. Connect them to chap-core

Set `SERVICEKIT_ORCHESTRATOR_URL` to chap-core's registration endpoint, and put the containers on chap-core's
Docker network so chap-core can reach them by service name:

```sh
SERVICEKIT_ORCHESTRATOR_URL='http://chap:8000/v2/services/$register' docker compose up -d
```

(The address depends on your chap-core deployment; with chap-devops, `chap-devops status` shows chap-core's.)

## If something goes wrong

- **`unhealthy`**: `docker logs <container>` shows why; the health check calls the service's `/health`.
- **`denied` when pulling**: the image is not public yet; build it from source with `make docker-build`.

## Next steps

- [Performance](../performance.md): what to expect at 5,000 to 100,000 locations.
