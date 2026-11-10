"""Benchmark the same global_median model served by Python chapkit and by chapkit-rs.

    python3 bench/run.py            (from the chapkit-rs-models root, after `make release`)

Runs each service as a real process on localhost and measures:

- startup: process start until /health answers
- chap-core cycle: config + $train + wait + $predict + wait, polling exactly like chap-core's
  wait_for_job (poll immediately, then every 2 s). This is what a chap-core backtest pays per model call.
- raw cycle: the same with 5 ms polling, i.e. how long the work itself takes end to end over HTTP
- load: 8 concurrent clients each running predict cycles for a fixed time; throughput, CPU seconds
  and average cores used (CPU time / wall time)
- peak RSS of the service process, sampled every 50 ms through the whole run
- $predict-sync latency (chapkit-rs only)

Variants: Python chapkit, chapkit-rs (eager completion on, the default), chapkit-rs with
CHAPKIT_EAGER_WAIT_MS=0, which separates the polling trick from the speed of Rust.
Uses only the standard library. Writes bench/results.json.
"""

import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parent.parent
PY = WORKSPACE / "chapkit" / ".venv" / "bin" / "python"
RS_BIN = ROOT / "bin" / "baseline-global-median"

CHAP_CORE_POLL = 2.0
CHAP_CORE_CYCLES = 3
RAW_CYCLES = 20
LOAD_CLIENTS = 8
LOAD_SECONDS = 10.0
SYNC_CALLS = 300


def http(method, url, body=None, timeout=60):
    """JSON request; `body` may be pre-encoded bytes so client-side encoding is not measured."""
    data = body if isinstance(body, bytes) else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method=method, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


def ps(pid, field):
    out = subprocess.run(["ps", "-o", f"{field}=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return out


def cpu_seconds(pid):
    """Total CPU time (user + system) of a process, in seconds."""
    t = ps(pid, "time")  # [[dd-]hh:]mm:ss.ss
    if not t:
        return 0.0
    days, _, rest = t.rpartition("-")
    parts = [float(p) for p in rest.split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs + (float(days) * 86400 if days else 0.0)


class RssSampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.peak_kb, self.stop = pid, 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            v = ps(self.pid, "rss")
            if v.isdigit():
                self.peak_kb = max(self.peak_kb, int(v))
            time.sleep(0.05)


class Service:
    def __init__(self, name, cmd, port, env=None, cwd=None):
        self.name, self.port, self.base = name, port, f"http://127.0.0.1:{port}"
        full_env = {**os.environ, "PORT": str(port), "LOG_LEVEL": "warning", **(env or {})}
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, env=full_env, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while True:
            try:
                http("GET", self.base + "/health", timeout=1)
                break
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"{name} exited during startup")
                time.sleep(0.01)
        self.startup_ms = (time.perf_counter() - t0) * 1000
        self.rss = RssSampler(self.proc.pid)
        self.rss.start()

    def close(self):
        self.rss.stop.set()
        self.proc.terminate()
        self.proc.wait(timeout=20)


def wait_job(base, job_id, poll):
    while True:
        r = http("GET", f"{base}/api/v1/jobs/{job_id}")
        if r["status"] in ("completed", "failed", "canceled"):
            if r["status"] != "completed":
                raise RuntimeError(f"job {job_id} {r['status']}: {r.get('error')}")
            return
        time.sleep(poll)


def cycle(base, config_id, train, predict, poll):
    """One chap-core style train + predict over HTTP, returns seconds."""
    t0 = time.perf_counter()
    sub = http("POST", base + "/api/v1/ml/$train", train_body(config_id, train))
    wait_job(base, sub["job_id"], poll)
    sub = http("POST", base + "/api/v1/ml/$predict", predict_body(sub["artifact_id"], predict))
    wait_job(base, sub["job_id"], poll)
    http("GET", f"{base}/api/v1/artifacts/{sub['artifact_id']}")
    return time.perf_counter() - t0


_ENCODED = {}


def train_body(config_id, train):
    """Pre-encoded `$train` body (encoded once per config, then reused)."""
    key = ("train", config_id)
    if key not in _ENCODED:
        _ENCODED[key] = json.dumps({"config_id": config_id, **train}).encode()
    return _ENCODED[key]


def predict_body(model_id, predict):
    """Pre-encoded `$predict` body; the inner frames are encoded once and spliced."""
    if "predict" not in _ENCODED:
        _ENCODED["predict"] = json.dumps(predict).encode()[1:]
    return b'{"artifact_id":"' + model_id.encode() + b'",' + _ENCODED["predict"]


def predict_cycle(base, model_id, predict, poll):
    t0 = time.perf_counter()
    sub = http("POST", base + "/api/v1/ml/$predict", predict_body(model_id, predict))
    wait_job(base, sub["job_id"], poll)
    return time.perf_counter() - t0


def bench(svc, train, predict, has_sync):
    base = svc.base
    out = {"startup_ms": round(svc.startup_ms, 1)}
    time.sleep(0.2)
    rss = ps(svc.proc.pid, "rss")
    out["rss_idle_mb"] = round(int(rss) / 1024, 1) if rss.isdigit() else None
    cfg = http("POST", base + "/api/v1/configs", {"name": "bench", "data": {"prediction_periods": 3}})

    chap = [cycle(base, cfg["id"], train, predict, CHAP_CORE_POLL) for _ in range(CHAP_CORE_CYCLES)]
    out["chap_core_cycle_s"] = round(statistics.median(chap), 3)

    raw = [cycle(base, cfg["id"], train, predict, 0.005) for _ in range(RAW_CYCLES)]
    out["raw_cycle_ms"] = round(statistics.median(raw) * 1000, 1)

    sub = http("POST", base + "/api/v1/ml/$train", train_body(cfg["id"], train))
    wait_job(base, sub["job_id"], 0.005)
    model_id = sub["artifact_id"]

    # Fixed work, so memory is compared for the same amount of stored data.
    for _ in range(100):
        predict_cycle(base, model_id, predict, 0.005)
    rss = ps(svc.proc.pid, "rss")
    out["rss_after_100_predictions_mb"] = round(int(rss) / 1024, 1) if rss.isdigit() else None

    # Load: concurrent predict cycles. Jobs run one at a time in both (chapkit's default
    # concurrency is 1), so this measures the request path and job throughput.
    deadline = time.perf_counter() + LOAD_SECONDS
    counts = []

    def worker():
        n = 0
        while time.perf_counter() < deadline:
            predict_cycle(base, model_id, predict, 0.005)
            n += 1
        counts.append(n)

    cpu0, t0 = cpu_seconds(svc.proc.pid), time.perf_counter()
    with ThreadPoolExecutor(LOAD_CLIENTS) as ex:
        for _ in range(LOAD_CLIENTS):
            ex.submit(worker)
    wall, cpu = time.perf_counter() - t0, cpu_seconds(svc.proc.pid) - cpu0
    out["load_predictions_per_s"] = round(sum(counts) / wall, 1)
    out["load_cpu_s"] = round(cpu, 2)
    out["load_cores_used"] = round(cpu / wall, 2)
    out["cpu_ms_per_prediction"] = round(cpu * 1000 / max(1, sum(counts)), 2)

    if has_sync:
        body = predict_body(model_id, predict)
        lat = []
        for _ in range(SYNC_CALLS):
            t = time.perf_counter()
            http("POST", base + "/api/v1/ml/$predict-sync", body)
            lat.append(time.perf_counter() - t)
        lat.sort()
        out["predict_sync_p50_ms"] = round(lat[len(lat) // 2] * 1000, 2)
        out["predict_sync_p99_ms"] = round(lat[int(len(lat) * 0.99)] * 1000, 2)
        out.update(ab_load(svc, base + "/api/v1/ml/$predict-sync", body))

    out["peak_rss_mb"] = round(svc.rss.peak_kb / 1024, 1)
    return out


def ab_load(svc, url, body, n=3000, c=8):
    """$predict-sync under load with ApacheBench (native client, so the client is not the bottleneck)."""
    path = ROOT / "bench" / ".ab_body.json"
    path.write_bytes(body)
    cpu0, t0 = cpu_seconds(svc.proc.pid), time.perf_counter()
    out = subprocess.run(["ab", "-q", "-n", str(n), "-c", str(c), "-p", str(path), "-T", "application/json", url],
                         capture_output=True, text=True).stdout
    wall, cpu = time.perf_counter() - t0, cpu_seconds(svc.proc.pid) - cpu0
    path.unlink(missing_ok=True)
    def grab(prefix):
        for line in out.splitlines():
            if line.strip().startswith(prefix):
                return line
        return ""
    rps = grab("Requests per second").split(":")[1].split()[0] if grab("Requests per second") else "0"
    failed = grab("Failed requests").split(":")[1].strip() if grab("Failed requests") else "?"
    pct = {l.split()[0]: l.split()[1] for l in out.splitlines() if l.strip().endswith(tuple("0123456789")) and "%" in l.split()[0]}
    return {
        "sync_load_requests_per_s": float(rps),
        "sync_load_failed": failed,
        "sync_load_p50_ms": float(pct.get("50%", "nan")),
        "sync_load_p99_ms": float(pct.get("99%", "nan")),
        "sync_load_cores_used": round(cpu / wall, 2),
        "sync_load_cpu_ms_per_request": round(cpu * 1000 / n, 2),
    }


def main():
    if not RS_BIN.exists():
        sys.exit("run `make release` first")
    # One dataset for everyone: 20 locations x 120 months of training data, 3 future months.
    gen = Service("generator", [str(RS_BIN), "serve"], 8199)
    q = "num_locations=20&num_periods=120&num_features=3&seed=1"
    train = {"data": http("GET", gen.base + "/api/v1/ml/$generate-sample-data?" + q)["data"]}
    sample = http("GET", gen.base + "/api/v1/ml/$generate-sample-data?kind=predict&" + q.replace("num_periods=120", "num_periods=3"))
    gen.close()
    predict = {"historic": train["data"], "future": sample["future"], "run_info": {"prediction_length": 3}}

    variants = [
        ("python-chapkit", [str(PY), "-m", "uvicorn", "main:app", "--port", "8101", "--log-level", "warning"], 8101, {},
         ROOT / "bench" / "python_global_median", False),
        ("chapkit-rs", [str(RS_BIN), "serve"], 8102, {}, None, True),
        ("chapkit-rs-no-eager", [str(RS_BIN), "serve"], 8103, {"CHAPKIT_EAGER_WAIT_MS": "0"}, None, True),
    ]
    results = {"dataset": {"train_rows": len(train["data"]["data"]), "future_rows": len(sample["future"]["data"])}}
    for name, cmd, port, env, cwd, has_sync in variants:
        print(f"== {name}", flush=True)
        svc = Service(name, cmd, port, env, cwd)
        try:
            results[name] = bench(svc, train, predict, has_sync)
        finally:
            svc.close()
        print(json.dumps(results[name], indent=1), flush=True)

    results["sizes"] = {
        "chapkit_rs_binary_mb": round(RS_BIN.stat().st_size / 1e6, 1),
        "python_chapkit_venv_mb": round(
            int(subprocess.run(["du", "-sk", str(WORKSPACE / "chapkit" / ".venv")], capture_output=True, text=True).stdout.split()[0]) / 1024, 1
        ),
    }
    (ROOT / "bench" / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["sizes"]))


if __name__ == "__main__":
    main()
