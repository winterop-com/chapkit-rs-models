"""Draw the figures in docs/src/figures from the models' own output on the example data.

    ../../chap-core/.venv/bin/python scripts/make_figures.py      (make docs-figures; needs `make release`)

For each model, runs its binary's `demo` on the monthly fixture (DEMO_OUTPUT writes the prediction table),
then plots each location's history and the forecast median with the middle 50% and 80% of the samples.
Also draws the quantile-sampling illustration. Needs matplotlib (chap-core's environment has it).
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "src" / "figures"
FIXTURE = ROOT / "crates" / "baseline-core" / "fixtures" / "baselines_monthly.json"
MODELS = {
    "global_median": "Global median",
    "seasonal_median": "Seasonal median",
    "seasonal_shift_persistence": "Seasonal shift persistence",
}
plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
COLOR = "#2b6cb0"


def frame(wire):
    cols = wire["columns"]
    return [dict(zip(cols, row)) for row in wire["data"]]


def month_index(period):
    y, m = map(int, period.split("-"))
    return y * 12 + m - 1


def label(i):
    return f"{i // 12}-{i % 12 + 1:02d}"


def forecast_figure(model, title, history, trained_on, pred):
    locations = sorted({r["location"] for r in pred})
    fig, axes = plt.subplots(1, len(locations), figsize=(2.6 * len(locations), 2.9), sharey=True)
    for ax, loc in zip(np.atleast_1d(axes), locations):
        h = sorted((month_index(r["time_period"]), r["disease_cases"]) for r in history if r["location"] == loc)
        h = [(t, v) for t, v in h if t >= month_index("2020-01")]
        if h:
            ax.plot([t for t, _ in h], [np.nan if v is None else v for _, v in h], color="#444", lw=1.2, label="observed")
        rows = sorted((month_index(r["time_period"]), r) for r in pred if r["location"] == loc)
        t = [i for i, _ in rows]
        samples = np.array([[r[f"sample_{k}"] for k in range(100)] for _, r in rows])
        q = {p: np.percentile(samples, p, axis=1) for p in (10, 25, 50, 75, 90)}
        ax.fill_between(t, q[10], q[90], color=COLOR, alpha=0.18, lw=0, label="80% of samples")
        ax.fill_between(t, q[25], q[75], color=COLOR, alpha=0.35, lw=0, label="50% of samples")
        ax.plot(t, q[50], color=COLOR, lw=1.6, label="median")
        ax.set_title(f"location {loc}" + ("" if loc in trained_on else "\n(not in the training data)"), fontsize=9)
        ticks = [month_index("2020-01"), month_index("2021-01")]
        ax.set_xticks(ticks, ["Jan 2020", "Jan 2021"])
        ax.set_xticks([month_index("2020-07"), month_index("2021-07")], minor=True)
        ax.grid(axis="y", color="#ddd", lw=0.6)
    np.atleast_1d(axes)[0].set_ylabel("disease_cases")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=4, loc="lower center")
    fig.suptitle(f"{title}: forecast for May to October 2021", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / f"{model}.svg")
    plt.close(fig)


def quantile_figure():
    values = np.array([24.0, 27.0, 31.0, 13.0, 21.0, 26.0, 19.0, 33.0])
    levels = np.sort(np.append((np.arange(99) + 0.5) / 99, 0.5))
    samples = np.quantile(values, levels)
    fig, ax = plt.subplots(figsize=(6.4, 1.9))
    ax.scatter(samples, np.zeros_like(samples), s=9, color=COLOR, alpha=0.7, label="100 samples")
    ax.scatter(values, np.full_like(values, 0.6), marker="x", s=30, color="#444", label="observed values")
    ax.axvline(np.median(values), color="#c05621", lw=1, ls="--", label=f"median {np.median(values):g}")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("disease_cases")
    ax.set_ylim(-0.4, 0.9)
    fig.legend(frameon=False, fontsize=8, ncol=3, loc="lower center")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(OUT / "quantiles.svg")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fixture = json.loads(FIXTURE.read_text())
    history = frame(fixture["historic"])
    trained_on = {r["location"] for r in frame(fixture["train"])}
    with tempfile.TemporaryDirectory() as tmp:
        for model, title in MODELS.items():
            binary = ROOT / "bin" / f"baseline-{model.replace('_', '-')}"
            path = Path(tmp) / f"{model}.json"
            subprocess.run([str(binary), "demo"], check=True, capture_output=True, env={**os.environ, "DEMO_OUTPUT": str(path)})
            forecast_figure(model, title, history, trained_on, frame(json.loads(path.read_text())))
    quantile_figure()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
