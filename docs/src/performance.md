# Performance

Chap is heading for forecasts at sub-district and facility level: 5,000 to 100,000 locations and more. This
page measures how the ports and chap-core's Python baselines scale towards that, on one machine (Apple
Silicon, 8 performance cores, 64 GB).

## How it is measured

`make bench-scale` generates monthly data of a given size, identical on both sides, and runs each model in its
own process: chap-core's built-in baseline (`bench/scale_builtin.py`) and the Rust port (`baseline-bench`), the
latter once on one core and once on all cores. For each run:

- **train + predict**: wall time of training and forecasting 3 months, in milliseconds;
- **CPU and cores used**: CPU time measured inside the process, divided by wall time;
- **peak memory**: of the whole process, data included;
- **convert** (chap-core): turning pandas into chap-core's `DataSet` before calling the model;
- **decode** (chapkit-rs): parsing the same data from JSON, as it arrives over HTTP.

chap-core's CPU time equals its wall time in every run: it uses one core.

## Towards facility level: 100 to 100,000 locations, 10 years of months

Train + predict in milliseconds; peak memory in MB.

**Global median**

| locations | rows | chap-core | Rust, 1 core | Rust, all cores | chap-core MB | Rust MB |
|---|---|---|---|---|---|---|
| 1,000 | 120,000 | 48 | 7.0 | 7.5 | 160 | 26 |
| 5,000 | 600,000 | 240 | 34 | 32 | 316 | 106 |
| 10,000 | 1.2 million | 477 | 70 | 62 | 505 | 188 |
| 50,000 | 6 million | 2,346 | 356 | 304 | 2,144 | 950 |
| 100,000 | 12 million | 4,814 | 737 | 602 | 3,739 | 1,784 |

**Seasonal median**

| locations | chap-core | Rust, 1 core | Rust, all cores | chap-core MB | Rust MB |
|---|---|---|---|---|---|
| 1,000 | 334 | 17 | 13.5 | 171 | 44 |
| 5,000 | 1,624 | 84 | 53 | 379 | 174 |
| 10,000 | 3,152 | 162 | 96 | 636 | 348 |
| 50,000 | 15,833 | 833 | 449 | 2,576 | 1,676 |
| 100,000 | 32,604 | 1,688 | 950 | 4,160 | 3,343 |

**Seasonal shift persistence**

| locations | chap-core | Rust, 1 core | Rust, all cores | chap-core MB | Rust MB |
|---|---|---|---|---|---|
| 1,000 | 356 | 34 | 17 | 210 | 61 |
| 5,000 | 1,825 | 172 | 71 | 588 | 277 |
| 10,000 | 3,653 | 348 | 117 | 1,020 | 549 |
| 50,000 | 18,821 | 1,784 | 564 | 4,586 | 2,667 |
| 100,000 | 38,011 | 3,501 | 1,126 | 7,782 | 5,212 |

At 100,000 locations a seasonal median forecast takes chap-core 33 seconds, plus 47 seconds to convert the
data; the port takes 1 second on all cores, and decodes the data in 1.8 seconds.

**Persistence** is not yet in these tables: `make bench-scale` includes it, but the results above predate it.

## Long histories: 100 locations, up to 5,000 months

| model | size | chap-core | Rust, 1 core | Rust, all cores |
|---|---|---|---|---|
| global median | 100 x 5,000 | 14.5 | 5.3 | 4.0 |
| seasonal median | 100 x 5,000 | 105 | 21 | 8.4 |
| seasonal shift persistence | 100 x 5,000 | 170 | 47 | 15 |
| global median | 2,000 x 2,000 | 145 | 55 | 41 |

With few locations and very long series the gap is smallest: numpy handles one long array per location well.

## What this shows

- **Faster at every size.** On one core against chap-core's one core the ports are 2.7 to 25 times faster; on
  all cores 3.6 to 35 times. These are model-only numbers, the fair ones to quote.
- **The speedup holds as locations grow**: from 5,000 to 100,000 locations about 7, 19 and 11 times on one core
  (global median, seasonal median, seasonal shift persistence), and 8, 31 to 34 and 26 to 34 times on all cores.
  Both sides scale linearly, so the ports stay ahead by the same factor.
- **Less memory at every size**: under a tenth of chap-core's at small sizes; at 100,000 locations about half
  (global median), four fifths (seasonal median) and two thirds (seasonal shift persistence).
- **All cores trade CPU for latency.** Running locations in parallel shortens a call but uses more CPU in total
  than one core (1.8 to 10 cores busy). A service answering many calls at once gets the most work per core with
  `RAYON_NUM_THREADS=1`.
- **The data step is context, not a speedup.** chap-core's conversion costs more than its models, and decoding
  the same data from JSON is far cheaper for the port. But when chap-core calls a service it still prepares and
  serializes its data, which this benchmark does not measure, so the two cannot simply be added up. The ports'
  times include grouping rows by location, which chap-core does during conversion.

## At facility level, the data format matters

At 100,000 locations a 10-year training request is 589 MB of JSON in chapkit's row format, and decoding it
takes longer than running any of the three models measured. For facility-level forecasting, a more compact columnar
format between chap-core and model services would save more time than any further model speedup.

## How the ports got here

The ports were profiled, not guessed at: parsing period strings, grouping rows by location (by comparing
strings) and looking up ratios in maps were the hot spots. Fixing those, storing strings as codes into a
dictionary of distinct values, and computing the pooled ratios of seasonal shift persistence only when needed
made one core about twice as fast and roughly halved memory at large sizes. polars was measured as an alternative for the
per-location work and was 3 to 4 times slower here, mostly because converting to it costs as much as a whole
training.

All runs are in `bench/scale_results.json`; `make bench-scale` repeats them, and `make bench-compare` checks the
outputs are the same ([details](verification.md)).
