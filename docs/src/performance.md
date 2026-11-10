# Performance

Chap is heading for forecasts at sub-district and facility level: 5,000 to 100,000 locations and more. This
page measures how the ports and chap-core's Python baselines scale towards that, on one machine (Apple
Silicon, 8 performance cores, 64 GB). The runs are from 2026-11-10, against chap-core 2.7.0.

## How it is measured

`make bench-scale` generates monthly data of a given size, identical on both sides, and runs each model in its
own process: chap-core's built-in baseline (`bench/scale_builtin.py`) and the chapkit-rs port
(`baseline-bench`), the latter once on one core and once on all cores. For each run:

- **train + predict**: wall time of training and forecasting 3 months, in milliseconds;
- **CPU and cores used**: CPU time measured inside the process, divided by wall time;
- **peak memory**: of the whole process, data included;
- **convert** (chap-core): turning pandas into chap-core's `DataSet` before calling the model;
- **decode** (chapkit-rs): parsing the same data from JSON, as it arrives over HTTP.

chap-core uses one core in every run: its CPU time divided by wall time is at most 1.0.

chap-core 2.7.0's model times are between 1.7% slower and 8.6% faster than 2.6.0's on the same sizes (median 3.5%
faster, 34 of 45 runs within 5%). 2.7's speedup (#857) was in backtest code, not in the models.

## Towards facility level: 100 to 100,000 locations, 10 years of months

Train + predict in milliseconds; peak memory in MB.

**Global median**

| locations | rows | chap-core | chapkit-rs, 1 core | chapkit-rs, all cores | chap-core MB | chapkit-rs MB |
|---|---|---|---|---|---|---|
| 1,000 | 120,000 | 47 | 7.0 | 7.4 | 159 | 25 |
| 5,000 | 600,000 | 220 | 33 | 31 | 307 | 103 |
| 10,000 | 1.2 million | 453 | 68 | 62 | 464 | 185 |
| 50,000 | 6 million | 2,310 | 348 | 308 | 1,747 | 942 |
| 100,000 | 12 million | 4,636 | 694 | 595 | 3,347 | 1,777 |

**Persistence**

| locations | chap-core | chapkit-rs, 1 core | chapkit-rs, all cores | chap-core MB | chapkit-rs MB |
|---|---|---|---|---|---|
| 1,000 | 166 | 16 | 8.9 | 181 | 28 |
| 5,000 | 842 | 83 | 42 | 383 | 113 |
| 10,000 | 1,719 | 170 | 83 | 647 | 218 |
| 50,000 | 8,847 | 945 | 425 | 2,672 | 1,057 |
| 100,000 | 17,349 | 1,797 | 746 | 5,015 | 2,107 |

**Seasonal median**

| locations | chap-core | chapkit-rs, 1 core | chapkit-rs, all cores | chap-core MB | chapkit-rs MB |
|---|---|---|---|---|---|
| 1,000 | 320 | 17 | 13 | 172 | 41 |
| 5,000 | 1,575 | 81 | 50 | 327 | 172 |
| 10,000 | 3,116 | 164 | 94 | 511 | 329 |
| 50,000 | 15,677 | 894 | 438 | 2,131 | 1,610 |
| 100,000 | 31,498 | 1,701 | 889 | 4,250 | 3,205 |

**Seasonal shift persistence**

| locations | chap-core | chapkit-rs, 1 core | chapkit-rs, all cores | chap-core MB | chapkit-rs MB |
|---|---|---|---|---|---|
| 1,000 | 343 | 49 | 17 | 210 | 61 |
| 5,000 | 1,744 | 169 | 61 | 555 | 273 |
| 10,000 | 3,599 | 361 | 112 | 895 | 540 |
| 50,000 | 18,249 | 1,696 | 520 | 3,967 | 2,655 |
| 100,000 | 36,629 | 3,494 | 1,206 | 7,041 | 5,304 |

At 100,000 locations a seasonal median forecast takes chap-core 31.5 seconds, plus 45 seconds to convert the
data; the port takes 0.9 seconds on all cores, and decodes the data in 1.7 seconds.

## Long histories: 100 locations, up to 5,000 months

| model | size | chap-core | chapkit-rs, 1 core | chapkit-rs, all cores |
|---|---|---|---|---|
| global median | 100 x 5,000 | 15 | 5.5 | 4.0 |
| persistence | 100 x 5,000 | 77 | 35 | 9.3 |
| seasonal median | 100 x 5,000 | 100 | 21 | 9.7 |
| seasonal shift persistence | 100 x 5,000 | 169 | 44 | 14 |
| global median | 2,000 x 2,000 | 143 | 49 | 35 |

With few locations and very long series the gap is smallest: numpy handles one long array per location well.

## What this shows

- **Faster at every size.** On one core against chap-core's one core the ports are 2.2 to 24 times faster; on
  all cores 3.7 to 36 times. These are model-only numbers, the fair ones to quote.
- **The speedup holds as locations grow**: from 5,000 to 100,000 locations about 7, 9 to 10, 18 to 19 and 10 to
  11 times on one core (global median, persistence, seasonal median, seasonal shift persistence), and 7 to 8, 20
  to 23, 31 to 36 and 29 to 35 times on all cores. Both sides scale linearly, so the ports stay ahead by the same
  factor.
- **Less memory at every size**: about a tenth of chap-core's at 100 locations; at 100,000 locations about half
  (global median), two fifths (persistence) and three quarters (seasonal median, seasonal shift persistence).
- **All cores trade CPU for latency.** Running locations in parallel shortens a call but uses more CPU in total
  than one core (1.9 to 10 cores busy). A service answering many calls at once gets the most work per core with
  `RAYON_NUM_THREADS=1`.
- **The data step is context, not a speedup.** chap-core's conversion costs more than its models, and decoding
  the same data from JSON is far cheaper for the port. But when chap-core calls a service it still prepares and
  serializes its data, which this benchmark does not measure, so the two cannot simply be added up. The ports'
  times include grouping rows by location, which chap-core does during conversion.

## At facility level, the data format matters

At 100,000 locations a 10-year training request is 589 MB of JSON in chapkit's row format, and decoding it
takes 1.7 seconds, longer than any of the four models takes on all cores. For facility-level forecasting, a more
compact columnar format between chap-core and model services would save more time than any further model speedup.

## How the ports got here

The ports were profiled, not guessed at: parsing period strings, grouping rows by location (by comparing
strings) and looking up ratios in maps were the hot spots. Fixing those, storing strings as codes into a
dictionary of distinct values, and computing the pooled ratios of seasonal shift persistence only when needed
made one core about twice as fast and roughly halved memory at large sizes. polars was measured as an alternative for the
per-location work and was 3 to 4 times slower here, mostly because converting to it costs as much as a whole
training.

All runs are in `bench/scale_results.json`; `make bench-scale` repeats them, and `make bench-compare` checks the
outputs are the same ([details](verification.md)).
