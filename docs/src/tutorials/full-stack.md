# Run the ports in a full Chap stack

The previous tutorial talks to chap-core directly. In a real deployment people use the ports from DHIS2: the
Modeling App sends a dataset to chap-core, chap-core trains and forecasts with a model, and the forecasts go back
to DHIS2. Here you run that whole stack (chap-core, DHIS2 and the Modeling App) with the ports registered next
to chap-core's own built-in baselines.

## The steps

The chapkit-rs runbook has every step, with what each printed when it was last run:
[Full stack: chap-core, DHIS2, Modeling App](https://winterop-com.github.io/chapkit-rs/guide/full-stack.html).
In short, with [chap-devops](https://github.com/dhis2-chap/chap-devops):

1. create and start a deployment with chap-core and DHIS2 (seeded with the Laos demo database: weekly dengue
   cases for 18 provinces);
2. add the four ports' images as models; they register with chap-core like any Chap model service;
3. connect DHIS2 to chap-core, which installs the Modeling App;
4. in the Modeling App, create datasets, run evaluations (backtests) and a prediction with the ports and with
   chap-core's built-in baselines.

## What the run showed

On 136 Laos districts, weekly, with the whole stack on one laptop:

- **Accuracy**: every metric (CRPS, MAE, RMSE, coverage, ...) of every port is identical to that of chap-core's
  built-in baseline to 10 decimals. The one exception, seasonal shift persistence's `ratio_above_truth` off by
  3e-6, came from a last-bit rounding difference that flipped one tie in about 350,000 samples; since 1.0.3 the
  port is bit for bit chap-core's ([Same output as chap-core](../verification.md)).
- **Prediction**: a 14.4 s chap-core job, of which the service's part was about 80 ms (train 22 ms, predict 46 ms,
  reading the result 12 ms).
- **Backtests**: 42 s per model, nearly all of it chap-core's overhead for each split; the service answered in
  under a second in total.

So in a full stack the ports give the built-ins' numbers, and the time a user waits is chap-core's.

## Next steps

- [Run the models in Docker](docker.md), as the stack above does.
- [Same output as chap-core](../verification.md): how the ports are checked number for number.
