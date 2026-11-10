# Run a baseline and read its forecast

You start a model as a service, give it training data, train it, and ask for a forecast, with the same HTTP
calls chap-core makes. Along the way you see what the data and the forecast look like.

## Before you start

- The models built ([Set up your machine](setup.md)).
- `curl` and `jq` (`brew install jq` or `apt install jq`).

## 1. Start the service

```sh
PORT=8001 bin/baseline-seasonal-median
```

Leave it running and open a second terminal. Check it is up:

```sh
curl -s localhost:8001/health
```

```json
{"status":"healthy","checks":{"database":{"state":"healthy"}}}
```

The API documentation is at <http://localhost:8001/docs>.

## 2. Get some data

Every service can generate realistic sample data. Three locations, 36 months:

```sh
B=http://localhost:8001/api/v1
curl -s "$B/ml/\$generate-sample-data?num_locations=3&num_periods=36&seed=1" > train.json
jq '{columns: .data.columns, rows: (.data.data | length), first: .data.data[0]}' train.json
```

```json
{
  "columns": ["time_period", "location", "disease_cases", "population", "rainfall",
              "mean_temperature", "feature_0", "feature_1", "feature_2"],
  "rows": 108,
  "first": ["2020-01", "location_0", 15.0, 558187, 40.66, 19.75, 10.07, 44.19, 83.77]
}
```

This is the table chap-core sends: one row per location and month. The seasonal median only uses
`time_period`, `location` and `disease_cases`. Then the data to forecast from: the same history, plus the months
to forecast (with `disease_cases` empty):

```sh
curl -s "$B/ml/\$generate-sample-data?kind=predict&num_locations=3&num_periods=36&seed=1" > predict.json
jq '.future.data[0]' predict.json
```

```json
["2023-01", "location_0", null, 558187, 0.32, 19.76, 11.67, 56.81, 86.95]
```

## 3. Create a configuration

chap-core always trains a configured model. The seasonal median has no options, so the configuration is empty:

```sh
CONFIG=$(curl -s -X POST $B/configs -H 'content-type: application/json' \
  -d '{"name": "tutorial", "data": {}}' | jq -r .id)
```

## 4. Train

```sh
jq -n --arg c $CONFIG --slurpfile t train.json '{config_id: $c, data: $t[0].data}' \
  | curl -s -X POST "$B/ml/\$train" -H 'content-type: application/json' -d @- | tee job.json
```

```json
{"job_id":"01M72ENJ81GKYVRBE06Y1X3D04","artifact_id":"01M72ENJ810JKFE1ZYDKTFT31E","message":"Training job submitted. ..."}
```

Training runs as a job; `artifact_id` is where the trained model will be stored. Check the job:

```sh
curl -s $B/jobs/$(jq -r .job_id job.json) | jq '{status, started_at, finished_at}'
```

```json
{"status": "completed", "started_at": "2026-11-10T10:04:43.649934Z", "finished_at": "2026-11-10T10:04:43.651326Z"}
```

It finished in under 2 milliseconds, before the first check.

## 5. Forecast

chap-core asks for a forecast as a job too (`$predict`, then reads the result artifact). chapkit-rs also answers
directly, which is easier to try:

```sh
MODEL=$(jq -r .artifact_id job.json)
jq -n --arg m $MODEL --slurpfile p predict.json \
  '{artifact_id: $m, historic: $p[0].historic, future: $p[0].future}' > forecast-request.json
curl -s -X POST "$B/ml/\$predict-sync" -H 'content-type: application/json' -d @forecast-request.json > forecast.json
jq '{columns: (.columns[:4] + ["..."]), rows: (.data | length), first: (.data[0][:4])}' forecast.json
```

```json
{"columns": ["time_period", "location", "sample_0", "sample_1", "..."], "rows": 108,
 "first": ["2023-01", "location_0", 13.02020202020202, 13.06060606060606]}
```

## 6. Read the forecast

Each row is one location and month, with 100 samples (`sample_0` to `sample_99`): plausible values for that
month's cases. The median of a row is the point forecast; the spread of the samples is the uncertainty.

```sh
jq -r '.data[] | [.[0], .[1], ((.[2:] | sort | .[49] + .[50]) / 2)] | @tsv' forecast.json | head -4
```

```text
2023-01	location_0	15
2023-01	location_1	56
2023-01	location_2	73
2023-02	location_0	27
```

prints month, location and median for the first rows. For the seasonal median, a January forecast is the median
of that location's earlier Januaries: for `location_0` those were 15, 17 and 13 (2020 to 2022), so the median is
15 ([how](../models/seasonal-median.md)). Check it yourself:

```sh
jq -c '[.data.data[] | select(.[1] == "location_0" and (.[0] | endswith("-01"))) | [.[0], .[2]]]' train.json
```

## If something goes wrong

- **`Connection refused`**: the service is not running, or on another port (`PORT`).
- **`{"type":"urn:servicekit:error:validation-failed", ...}`**: the request body is not what the endpoint
  expects; the `errors` field says which field.

## Next steps

- [Run the baselines through chap-core](chap-core.md)
- The [seasonal median page](../models/seasonal-median.md) explains every number in the forecast.
