#![allow(clippy::unwrap_used)]
//! Train and predict end to end through chapkit, the checkpoint round trip, and the samples
//! against the original's forecast distribution.

use auto_arima::{AutoArima, Config};
use chapkit::{DataFrame, Model, PredictInput, RunInfo, TrainInput};
use serde_json::{Value, json};

fn fixture(name: &str) -> Value {
    let path = format!("{}/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The fixture's series as a chapkit table: the first `n` periods of each location as history,
/// and the `h` after them as the future (without cases).
fn frames(f: &Value, n: usize, h: usize) -> (DataFrame, DataFrame) {
    let (mut historic, mut future) = (Vec::new(), Vec::new());
    for (location, loc) in f.as_object().unwrap() {
        let periods = loc["periods"].as_array().unwrap();
        let values = loc["values"].as_array().unwrap();
        for i in 0..n {
            historic.push(json!([periods[i], location, values[i]]));
        }
        for p in &periods[n..n + h] {
            future.push(json!([p, location]));
        }
    }
    let frame = |columns: Value, data: Vec<Value>| {
        serde_json::from_value(json!({"columns": columns, "data": data})).unwrap()
    };
    (
        frame(
            json!(["time_period", "location", "disease_cases"]),
            historic,
        ),
        frame(json!(["time_period", "location"]), future),
    )
}

fn seeded(seed: u64) -> Config {
    Config {
        seed: Some(seed),
        ..Config::default()
    }
}

#[test]
fn trains_predicts_and_round_trips() {
    let f = fixture("fable_monthly.json");
    let (historic, future) = frames(&f, 27, 3);
    let run = RunInfo::with_periods(3);
    let model = AutoArima;
    let config = seeded(7);
    let trained = model
        .train(&config, TrainInput::new(&historic, &run, None))
        .unwrap();
    assert_eq!(trained.season, 12);
    assert_eq!(trained.locations.len(), 3);
    let predict = |t: &auto_arima::Trained| {
        model
            .predict(
                t,
                &config,
                PredictInput::new(&historic, &future, &run, None),
            )
            .unwrap()
    };
    let out = predict(&trained);
    assert_eq!(out.n_rows(), 9);
    assert!(out.has("sample_99") && !out.has("sample_100"));

    // The checkpoint gives the same model and so the same samples.
    let checkpoint = model.save(&trained).unwrap();
    let bytes = checkpoint.to_bytes().unwrap();
    let loaded = model
        .load(&chapkit::Checkpoint::from_bytes(&bytes).unwrap())
        .unwrap();
    assert_eq!(loaded, trained);
    assert_eq!(predict(&loaded), out);
}

/// The 100 samples of each forecast period are draws from the original's normal distribution:
/// fable's mean and sd after `refit()` on the history (the fixture's backtest at 27 periods).
#[test]
fn samples_follow_the_original_forecast() {
    let f = fixture("fable_monthly.json");
    let (historic, future) = frames(&f, 27, 3);
    let run = RunInfo::with_periods(3);
    let config = Config {
        n_samples: 20_000,
        seed: Some(1),
    };
    let model = AutoArima;
    let trained = model
        .train(&config, TrainInput::new(&historic, &run, None))
        .unwrap();
    let out = model
        .predict(
            &trained,
            &config,
            PredictInput::new(&historic, &future, &run, None),
        )
        .unwrap();
    let columns: Vec<_> = (0..config.n_samples)
        .map(|i| out.column(&format!("sample_{i}")).unwrap())
        .collect();
    let locations = out.column("location").unwrap();
    let periods = out.column("time_period").unwrap();
    for row in 0..out.n_rows() {
        let location = locations.str_at(row).unwrap();
        let period = periods.str_at(row).unwrap();
        let all = f[location]["periods"].as_array().unwrap();
        let step = all.iter().position(|x| x.as_str() == Some(period)).unwrap() - 27;
        let expected = &f[location]["backtest"][0]["forecast"];
        let (mean, sd) = (
            expected["mean"][step].as_f64().unwrap(),
            expected["sd"][step].as_f64().unwrap(),
        );
        let draws: Vec<f64> = columns.iter().map(|c| c.f64_at(row).unwrap()).collect();
        let n = draws.len() as f64;
        let m = draws.iter().sum::<f64>() / n;
        let var = draws.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
        // Within 4 standard errors of the mean, and the sd within 3%.
        assert!(
            (m - mean).abs() < 4.0 * sd / n.sqrt(),
            "{location} {step}: {m} vs {mean}"
        );
        assert!((var.sqrt() / sd - 1.0).abs() < 0.03, "{location} {step}");
    }
}

#[test]
fn weekly_data_uses_a_52_week_season() {
    let f = fixture("fable_weekly.json");
    let (historic, future) = frames(&f, 120, 2);
    let run = RunInfo::with_periods(2);
    let model = AutoArima;
    let trained = model
        .train(&seeded(3), TrainInput::new(&historic, &run, None))
        .unwrap();
    assert_eq!(trained.season, 52);
    let out = model
        .predict(
            &trained,
            &seeded(3),
            PredictInput::new(&historic, &future, &run, None),
        )
        .unwrap();
    assert_eq!(out.n_rows(), 8);
}

#[test]
fn rejects_an_unknown_location_and_no_samples() {
    let f = fixture("fable_monthly.json");
    let (historic, _) = frames(&f, 27, 3);
    let run = RunInfo::with_periods(3);
    let model = AutoArima;
    let trained = model
        .train(&seeded(1), TrainInput::new(&historic, &run, None))
        .unwrap();
    let future: DataFrame = serde_json::from_value(json!({
        "columns": ["time_period", "location"],
        "data": [["2012-04", "Nowhere"]]
    }))
    .unwrap();
    assert!(
        model
            .predict(
                &trained,
                &seeded(1),
                PredictInput::new(&historic, &future, &run, None)
            )
            .is_err()
    );
    let none = Config {
        n_samples: 0,
        seed: None,
    };
    assert!(
        model
            .train(&none, TrainInput::new(&historic, &run, None))
            .is_err()
    );
}

/// A configuration cannot ask for unbounded memory: `n_samples` is at most 100,000 (where the
/// configuration is read, and in `$validate`), and a forecast holds at most 250 million samples.
#[test]
fn bounds_the_samples() {
    for n in [0, 100_001, 1_000_000_000] {
        assert!(
            serde_json::from_value::<Config>(json!({"n_samples": n})).is_err(),
            "{n}"
        );
    }
    let most: Config = serde_json::from_value(json!({"n_samples": 100_000})).unwrap();
    assert_eq!(most.n_samples, 100_000);

    let f = fixture("fable_monthly.json");
    let (historic, future) = frames(&f, 27, 3);
    let run = RunInfo::with_periods(3);
    let model = AutoArima;
    let huge = Config {
        n_samples: 1_000_000_000,
        seed: Some(1),
    };
    let train = TrainInput::new(&historic, &run, None);
    let predict = PredictInput::new(&historic, &future, &run, None);
    assert_eq!(model.validate_train(&huge, train).len(), 1);
    assert_eq!(model.validate_predict(&huge, predict).len(), 1);
    assert!(model.train(&huge, train).is_err());
    let trained = model.train(&seeded(1), train).unwrap();
    assert!(model.predict(&trained, &huge, predict).is_err());
    assert_eq!(model.validate_predict(&seeded(1), predict).len(), 0);

    // Within the per-period bound, but over the budget for 2,501 future rows.
    let many: DataFrame = serde_json::from_value(json!({
        "columns": ["time_period", "location"],
        "data": (0..2501).map(|i| json!(["2012-04", format!("L{i}")])).collect::<Vec<_>>(),
    }))
    .unwrap();
    let most = Config {
        n_samples: 100_000,
        seed: Some(1),
    };
    let predict = PredictInput::new(&historic, &many, &run, None);
    assert_eq!(model.validate_predict(&most, predict).len(), 1);
    assert_eq!(model.validate_predict(&seeded(1), predict).len(), 0);
}

/// The weekly fixture with one more historic row for `boaco` at `period`.
fn weekly_with(period: &str) -> (auto_arima::Trained, DataFrame, DataFrame) {
    let f = fixture("fable_weekly.json");
    let (historic, future) = frames(&f, 120, 2);
    let run = RunInfo::with_periods(2);
    let trained = AutoArima
        .train(&seeded(3), TrainInput::new(&historic, &run, None))
        .unwrap();
    let mut rows: Value = serde_json::to_value(&historic).unwrap();
    rows["data"]
        .as_array_mut()
        .unwrap()
        .insert(0, json!([period, "boaco", 1.0]));
    (trained, serde_json::from_value(rows).unwrap(), future)
}

/// A gap-free grid is bounded before it is filled: a history whose periods lie far apart (here
/// 2,017 years of weeks, over 100,000) is refused, not expanded.
#[test]
fn far_apart_history_is_refused() {
    let run = RunInfo::with_periods(2);
    let (trained, historic, future) = weekly_with("0001W01");
    let err = AutoArima
        .predict(
            &trained,
            &seeded(3),
            PredictInput::new(&historic, &future, &run, None),
        )
        .unwrap_err();
    assert!(err.to_string().contains("100000 periods"), "{err}");
    // The same in training.
    let err = AutoArima
        .train(&seeded(3), TrainInput::new(&historic, &run, None))
        .unwrap_err();
    assert!(err.to_string().contains("100000 periods"), "{err}");
    // A century of weeks before the data is still taken.
    let (trained, historic, future) = weekly_with("1916W01");
    let out = AutoArima
        .predict(
            &trained,
            &seeded(3),
            PredictInput::new(&historic, &future, &run, None),
        )
        .unwrap();
    assert_eq!(out.n_rows(), 8);
}

/// A history of another resolution than the training data is refused before any filling (a month
/// sorts before every week, so filling months up to a week would never end).
#[test]
fn history_of_another_resolution_is_refused() {
    let run = RunInfo::with_periods(2);
    // A month in the weekly history.
    let (trained, historic, future) = weekly_with("2015-01");
    let err = AutoArima
        .predict(
            &trained,
            &seeded(3),
            PredictInput::new(&historic, &future, &run, None),
        )
        .unwrap_err();
    assert!(err.to_string().contains("weekly"), "{err}");
    // A week in a monthly history.
    let f = fixture("fable_monthly.json");
    let (historic, future) = frames(&f, 27, 3);
    let run = RunInfo::with_periods(3);
    let trained = AutoArima
        .train(&seeded(1), TrainInput::new(&historic, &run, None))
        .unwrap();
    let mut rows: Value = serde_json::to_value(&historic).unwrap();
    let location = rows["data"][0][1].clone();
    rows["data"]
        .as_array_mut()
        .unwrap()
        .push(json!(["2024W02", location, 2.0]));
    let historic: DataFrame = serde_json::from_value(rows).unwrap();
    let err = AutoArima
        .predict(
            &trained,
            &seeded(1),
            PredictInput::new(&historic, &future, &run, None),
        )
        .unwrap_err();
    assert!(err.to_string().contains("monthly"), "{err}");
}
