#![allow(clippy::unwrap_used)]
//! The features must be the original's, number for number: the rows the forest is fitted on and
//! what each forecast starts from. The fixture comes from the original model's own code
//! (`scripts/gen_simple_multistep_fixtures.py`) on a slice of its example data with a gap in the
//! cases and in a covariate.

use chapkit::{DataFrame, Period};
use serde_json::Value;
use simple_multistep::{PredictTables, training_set};

fn fixture() -> Value {
    serde_json::from_str(include_str!("../fixtures/features.json")).unwrap()
}

fn frame(v: &Value) -> DataFrame {
    serde_json::from_value(v.clone()).unwrap()
}

fn numbers(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn training_rows_match_the_original() {
    let f = fixture();
    let lags = f["n_target_lags"].as_u64().unwrap() as usize;
    let covariates = strings(&f["covariates"]);
    let (locations, rows) =
        training_set(&frame(&f["historic"]), "disease_cases", &covariates, lags).unwrap();
    assert_eq!(locations, ["LA-AT", "LA-BK", "LA-BL"]);
    let expected = f["train"].as_array().unwrap();
    assert_eq!(rows.len(), expected.len());
    for (row, want) in rows.iter().zip(expected) {
        assert_eq!(row.location, want["location"].as_str().unwrap());
        assert_eq!(
            row.period.to_string(),
            want["time_period"].as_str().unwrap()
        );
        assert_eq!(
            row.x,
            numbers(&want["x"]),
            "{} {}",
            row.location,
            row.period
        );
        assert_eq!(row.y, want["y"].as_f64().unwrap());
    }
}

#[test]
fn forecast_inputs_match_the_original() {
    let f = fixture();
    let lags = f["n_target_lags"].as_u64().unwrap() as usize;
    let covariates = strings(&f["covariates"]);
    let future = frame(&f["future"]);
    let periods: Vec<Period> = {
        let mut p: Vec<Period> = (0..future.n_rows())
            .map(|r| {
                Period::parse(future.column("time_period").unwrap().str_at(r).unwrap()).unwrap()
            })
            .collect();
        p.sort();
        p.dedup();
        p
    };
    let tables = PredictTables::read(
        &frame(&f["historic"]),
        &future,
        "disease_cases",
        &covariates,
    )
    .unwrap();
    let n_cov = covariates.len() * 3;
    for (i, (location, want)) in f["predict"].as_object().unwrap().iter().enumerate() {
        let (previous, steps) = tables.inputs(location, &periods, lags).unwrap();
        assert_eq!(previous, numbers(&want["previous_y"]), "{location}");
        let exog = want["exog"].as_array().unwrap();
        assert_eq!(steps.len(), exog.len());
        for (step, want) in steps.iter().zip(exog) {
            let want = numbers(want);
            assert_eq!(step[..], want[..n_cov], "{location}");
            // The rest of the original's row is the location indicator.
            let indicator: Vec<f64> = (0..3).map(|j| f64::from(u8::from(j == i))).collect();
            assert_eq!(want[n_cov..], indicator[..]);
        }
    }
}
