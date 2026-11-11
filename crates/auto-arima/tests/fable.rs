#![allow(
    clippy::unwrap_used,
    clippy::unreadable_literal,
    clippy::excessive_precision
)]
//! The port against the original's own fable::ARIMA (`scripts/gen_auto_arima_fixtures.R`): the
//! orders it chooses, its forecasts after training, and its forecasts after `refit()` on longer
//! histories, monthly (Laos) and weekly (Nicaragua).

use std::collections::BTreeMap;

use auto_arima::model;
use chapkit::ts::Forecast;
use serde_json::Value;

fn fixture(name: &str) -> Value {
    let path = format!("{}/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect()
}

fn orders(v: &Value) -> ([usize; 6], bool) {
    (
        ["p", "d", "q", "P", "D", "Q"].map(|k| v[k].as_u64().unwrap() as usize),
        v["constant"].as_bool().unwrap(),
    )
}

fn orders_of(fit: &model::Fitted) -> ([usize; 6], bool) {
    let s = fit.spec;
    (
        [s.p, s.d, s.q, s.seasonal_p, s.seasonal_d, s.seasonal_q],
        s.constant,
    )
}

/// Largest relative difference (relative to the larger of |expected| and 1).
fn rel(got: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(got.len(), expected.len());
    got.iter()
        .zip(expected)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .fold(0.0, f64::max)
}

/// Relative differences of a 3-step forecast's means and sds from fable's.
fn differences(forecast: &Forecast, expected: &Value) -> (f64, f64) {
    let sd: Vec<f64> = forecast.variance.iter().map(|v| v.sqrt()).collect();
    (
        rel(&forecast.mean, &floats(&expected["mean"])),
        rel(&sd, &floats(&expected["sd"])),
    )
}

/// Every fit in the fixture must have fable's orders. Returns, per fit (`location/n`), the worst
/// relative difference of its forecast means and sds: after training and, for the model trained
/// on chap-core's training set, after each refit.
fn check(name: &str, season: usize) -> BTreeMap<String, (f64, f64)> {
    let f = fixture(name);
    let mut out = BTreeMap::new();
    for (location, loc) in f.as_object().unwrap() {
        let y = floats(&loc["values"]);
        for expected in loc["fits"].as_array().unwrap() {
            let n = expected["n"].as_u64().unwrap() as usize;
            let fit = model::fit(&y[..n], season).unwrap();
            assert_eq!(
                orders_of(&fit),
                orders(expected),
                "{location}/{n}: {}",
                fit.spec
            );
            let d = differences(&fit.forecast(3).unwrap(), &expected["forecast"]);
            println!(
                "{location}/{n} {}: mean {:.1e}, sd {:.1e}",
                fit.spec, d.0, d.1
            );
            out.insert(format!("{location}/{n}"), d);
        }
        let n0 = loc["train"]["n"].as_u64().unwrap() as usize;
        let trained = model::fit(&y[..n0], season).unwrap();
        assert_eq!(orders_of(&trained), orders(&loc["train"]));
        for split in loc["backtest"].as_array().unwrap() {
            let n = split["history"].as_u64().unwrap() as usize;
            let d = differences(
                &model::forecast(&trained, &y[..n], 3).unwrap(),
                &split["forecast"],
            );
            println!("  refit on {n}: mean {:.1e}, sd {:.1e}", d.0, d.1);
            let worst = out.entry(format!("{location}/{n0}")).or_insert((0.0, 0.0));
            *worst = (worst.0.max(d.0), worst.1.max(d.1));
        }
    }
    out
}

#[test]
fn monthly_matches_fable() {
    // 3 locations x 7 training lengths: every order as fable's, means and sds within 1e-5.
    for (fit, (mean, sd)) in check("fable_monthly.json", 12) {
        assert!(mean < 1e-5 && sd < 1e-5, "{fit}: {mean:e} {sd:e}");
    }
}

#[test]
fn weekly_matches_fable() {
    // 4 locations x 2 training lengths (200 weeks, and 242: chap-core's training set for 7 splits
    // of 3 weeks, refitted on 242 to 248 weeks): every order as fable's, means and sds within
    // 1e-5, except matagalpa/242, ARIMA(3,1,1)(1,0,1)[52]. For that model R's initial state
    // covariance (getQ0, Gardner 1980) is inaccurate (it has a negative eigenvalue); chapkit-ts
    // uses the exact covariance where Gardner's fails its own equation, so the likelihood, and with
    // it the coefficients and forecasts, differ from fable's. The orders and AICc agree, and R
    // itself gives coefficients 1e-2 apart on arm64 and x86 for this model.
    for (fit, (mean, sd)) in check("fable_weekly.json", 52) {
        let (tol_mean, tol_sd) = if fit == "matagalpa/242" {
            (1e-2, 5e-3)
        } else {
            (1e-5, 1e-5)
        };
        assert!(mean < tol_mean && sd < tol_sd, "{fit}: {mean:e} {sd:e}");
    }
}

/// A forecast from a shorter history than the training data, as chap-core's flow sends one: fable
/// refits ARIMA(1,1,0)(1,1,0)[12], trained on Bokeo's 36 months, on its first 20 months (R in the
/// original's image: `refit(model, history[1:20, ])`, `forecast(h = 3)`).
#[test]
fn refit_on_a_short_history_matches_fable() {
    let f = fixture("fable_monthly.json");
    let y = floats(&f["Bokeo"]["values"]);
    let trained = model::fit(&y, 12).unwrap();
    assert_eq!(trained.spec.to_string(), "ARIMA(1,1,0)(1,1,0)[12]");
    let fc = model::forecast(&trained, &y[..20], 3).unwrap();
    let expected = serde_json::json!({
        "mean": [26.915721112410143, 22.784295068587959, 18.423378385624076],
        "sd": [2.4339038870172334, 2.8172607367800619, 3.3643678266008825],
    });
    let (mean, sd) = differences(&fc, &expected);
    // 1e-5, as for the other fits: the optimum moves in the last digits between platforms (on
    // Linux x86_64 the sd differs from fable's by 1.1e-6, on macOS arm64 by less than 1e-6).
    assert!(mean < 1e-5 && sd < 1e-5, "{mean:e} {sd:e}");
}
