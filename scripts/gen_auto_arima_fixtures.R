# Golden fixtures for crates/auto-arima from the original's own fable::ARIMA, run in its image
# (ghcr.io/chap-models/auto_arima_chapkit, fable 0.5.0, R 4.5.0). From the repository root:
#
#   IMAGE=ghcr.io/chap-models/auto_arima_chapkit@sha256:49f2b0ecce17cdb908606f4797bc93b6ac8b8c19e444a1f768226f764ff3314f
#   docker run --rm --platform linux/amd64 -v "$PWD":/r -w /r --entrypoint Rscript $IMAGE \
#     scripts/gen_auto_arima_fixtures.R DATA.csv crates/auto-arima/fixtures/OUT.json WEEKLY LENGTHS N0 SPLITS H
#
# monthly: laos_subset.csv (chap-core's example_data) fable_monthly.json 0 24,26,27,28,30,33,36 27 7 3
# weekly:  nicaragua_weekly4.csv fable_weekly.json 1 200,242 242 7 3 (chap-core's nicaragua_weekly_data.csv:
#          boaco, leon, managua, matagalpa, 2016-02-29 to 2020-12-20, 251 complete weeks)
#
# For each location (sorted) it records the input series, then:
# - fits: ARIMA(disease_cases) on the first n periods, as train.R fits it: the chosen orders,
#   coefficients, sigma2, AICc, and the h-step forecast mean and sd;
# - backtest: the fit on the first N0 periods refitted, as predict.R does (refit() keeps the
#   coefficients), on the first N0 + i periods for i in 0..SPLITS, and its h-step forecast.
# The original reads monthly periods only (yearmonth); weekly series get a yearweek index here.
suppressPackageStartupMessages({
  library(dplyr); library(fable); library(tsibble); library(distributional); library(jsonlite)
})
a <- commandArgs(trailingOnly = TRUE)
df <- read.csv(a[1])
weekly <- a[3] == "1"
lens <- as.integer(strsplit(a[4], ",")[[1]])
n0 <- as.integer(a[5]); splits <- as.integer(a[6]); h <- as.integer(a[7])
idx <- if (weekly) function(x) yearweek(as.Date(substr(x, 1, 10))) else function(x) yearmonth(x)

spec_of <- function(m) {
  f <- m[[1]][[1]]$fit
  s <- f$spec
  list(p = s$p, d = s$d, q = s$q, P = s$P, D = s$D, Q = s$Q, period = s$period, constant = s$constant,
       coef = as.list(setNames(f$par$estimate, f$par$term)), sigma2 = f$fit$sigma2, aicc = f$fit$AICc)
}
forecast_of <- function(m) {
  d <- forecast(m, h = h)$disease_cases
  list(mean = mean(d), sd = sqrt(variance(d)))
}

out <- list()
for (loc in sort(unique(df$location))) {
  x <- df[df$location == loc, c("time_period", "disease_cases")]
  x <- x[order(x$time_period), ]
  series <- x |> mutate(time_period = idx(time_period)) |> as_tsibble(index = time_period)
  fits <- lapply(lens, function(n) {
    m <- series[seq_len(n), ] |> model(ARIMA(disease_cases))
    c(list(n = n), spec_of(m), list(forecast = forecast_of(m)))
  })
  m0 <- series[seq_len(n0), ] |> model(ARIMA(disease_cases))
  backtest <- lapply(0:(splits - 1), function(i) {
    r <- refit(m0, series[seq_len(n0 + i), ])
    list(history = n0 + i, sigma2 = r[[1]][[1]]$fit$fit$sigma2, forecast = forecast_of(r))
  })
  out[[loc]] <- list(periods = x$time_period, values = x$disease_cases, fits = fits,
                     train = c(list(n = n0), spec_of(m0)), backtest = backtest)
}
write(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), a[2])
