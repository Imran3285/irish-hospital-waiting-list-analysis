"""
Forecast model comparison, prediction intervals and wait-time target benchmarking
Author: Muhammad Imran

Runs entirely from data/national_monthly.csv (no raw files needed):
    pip install pandas numpy scipy scikit-learn matplotlib
    python src/forecast_and_targets.py

Part A - Forecasting
  * Four models for each list (outpatient, inpatient/day case):
      1. Naive              - next month = this month
      2. Drift              - this month + average monthly change so far
      3. Holt-Winters       - additive trend + 12-month seasonality (fitted with scipy)
      4. Random Forest      - predicts the monthly change from lagged changes
  * Rolling-origin backtest: 13 forecast origins, horizons 1-6 months, every model
    sees only data available at that origin.
  * Best model = lowest average error over the 6-month horizon.
  * 80% / 95% prediction intervals from the best model's own backtest errors
    at each horizon (empirical, no distribution assumed).

Part B - Benchmarking against Irish wait-time targets
  * % of patients waiting under 12 months   vs 90% target (WLAP 2025, outpatients)
  * Estimated weighted average wait          vs < 5.5 months target (WTAP 2026)
    (band midpoints 3, 9, 15, 24 months; checked against the published figure)
  * Sláintecare 10-week (OPD) / 12-week (IPDC) maximums cannot be measured with
    6-month bands; the 0-6 month share is reported as an upper bound.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor

DATA = "data/national_monthly.csv"
IMG = "images"
OUT = "data"
H = 6                 # forecast horizon (months)
N_ORIGINS = 13        # rolling-origin backtest origins
SEED = 42
os.makedirs(IMG, exist_ok=True)

LISTS = {"op": "Outpatient", "ipdc": "Inpatient/Day Case"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load():
    d = pd.read_csv(DATA)
    d["month"] = pd.PeriodIndex(d["month"], freq="M")
    d = d.set_index("month").sort_index()
    full = pd.period_range(d.index.min(), d.index.max(), freq="M")
    missing = [str(m) for m in full if m not in d.index]
    d = d.reindex(full).interpolate()            # June 2021 (HSE cyberattack) only
    return d, missing


# ---------------------------------------------------------------------------
# Models: each takes a history (np.array) and returns H forecasts
# ---------------------------------------------------------------------------
def naive(y):
    return np.repeat(y[-1], H)


def drift(y):
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + slope * np.arange(1, H + 1)


def holt_winters(y, m=12):
    """Additive Holt-Winters; smoothing parameters chosen by minimising
    one-step-ahead squared error on the history."""
    y = np.asarray(y, float)

    def run(params, forecast=False):
        a, b, g = params
        level = y[:m].mean()
        trend = (y[m:2 * m].mean() - y[:m].mean()) / m
        season = list(y[:m] - level)
        sse = 0.0
        for t in range(len(y)):
            s = season[t % m]
            pred = level + trend + s
            sse += (y[t] - pred) ** 2
            new_level = a * (y[t] - s) + (1 - a) * (level + trend)
            trend = b * (new_level - level) + (1 - b) * trend
            season[t % m] = g * (y[t] - new_level) + (1 - g) * s
            level = new_level
        if forecast:
            n = len(y)
            return np.array([level + h * trend + season[(n + h - 1) % m] for h in range(1, H + 1)])
        return sse

    best = minimize(run, x0=[0.5, 0.1, 0.1], bounds=[(0.01, 0.99)] * 3, method="L-BFGS-B")
    return run(best.x, forecast=True)


def rf_features(changes, months):
    """Lagged monthly changes + 3-month rolling mean + calendar month."""
    df = pd.DataFrame({"chg": changes, "month": months})
    for lag in (1, 2, 3, 6, 12):
        df[f"lag{lag}"] = df["chg"].shift(lag)
    df["roll3"] = df["chg"].shift(1).rolling(3).mean()
    return df


def random_forest(y, months):
    """Predicts the month-to-month change (a Random Forest cannot extrapolate
    beyond the training range, and the list is at record levels), then adds it
    to the last known value. Multi-step forecasts are built recursively."""
    y = list(np.asarray(y, float))
    months = list(months)
    for _ in range(H):
        changes = np.diff(y, prepend=np.nan)
        df = rf_features(changes, months).dropna()
        X, t = df.drop(columns="chg"), df["chg"]
        model = RandomForestRegressor(n_estimators=300, random_state=SEED, min_samples_leaf=2)
        model.fit(X, t)
        nxt_month = (months[-1] % 12) + 1
        row = rf_features(np.append(changes, np.nan), months + [nxt_month]).iloc[[-1]].drop(columns="chg")
        y.append(y[-1] + model.predict(row)[0])
        months.append(nxt_month)
    return np.array(y[-H:])


MODELS = {"Naive": naive, "Drift": drift, "Holt-Winters": holt_winters, "Random Forest": random_forest}


def forecast(name, y, months):
    f = MODELS[name]
    return f(y, months) if name == "Random Forest" else f(y)


# ---------------------------------------------------------------------------
# Part A - backtest, model selection, intervals, final forecast
# ---------------------------------------------------------------------------
def backtest(y, months):
    """Rolling-origin evaluation. Returns long table of percentage errors."""
    n = len(y)
    rows = []
    first_origin = n - H - N_ORIGINS + 1
    for origin in range(first_origin, n - H + 1):          # history = y[:origin]
        hist, actual = y[:origin], y[origin:origin + H]
        for name in MODELS:
            pred = forecast(name, hist, months[:origin])
            for h in range(H):
                rows.append({"model": name, "origin": origin, "h": h + 1,
                             "pct_error": (pred[h] - actual[h]) / actual[h] * 100})
    return pd.DataFrame(rows)


def part_a(d):
    months = list(d.index.month)
    summary, forecasts = [], []
    for key, label in LISTS.items():
        y = d[f"{key}_total"].to_numpy(float)
        bt = backtest(y, months)
        bt["abs_pct_error"] = bt["pct_error"].abs()
        by_h = bt.pivot_table(index="model", columns="h", values="abs_pct_error", aggfunc="mean")
        by_h["avg_1_6"] = by_h.mean(axis=1)
        best = by_h["avg_1_6"].idxmin()
        for model, r in by_h.iterrows():
            summary.append({"list": label, "model": model,
                            **{f"mape_h{h}": round(r[h], 2) for h in range(1, H + 1)},
                            "mape_avg_1_6": round(r["avg_1_6"], 2), "selected": model == best})

        # empirical intervals from the selected model's errors at each horizon
        errs = bt[bt.model == best]
        q80 = errs.groupby("h")["abs_pct_error"].quantile(0.80).to_numpy() / 100
        q95 = errs.groupby("h")["abs_pct_error"].quantile(0.95).to_numpy() / 100
        point = forecast(best, y, months)
        future = pd.period_range(d.index[-1] + 1, periods=H, freq="M")
        for i, m in enumerate(future):
            forecasts.append({"list": label, "month": str(m), "model": best,
                              "forecast": round(point[i]),
                              "lower_80": round(point[i] * (1 - q80[i])), "upper_80": round(point[i] * (1 + q80[i])),
                              "lower_95": round(point[i] * (1 - q95[i])), "upper_95": round(point[i] * (1 + q95[i]))})
    summary = pd.DataFrame(summary)
    forecasts = pd.DataFrame(forecasts)
    summary.to_csv(f"{OUT}/model_comparison.csv", index=False)
    forecasts.to_csv(f"{OUT}/forecast_with_intervals.csv", index=False)
    return summary, forecasts


# ---------------------------------------------------------------------------
# Part B - target benchmarking
# ---------------------------------------------------------------------------
MID = {"0_6": 3, "6_12": 9, "12_18": 15, "18_plus": 24}     # months; 18+ assumed 24


def part_b(d, missing):
    b = d.drop(index=[pd.Period(m, freq="M") for m in missing])   # real months only
    out = pd.DataFrame(index=b.index.astype(str))
    for key in LISTS:
        tot = b[f"{key}_total"]
        out[f"{key}_pct_under_12m"] = ((b[f"{key}_0_6"] + b[f"{key}_6_12"]) / tot * 100).round(1)
        out[f"{key}_pct_under_6m"] = (b[f"{key}_0_6"] / tot * 100).round(1)
        out[f"{key}_weighted_avg_wait_months"] = (sum(b[f"{key}_{band}"] * mid for band, mid in MID.items()) / tot).round(2)
        # patients that would need to move under 12 months to hit 90%
        out[f"{key}_gap_to_90pct"] = np.ceil(np.maximum(0, 0.9 * tot - (b[f"{key}_0_6"] + b[f"{key}_6_12"]))).astype(int)
    out.index.name = "month"
    out.to_csv(f"{OUT}/target_benchmark.csv")
    return out


if __name__ == "__main__":
    data, missing_months = load()
    print("Interpolated months:", missing_months)
    comparison, fc = part_a(data)
    print(comparison.to_string(index=False))
    print(fc.to_string(index=False))
    targets = part_b(data, missing_months)
    print(targets.loc[["2021-09", "2024-12", "2025-12", "2026-09"]].T.to_string())

    # all charts are drawn by charts.py so they share one style
    import charts
    charts.main()
