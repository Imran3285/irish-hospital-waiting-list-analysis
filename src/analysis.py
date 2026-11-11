"""
Irish Hospital Waiting List Analysis (NTPF Open Data)
Author: Muhammad Imran

What this script does
1. Loads every NTPF waiting list CSV in the raw_data/ folder (old and new formats).
2. Cleans and standardises columns, dates, specialties and time bands.
3. Answers key questions:
   - How has the national waiting list changed over time?
   - Which specialties and hospitals have the most patients waiting?
   - What share of patients wait longer than 12 months?
4. Trains a Random Forest model on monthly totals and forecasts the next 6 months.
5. Exports clean tables (for Power BI) and charts (for GitHub / LinkedIn).

How to run
    pip install pandas scikit-learn matplotlib
    python src/analysis.py
Download the NTPF CSVs (OpenData_OPNational01/02, OpenData_IPDCNational01/02) into a folder called raw_data/ and run from the repository root: python src/analysis.py
"""

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

DATA_DIR = "raw_data"
OUT_DIR = "outputs"
FORECAST_MONTHS = 6
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load and standardise
# ---------------------------------------------------------------------------
# NTPF changed its column names over the years, so we map all versions to one set.
COLUMN_MAP = {
    "archive_date": "date", "archivedate": "date",
    "group": "hospital_group", "hospital_group": "hospital_group",
    "hospital": "hospital", "hospitalname": "hospital", "hospital_name": "hospital",
    "speciality": "specialty", "specialty": "specialty", "specialty_name": "specialty",
    "adult_child": "adult_child",
    "age_profile": "age_profile", "age_categorie": "age_profile", "age_category": "age_profile",
    "time_bands": "time_band", "time_band": "time_band",
    "total": "total",
    "case_type": "case_type",
}


def clean_name(col):
    return re.sub(r"[^a-z0-9_+\-]", "", col.strip().lower().replace(" ", "_"))


def load_file(path):
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df.columns = [clean_name(c) for c in df.columns]
    df = df.rename(columns={c: COLUMN_MAP[c] for c in df.columns if c in COLUMN_MAP})

    # Some newer files are "wide": one column per time band (0-6 Months, 6-12 Months...).
    band_cols = [c for c in df.columns if "month" in c and c != "date"]
    if "time_band" not in df.columns and band_cols:
        df = df.drop(columns=["total"], errors="ignore")
        id_cols = [c for c in df.columns if c not in band_cols]
        df = df.melt(id_vars=id_cols, value_vars=band_cols,
                     var_name="time_band", value_name="total")

    name = os.path.basename(path).lower()
    df["list_type"] = "Inpatient/Day Case" if ("ipdc" in name or "inpatient" in name) else "Outpatient"
    # Each file uses its own date format, so parse dates per file
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed", dayfirst=True)
    df["source_file"] = os.path.basename(path)
    # NTPF publishes the same national list twice: by hospital (National01) and by specialty (National02).
    # Keep track of which, so totals are not double counted.
    df["view"] = "Hospital" if "hospital" in df.columns else "Specialty"
    return df


files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
if not files:
    raise SystemExit(f"No CSV files found in '{DATA_DIR}/'. Download them from https://www.ntpf.ie/waiting-list-data/open-data/")

df = pd.concat([load_file(f) for f in files], ignore_index=True)

df["total"] = pd.to_numeric(df["total"], errors="coerce")
df = df.dropna(subset=["date", "total"])
df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
for col in ["hospital", "specialty", "time_band", "adult_child"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()


def band_over_12_months(band):
    """True if a time band means the patient has waited more than 12 months."""
    b = str(band).lower()
    nums = [int(n) for n in re.findall(r"\d+", b)]
    if not nums:
        return False
    if "+" in b or "over" in b or ">" in b:
        return nums[0] >= 12
    return min(nums) >= 12


if "time_band" in df.columns:
    df["over_12_months"] = df["time_band"].apply(band_over_12_months)
else:
    df["over_12_months"] = False

print(f"Loaded {len(files)} files, {len(df):,} rows, "
      f"{df['month'].min():%b %Y} to {df['month'].max():%b %Y}")


# ---------------------------------------------------------------------------
# 2. Key findings
# ---------------------------------------------------------------------------
hosp_view = df[df["view"] == "Hospital"]
spec_view = df[df["view"] == "Specialty"]
monthly = (hosp_view.groupby(["list_type", "month"])["total"].sum()
             .reset_index().sort_values("month"))

# Latest snapshot for each list type (outpatient and inpatient files can end in different months)
latest = df[df["month"] == df.groupby("list_type")["month"].transform("max")]
latest_month = df["month"].max()

by_specialty = (latest[latest["view"] == "Specialty"].groupby(["list_type", "specialty"])["total"].sum()
                      .reset_index().sort_values("total", ascending=False))
by_hospital = (latest[latest["view"] == "Hospital"].groupby(["list_type", "hospital"])["total"].sum()
                     .reset_index().sort_values("total", ascending=False))

long_wait = (hosp_view.groupby(["list_type", "month"])
               .apply(lambda g: g.loc[g["over_12_months"], "total"].sum() / g["total"].sum() * 100,
                      include_groups=False)
               .reset_index(name="pct_over_12_months"))

print("\n=== KEY FINDINGS (use these in your CV) ===")
for lt in monthly["list_type"].unique():
    m = monthly[monthly["list_type"] == lt]
    first, last = m.iloc[0], m.iloc[-1]
    change = (last["total"] - first["total"]) / first["total"] * 100
    lw = long_wait[(long_wait["list_type"] == lt) & (long_wait["month"] == last["month"])]
    print(f"\n[{lt}]")
    print(f"  Waiting list {first['month']:%b %Y}: {first['total']:,.0f}  ->  "
          f"{last['month']:%b %Y}: {last['total']:,.0f}  ({change:+.1f}%)")
    if not lw.empty and lw["pct_over_12_months"].iloc[0] > 0:
        print(f"  Patients waiting over 12 months: {lw['pct_over_12_months'].iloc[0]:.1f}%")
    top_s = by_specialty[by_specialty["list_type"] == lt].head(3)
    print("  Top 3 specialties: " + ", ".join(f"{r.specialty} ({r.total:,.0f})" for r in top_s.itertuples()))
    top_h = by_hospital[by_hospital["list_type"] == lt].head(3)
    print("  Top 3 hospitals:   " + ", ".join(f"{r.hospital} ({r.total:,.0f})" for r in top_h.itertuples()))


# ---------------------------------------------------------------------------
# 3. Random Forest forecast of national monthly totals
# ---------------------------------------------------------------------------
def make_features(series):
    feat = pd.DataFrame({"y": series})
    for lag in (1, 2, 3, 6, 12):
        feat[f"lag_{lag}"] = feat["y"].shift(lag)
    feat["rolling_3"] = feat["y"].shift(1).rolling(3).mean()
    feat["month_of_year"] = feat.index.month
    return feat


forecasts = []
print("\n=== FORECAST ===")
# A Random Forest cannot predict values outside the range it was trained on, so it is
# trained on the month-to-month CHANGE in the list, which is then added to the last known level.
for lt in monthly["list_type"].unique():
    s = monthly[monthly["list_type"] == lt].set_index("month")["total"].asfreq("MS").interpolate()
    change = s.diff()
    feat = make_features(change).dropna()
    if len(feat) < 24:
        print(f"[{lt}] Not enough monthly history for a forecast ({len(feat)} usable months).")
        continue
    X, y = feat.drop(columns="y"), feat["y"]

    # Walk-forward test: predict each of the last 6 months using only the months before it
    preds, naive, actual = [], [], []
    for i in range(len(feat) - 6, len(feat)):
        m = RandomForestRegressor(n_estimators=300, random_state=42).fit(X.iloc[:i], y.iloc[:i])
        prev_level = s.iloc[s.index.get_loc(feat.index[i]) - 1]
        preds.append(prev_level + m.predict(X.iloc[[i]])[0])
        naive.append(prev_level)                      # baseline: next month = this month
        actual.append(s.loc[feat.index[i]])
    actual = np.array(actual)
    rf_err = np.mean(np.abs(actual - preds) / actual) * 100
    naive_err = np.mean(np.abs(actual - naive) / actual) * 100

    model = RandomForestRegressor(n_estimators=300, random_state=42).fit(X, y)
    hist, level = change.copy(), s.iloc[-1]
    for _ in range(FORECAST_MONTHS):
        nm = hist.index[-1] + pd.offsets.MonthBegin(1)
        hist.loc[nm] = np.nan
        step = model.predict(make_features(hist).iloc[[-1]].drop(columns="y"))[0]
        hist.loc[nm] = step
        level += step
        forecasts.append({"list_type": lt, "month": nm, "forecast": round(level)})

    print(f"[{lt}] Test error: Random Forest {rf_err:.2f}% vs repeat-last-month {naive_err:.2f}%")
    print(f"  Forecast for {nm:%b %Y}: {level:,.0f} patients")

forecast_df = pd.DataFrame(forecasts)


# ---------------------------------------------------------------------------
# 4. Exports for Power BI and charts
# ---------------------------------------------------------------------------
df.to_csv(f"{OUT_DIR}/clean_waiting_list.csv", index=False)
monthly.to_csv(f"{OUT_DIR}/monthly_totals.csv", index=False)
by_specialty.to_csv(f"{OUT_DIR}/latest_by_specialty.csv", index=False)
by_hospital.to_csv(f"{OUT_DIR}/latest_by_hospital.csv", index=False)
long_wait.to_csv(f"{OUT_DIR}/long_wait_share.csv", index=False)
forecast_df.to_csv(f"{OUT_DIR}/forecast.csv", index=False)

for lt in monthly["list_type"].unique():
    m = monthly[monthly["list_type"] == lt]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(m["month"], m["total"], color="#1f5f8b", linewidth=2, label="Actual")
    f = forecast_df[forecast_df["list_type"] == lt] if not forecast_df.empty else pd.DataFrame()
    if not f.empty:
        ax.plot(f["month"], f["forecast"], color="#e07b39", linestyle="--", linewidth=2, label="Forecast")
    ax.set_title(f"Irish {lt} Waiting List: Actual and Forecast")
    ax.set_ylabel("Patients waiting")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/trend_{lt.split('/')[0].lower()}.png", dpi=150)
    plt.close(fig)

    top = by_specialty[by_specialty["list_type"] == lt].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["specialty"], top["total"], color="#1f5f8b")
    ax.set_title(f"Top 10 Specialties by Patients Waiting ({lt}, {m['month'].max():%b %Y})")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/top_specialties_{lt.split('/')[0].lower()}.png", dpi=150)
    plt.close(fig)

print(f"\nDone. Clean tables and charts saved in '{OUT_DIR}/'.")
