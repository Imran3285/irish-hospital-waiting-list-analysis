"""Draws every chart used in the README, in one consistent style.

Reads the summary tables in data/ and writes images/01-08 (except 04, the
superseded Random Forest forecast). Run from the repository root, after
src/forecast_and_targets.py:   python src/charts.py
"""
import pandas as pd
import matplotlib.ticker as mtick

from chart_style import (BLUE, BLUE_80, BLUE_95, GREY, GREY_LIGHT, INK, INK2, TARGET,
                         new_figure, tidy, year_axis, save)

nat = pd.read_csv("data/national_monthly.csv", parse_dates=["month"])
tgt = pd.read_csv("data/target_benchmark.csv", parse_dates=["month"])
fc = pd.read_csv("data/forecast_with_intervals.csv", parse_dates=["month"])
mc = pd.read_csv("data/model_comparison.csv")
spec = pd.read_csv("data/op_specialty_sep2026.csv")

thousands = mtick.FuncFormatter(lambda v, _: f"{v:,.0f}k")
last = nat.iloc[-1]

# 01 Outpatient trend
fig, ax = new_figure(f"The outpatient list is at a 5-year high: {last.op_total:,.0f}",
                     "Patients waiting for a first outpatient appointment, April 2021 to September 2026")
tidy(ax)
ax.plot(nat.month, nat.op_total / 1000, color=BLUE, lw=2.2)
low = nat.loc[nat.op_total.idxmin()]
for row, text, dy, va in [(low, f"Low: {low.op_total:,.0f} ({low.month:%b %Y})", -14, "top"),
                          (last, f"{last.op_total:,.0f} ({last.month:%b %Y})", 10, "bottom")]:
    ax.scatter(row.month, row.op_total / 1000, color=BLUE, s=40, zorder=3, edgecolor="white", linewidth=1.5)
    ax.annotate(text, (row.month, row.op_total / 1000), xytext=(0, dy), textcoords="offset points",
                ha="center" if row is low else "right", va=va, fontsize=10, color=INK, fontweight="bold")
ax.set_ylim(520, 720); ax.yaxis.set_major_formatter(thousands); year_axis(ax)
save(fig, "images/01_outpatient_trend.png")

# 02 Long-wait share
op_share = (nat.op_12_18 + nat.op_18_plus) / nat.op_total * 100
ip_share = (nat.ipdc_12_18 + nat.ipdc_18_plus) / nat.ipdc_total * 100
fig, ax = new_figure(f"The share waiting over 12 months fell from {op_share.iloc[0]:.0f}% to {op_share.iloc[-1]:.0f}%",
                     "% of patients waiting more than 12 months")
tidy(ax)
ax.plot(nat.month, op_share, color=BLUE, lw=2.2)
ax.plot(nat.month, ip_share, color=GREY, lw=2)
ax.text(nat.month.iloc[2], op_share.iloc[2] + 1.5, "Outpatient", color=BLUE, fontsize=10.5, fontweight="bold")
ax.text(nat.month.iloc[2], ip_share.iloc[2] + 2.5, "Inpatient / day case", color=GREY, fontsize=10.5, fontweight="bold")
ax.set_ylim(0, 50); ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0)); year_axis(ax)
save(fig, "images/02_long_wait_share.png")

# 03 Top specialties
top = spec.head(15).iloc[::-1]
fig, ax = new_figure("Orthopaedics has the longest list; Dermatology the most long waiters",
                     "Top 15 outpatient specialties by patients waiting, September 2026", height=6.5)
tidy(ax, grid_axis="x")
ax.barh(top.specialty, top.total_sep2026 / 1000, color=GREY_LIGHT, height=0.68, label="Under 12 months")
ax.barh(top.specialty, top.over_12m_sep2026 / 1000, color=BLUE, height=0.68, label="Over 12 months")
for y, r in enumerate(top.itertuples()):
    ax.text(r.total_sep2026 / 1000 + 0.8, y, f"{r.total_sep2026:,.0f}", va="center", fontsize=9.5, color=INK2)
ax.set_xlim(0, 95); ax.xaxis.set_major_formatter(thousands)
ax.legend(loc="lower right", fontsize=10)
fig.subplots_adjust(left=0.2)
save(fig, "images/03_top_specialties.png")


# 05 / 06 Forecasts with intervals
def forecast_chart(key, label, hist_col, path, ylim):
    f = fc[fc.list == label]
    hist = nat[["month", hist_col]]
    end = f.iloc[-1]
    fig, ax = new_figure(f"{label} forecast: about {end.forecast / 1000:,.0f}k by {end.month:%B %Y}",
                         f"Holt-Winters forecast with 80% and 95% prediction ranges "
                         f"(80% range in {end.month:%b %Y}: {end.lower_80 / 1000:,.0f}k to {end.upper_80 / 1000:,.0f}k)")
    tidy(ax)
    ax.plot(hist.month, hist[hist_col] / 1000, color=BLUE, lw=2.2)
    xs = pd.concat([hist.month.tail(1), f.month])
    edge = hist[hist_col].iloc[-1] / 1000
    ax.fill_between(xs, [edge] + list(f.lower_95 / 1000), [edge] + list(f.upper_95 / 1000), color=BLUE_95, lw=0, label="95% range")
    ax.fill_between(xs, [edge] + list(f.lower_80 / 1000), [edge] + list(f.upper_80 / 1000), color=BLUE_80, lw=0, label="80% range")
    ax.plot(xs, [edge] + list(f.forecast / 1000), color=BLUE, lw=2, ls="--", label="Forecast")
    ax.axvline(hist.month.iloc[-1], color=GREY_LIGHT, lw=1)
    ax.text(hist.month.iloc[-1], ylim[1], "  Forecast", va="top", fontsize=9.5, color=INK2)
    ax.set_ylim(*ylim); ax.yaxis.set_major_formatter(thousands); year_axis(ax)
    ax.legend(loc="upper left", fontsize=10)
    save(fig, path, "Ranges built from the model's own backtest errors.")


forecast_chart("op", "Outpatient", "op_total", "images/05_op_forecast_intervals.png", (520, 740))
forecast_chart("ipdc", "Inpatient/Day Case", "ipdc_total", "images/06_ipdc_forecast_intervals.png", (60, 140))

# 07 Model comparison
fig, axes = new_figure("Holt-Winters was the most accurate model at every horizon, for both lists",
                       "Mean absolute % error by months ahead, rolling-origin backtest from 13 starting points", ncols=2)
greys = {"Random Forest": "#6f6d68", "Naive": "#a3a19b", "Drift": "#c9c7c1"}
for ax, lst in zip(axes, ["Outpatient", "Inpatient/Day Case"]):
    tidy(ax, lst)
    sub = mc[mc.list == lst]
    label_y = {}
    for r in sub.sort_values("mape_h6").itertuples():
        ys = [getattr(r, f"mape_h{h}") for h in range(1, 7)]
        hw = r.model == "Holt-Winters"
        ax.plot(range(1, 7), ys, color=BLUE if hw else greys[r.model], lw=2.6 if hw else 1.6,
                marker="o", ms=5 if hw else 3.5, zorder=3 if hw else 2)
        y = ys[-1]
        if label_y:  # keep end labels at least 0.38 points apart
            y = max(y, max(label_y.values()) + 0.38)
        label_y[r.model] = y
        ax.text(6.12, y, r.model, va="center", fontsize=9.5,
                color=BLUE if hw else INK2, fontweight="bold" if hw else "normal")
    ax.set_xlim(0.8, 7.6); ax.set_xticks(range(1, 7)); ax.set_xlabel("Months ahead")
    ax.set_ylim(0, 7.5); ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
save(fig, "images/07_model_comparison.png")

# 08 Targets
t = tgt.iloc[-1]
fig, axes = new_figure("Progress towards the national targets has stalled since December 2024",
                       f"September 2026: {t.op_pct_under_12m:.0f}% of outpatients wait under 12 months (target 90%); "
                       f"estimated average wait {t.op_weighted_avg_wait_months + 1e-6:.1f} months (target under 5.5)", ncols=2)
panels = [("% waiting under 12 months", "op_pct_under_12m", "ipdc_pct_under_12m", 90, "90% target", (50, 100)),
          ("Estimated average wait (months)", "op_weighted_avg_wait_months", "ipdc_weighted_avg_wait_months", 5.5,
           "5.5-month target", (4, 13))]
for ax, (title, op_col, ip_col, target, tlabel, ylim) in zip(axes, panels):
    tidy(ax, title)
    ax.plot(tgt.month, tgt[op_col], color=BLUE, lw=2.2, label="Outpatient")
    ax.plot(tgt.month, tgt[ip_col], color=GREY, lw=2, label="Inpatient / day case")
    ax.axhline(target, color=TARGET, lw=1.2, ls=(0, (4, 3)))
    ax.text(tgt.month.iloc[0], target + (ylim[1] - ylim[0]) * 0.015, tlabel, fontsize=9.5, color=INK2, va="bottom")
    ax.set_ylim(*ylim); year_axis(ax, 2)
    if op_col.startswith("op_pct"):
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.legend(loc="lower right", fontsize=10)
save(fig, "images/08_targets_benchmark.png", "Average wait estimated from wait-band midpoints (3, 9, 15, 24 months).")
