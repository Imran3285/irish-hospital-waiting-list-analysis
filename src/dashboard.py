"""One-page dashboard of the Irish hospital waiting list (Sep 2026).

Reads the summary tables in data/ and writes images/00_dashboard.png.
Run from the repository root:  python src/dashboard.py
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.gridspec import GridSpec

from chart_style import BLUE, FORECAST, FORECAST_80 as BLUE_LIGHT, INK, INK2, GRID, SURFACE, TARGET  # shared look

nat = pd.read_csv("data/national_monthly.csv", parse_dates=["month"])
tgt = pd.read_csv("data/target_benchmark.csv", parse_dates=["month"])
fc = pd.read_csv("data/forecast_with_intervals.csv", parse_dates=["month"])
spec = pd.read_csv("data/op_specialty_sep2026.csv")
hosp = pd.read_csv("data/op_hospital_sep2026.csv")

last, year_ago = nat.iloc[-1], nat[nat.month == nat.month.iloc[-1] - pd.DateOffset(years=1)].iloc[0]
t_last = tgt.iloc[-1]
fc_op = fc[fc.list == "Outpatient"]


def pct(a, b):
    return (a - b) / b * 100


fig = plt.figure(figsize=(16, 10), facecolor=SURFACE)
gs = GridSpec(1, 4, figure=fig, wspace=0.12, left=0.04, right=0.98, top=0.875, bottom=0.775)

fig.text(0.04, 0.955, "Irish Hospital Waiting Lists: September 2026", fontsize=20, fontweight="bold", color=INK)
fig.text(0.04, 0.925, "The outpatient list is at a 5-year high, and waits are drifting away from the 90% target",
         fontsize=12, color=INK2)

# KPI cards
cards = [
    ("Outpatients waiting", f"{last.op_total:,.0f}", f"{pct(last.op_total, year_ago.op_total):+.1f}% vs Sep 2025"),
    ("Inpatient / day case waiting", f"{last.ipdc_total:,.0f}", f"{pct(last.ipdc_total, year_ago.ipdc_total):+.1f}% vs Sep 2025"),
    ("Outpatients waiting under 12 months", f"{t_last.op_pct_under_12m:.1f}%",
     f"Target 90%  ·  {t_last.op_gap_to_90pct:,.0f} patients short"),
    ("Estimated average outpatient wait", f"{t_last.op_weighted_avg_wait_months + 1e-6:.1f} months", "Target under 5.5 months"),
]
for i, (label, value, note) in enumerate(cards):
    ax = fig.add_subplot(gs[0, i]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="white", edgecolor=GRID, lw=1))
    ax.add_patch(plt.Rectangle((0, 0), 0.012, 1, transform=ax.transAxes, facecolor=BLUE, lw=0))
    ax.text(0.06, 0.74, label, fontsize=10.5, color=INK2, transform=ax.transAxes)
    ax.text(0.06, 0.34, value, fontsize=21, fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.06, 0.1, note, fontsize=9.5, color=INK2, transform=ax.transAxes)


def tidy(ax, title):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)


# Trend + forecast
ax = fig.add_axes([0.06, 0.44, 0.62, 0.26]); tidy(ax, "Outpatient list since 2021, with forecast to March 2027 (80% range)")
ax.plot(nat.month, nat.op_total / 1000, color=BLUE, lw=2.2)
ax.fill_between(fc_op.month, fc_op.lower_80 / 1000, fc_op.upper_80 / 1000, color=BLUE_LIGHT, lw=0)
ax.plot(pd.concat([nat.month.tail(1), fc_op.month]), pd.concat([nat.op_total.tail(1), fc_op.forecast]) / 1000,
        color=FORECAST, lw=2.2, ls="--")
low = nat.loc[nat.op_total.idxmin()]
ax.scatter([low.month, last.month], [low.op_total / 1000, last.op_total / 1000], color=BLUE, s=36, zorder=3,
           edgecolor=SURFACE, linewidth=1.5)
ax.annotate(f"Low: {low.op_total:,.0f}\n{low.month:%b %Y}", (low.month, low.op_total / 1000), xytext=(0, -38),
            textcoords="offset points", ha="center", fontsize=9.5, color=INK2)
ax.annotate(f"{last.op_total:,.0f}", (last.month, last.op_total / 1000), xytext=(-8, 10),
            textcoords="offset points", ha="right", fontsize=10, fontweight="bold", color=INK)
ax.axvline(last.month, color=GRID, lw=1)
ax.text(last.month + pd.Timedelta(days=12), 525, "Forecast\n(Holt-Winters)", fontsize=9, color=INK2)
ax.set_ylabel("Patients (thousands)"); ax.set_ylim(500, 740)
ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# % under 12 months vs target
ax = fig.add_axes([0.75, 0.44, 0.23, 0.26]); tidy(ax, "Under 12 months vs 90% target")
ax.plot(tgt.month, tgt.op_pct_under_12m, color=BLUE, lw=2.2)
ax.axhline(90, color=TARGET, lw=1.2, ls=(0, (4, 3)))
ax.text(tgt.month.iloc[0], 91.2, "90% target", fontsize=9.5, color=INK2)
ax.annotate(f"{t_last.op_pct_under_12m:.0f}%", (t_last.month, t_last.op_pct_under_12m), xytext=(-4, -16),
            textcoords="offset points", ha="right", fontsize=10, fontweight="bold", color=INK)
ax.set_ylim(50, 95); ax.set_ylabel("% of outpatients")
ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Specialties: total with over-12-month part
top = spec.head(10).iloc[::-1]
ax = fig.add_axes([0.15, 0.09, 0.29, 0.25]); tidy(ax, "Top 10 specialties: patients waiting, and how many over 12 months")
ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=0.8)
ax.barh(top.specialty, top.total_sep2026 / 1000, color="#d3d1cb", height=0.66, label="Under 12 months")
ax.barh(top.specialty, top.over_12m_sep2026 / 1000, color=BLUE, height=0.66, label="Over 12 months")
for y, r in enumerate(top.itertuples()):
    ax.text(r.total_sep2026 / 1000 + 1, y, f"{r.total_sep2026:,.0f}", va="center", fontsize=9, color=INK2)
ax.set_xlabel("Patients (thousands)"); ax.set_xlim(0, 100)
ax.legend(loc="lower right", frameon=False, fontsize=9, bbox_to_anchor=(1.0, 0.0))

# Hospitals: totals with year-on-year change
toph = hosp.head(8).iloc[::-1]
ax = fig.add_axes([0.68, 0.09, 0.25, 0.25]); tidy(ax, "Busiest hospitals, with change in a year")
ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=0.8)
ax.barh(toph.hospital.str.replace("University Hospital", "UH").str.replace("Hospitals", "Hosp."),
        toph.total_sep2026 / 1000, color=BLUE, height=0.66)
for y, r in enumerate(toph.itertuples()):
    ax.text(r.total_sep2026 / 1000 + 0.6, y, f"{r.total_sep2026:,.0f}  ({pct(r.total_sep2026, r.total_sep2025):+.0f}%)",
            va="center", fontsize=9, color=INK2)
ax.set_xlabel("Patients (thousands)"); ax.set_xlim(0, 75)

fig.text(0.04, 0.012, "Source: NTPF open data (CC-BY 4.0), April 2021 to September 2026. Average wait estimated from wait-band "
         "midpoints. Forecast: Holt-Winters, selected in a 13-origin backtest.", fontsize=9, color=INK2)
fig.savefig("images/00_dashboard.png", dpi=130, facecolor=SURFACE)
print("saved images/00_dashboard.png")
