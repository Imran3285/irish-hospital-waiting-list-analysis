"""Shared look for every chart in this project (README charts and dashboard)."""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BLUE = "#2a78d6"          # the one accent: the series the title is about
FORECAST = "#6250d6"      # forecasts: violet, so they stand apart from actual data
FORECAST_80 = "#c3bdf0"   # 80% prediction range
FORECAST_95 = "#e4e1f9"   # 95% prediction range
GREY = "#8d8b85"          # comparison series
GREY_LIGHT = "#c9c7c1"    # background bars / minor series
INK, INK2 = "#0b0b0b", "#52514e"
GRID, SURFACE = "#e6e5e1", "#fcfcfb"
TARGET = "#52514e"        # targets and reference lines (dashed)

SOURCE = "Source: NTPF open data (CC-BY 4.0), April 2021 to September 2026."

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False,
})


def new_figure(title, subtitle, ncols=1, height=5.0, width=12.0):
    """Figure with a left-aligned headline (the finding) and a grey subtitle."""
    fig, axes = plt.subplots(1, ncols, figsize=(width, height))
    # margins fixed in inches, so every chart has the same header and footer
    fig.subplots_adjust(left=0.07, right=0.97, top=1 - 1.0 / height, bottom=0.75 / height, wspace=0.22)
    fig.text(0.07, 1 - 0.32 / height, title, fontsize=15, fontweight="bold", color=INK)
    fig.text(0.07, 1 - 0.6 / height, subtitle, fontsize=11, color=INK2)
    return fig, axes


def tidy(ax, panel_title=None, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if panel_title:
        ax.set_title(panel_title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8)


def year_axis(ax, step=1):
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def save(fig, path, note=""):
    fig.text(0.07, 0.12 / fig.get_figheight(), (SOURCE + " " + note).strip(), fontsize=9, color=INK2)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)
