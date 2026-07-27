"""
figure5_stratified_native.py -- Figure 5, stratified out-of-sample performance
===============================================================================
Rebuilt for legibility. The previous version placed three wide panels inside one
text width, which left each panel under two inches and drove the tick labels to
about 4 pt in print. Reviewer 2 asked for larger labels.

Two changes fix it:

  Layout. Panels (a) and (b) share the top row and panel (c) spans the full width
  beneath. Panel (c) carries the headline result, so it gets the room.

  Sizing. The figure is built at its printed size, 5.85 in for
  width=0.9\\textwidth in a 39 pc AMS text block, with every font set in points.
  What is specified is what prints.

Colours match Figure 2: Harley orange for the terrain-free model (KD95), Candy
Ruby Red for the terrain equation (C11). Close visual matches to the paint
colours, not official brand values.

Input : results/stratified_bias_native6h.csv
Output: figures/fig05_stratified.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "results", "stratified_bias_native6h.csv")
OUTPATH = os.path.join(HERE, "..", "figures", "fig05_stratified.png")

ORANGE = "#F47216"   # Harley orange, KD95
RUBY = "#A0132E"     # Candy Ruby Red, C11
MINUS = "−"     # typographic minus, used everywhere for consistency

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "axes.unicode_minus": True,
    "figure.dpi": 400,
})

INTENSITY = ["TS_34_63", "Cat1_2_64_95", "Cat3_ge96"]
TERRAIN = ["low_h<100", "med_100_300", "high_h>=300"]
INT_TICKS = ["TS\n34" + MINUS + "63", "Cat 1" + MINUS + "2\n64" + MINUS + "95",
             "Cat 3+\n$\\geq$96"]
TER_TICKS = ["Low\n$<$100 m", "Med\n100" + MINUS + "300 m", "High\n$\\geq$300 m"]
YMAX_RMSE = 14.6          # shared by panels (a) and (b) so they compare directly

W = 0.36          # bar width
GAP = 0.02        # surface gap between the paired bars


def signed(v, dp=2):
    """Format with a typographic minus and an explicit plus."""
    s = f"{abs(v):.{dp}f}"
    return (MINUS if v < 0 else "+") + s


def rmse_panel(ax, d, keys, ticks, ylabel=None, title=""):
    x = np.arange(len(keys))
    sub = d.loc[keys]
    ax.bar(x - W / 2 - GAP / 2, sub.KD95_RMSE, W, color=ORANGE, label="KD95", zorder=3)
    ax.bar(x + W / 2 + GAP / 2, sub.C11_RMSE, W, color=RUBY, label="C11", zorder=3)

    top = float(max(sub.KD95_RMSE.max(), sub.C11_RMSE.max()))
    for xi, (_, r) in zip(x, sub.iterrows()):
        ax.text(xi, max(r.KD95_RMSE, r.C11_RMSE) + YMAX_RMSE * 0.03,
                MINUS + f"{r.dRMSE:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=RUBY)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\nN = {int(n)}" for t, n in zip(ticks, sub.N)])
    ax.set_ylim(0, YMAX_RMSE)
    ax.set_title(title, loc="left", pad=6)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="#e4e4e4", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def bias_panel(ax, d):
    keys = TERRAIN
    x = np.arange(len(keys))
    sub = d.loc[keys]
    ax.bar(x - W / 2 - GAP / 2, sub.KD95_bias, W, color=ORANGE, label="KD95", zorder=3)
    ax.bar(x + W / 2 + GAP / 2, sub.C11_bias, W, color=RUBY, label="C11", zorder=3)
    ax.axhline(0, color="#444444", lw=0.9, zorder=4)

    for xi, (_, r) in zip(x, sub.iterrows()):
        for val, dx in ((r.KD95_bias, -W / 2 - GAP / 2), (r.C11_bias, W / 2 + GAP / 2)):
            off = 0.16 if val >= 0 else -0.16
            ax.text(xi + dx, val + off, signed(val), ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8, fontweight="bold", color="#222222")

    ax.annotate("KD95 keeps storms too strong\nover the mountains",
                xy=(2 - W / 2 - GAP / 2, 3.05), xytext=(0.98, 3.70),
                fontsize=8, color="#222222", ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(TER_TICKS)
    ax.set_ylim(-3.0, 4.7)
    ax.set_ylabel("Bias (kt), predicted " + MINUS + " observed")
    ax.set_title("(c) Bias by mean terrain elevation", loc="left", pad=6)
    ax.grid(True, axis="y", color="#e4e4e4", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    d = pd.read_csv(SRC).set_index("stratum")

    fig = plt.figure(figsize=(5.85, 5.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12],
                          hspace=0.52, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    rmse_panel(ax_a, d, INTENSITY, INT_TICKS,
               ylabel="Out-of-sample RMSE (kt)",
               title="(a) RMSE by landfall intensity")
    rmse_panel(ax_b, d, TERRAIN, TER_TICKS,
               title="(b) RMSE by mean terrain elevation")
    bias_panel(ax_c, d)

    ax_a.legend(loc="upper left", frameon=False, ncol=1,
                handlelength=1.1, labelspacing=0.35, borderpad=0.1)

    os.makedirs(os.path.dirname(OUTPATH), exist_ok=True)
    fig.savefig(OUTPATH, bbox_inches="tight")
    print(f"wrote {OUTPATH}")
    for k in INTENSITY + TERRAIN:
        r = d.loc[k]
        print(f"  {k:16s} N={int(r.N):4d}  KD95 {r.KD95_RMSE:6.2f}  C11 {r.C11_RMSE:6.2f}"
              f"  dRMSE {r.dRMSE:5.2f}  bias {r.KD95_bias:+6.2f} -> {r.C11_bias:+6.2f}")


if __name__ == "__main__":
    main()
