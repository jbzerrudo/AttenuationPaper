"""
figure2_pareto_native.py -- Figure 2, PySR Pareto front on the native 6-hourly data
====================================================================================
Pools the twelve independent PySR runs and plots, at each expression complexity,
the lowest mean squared error any run attained there.

Two points about the x axis, because they look like plotting bugs and are not:

  Complexities 2 and 4 carry no marker because no run produced an expression at
  those sizes that improved on a simpler one.  A Pareto front only retains an
  expression if it beats every cheaper expression, so a complexity with no
  improvement has no entry.  Nothing at complexity 2 beat V0 (452.87) and nothing
  at complexity 4 beat V0 x 0.83 (253.44).  The operator set does allow them
  (unary exp, log, sqrt, abs are available), so they are reachable, just never
  worth keeping.

  Complexity 26 and beyond are absent because the search ran with maxsize = 25.

The axis is therefore ticked at every integer from 1 to 25, so a reader can see
which complexities exist and which do not, rather than inferring it from a
coarse 0-2-4 grid.

Input : results/pysr_native_5m/run_*/pareto_front.csv
Output: figures/fig02_pareto_front.png
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "pysr_native_5m")
OUTPATH = os.path.join(os.path.dirname(__file__), "..", "figures", "fig02_pareto_front.png")

TERRAIN_TOKENS = ("h_mean", "hmean", "h_max")

# Marker colours, as requested: Harley-Davidson orange for terrain-free
# expressions, Honda Candy Ruby Red for expressions containing terrain.
# These are close visual matches to the paint colours, not official brand values.
ORANGE = "#F47216"   # Harley orange
RUBY   = "#A0132E"   # Candy Ruby Red

# Sized for width=0.9\textwidth in a 39 pc AMS text block, i.e. 5.85 in wide.
# Fonts are set in points so what is specified is what prints.
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 400,
})


def load_front():
    frames = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "run_*", "pareto_front.csv"))):
        df = pd.read_csv(path)
        df["seed"] = re.search(r"seed(\d+)", path).group(1)
        frames.append(df)
    if not frames:
        raise SystemExit(f"no pareto_front.csv found under {RESULTS}")
    allruns = pd.concat(frames, ignore_index=True)
    best = allruns.loc[allruns.groupby("complexity").loss.idxmin()].sort_values("complexity")
    best["has_terrain"] = best.equation.astype(str).apply(
        lambda e: any(tok in e for tok in TERRAIN_TOKENS))
    return best.reset_index(drop=True)


def main():
    best = load_front()
    fig, ax = plt.subplots(figsize=(5.85, 3.9))

    ax.plot(best.complexity, best.loss, "-", color="#9a9a9a", lw=0.9, zorder=1)

    plain = best[~best.has_terrain]
    terr = best[best.has_terrain]
    ax.plot(plain.complexity, plain.loss, "o", mfc="white", mec=ORANGE, mew=1.6,
            ms=6, zorder=3, label="no terrain variable")
    ax.plot(terr.complexity, terr.loss, "s", color=RUBY, ms=5.5, zorder=3,
            label=r"contains $\bar{h}$")

    knee = int(terr.complexity.min())
    ax.axvline(knee, color=RUBY, ls=":", lw=1.1, zorder=2)
    ax.text(knee - 0.35, best.loss.max() * 0.62, "knee: terrain enters",
            rotation=90, ha="right", va="center", color=RUBY, fontsize=7)

    for comp, off in [(7, (0, 11)), (9, (-2, 11)), (11, (10, 11)), (15, (0, 11))]:
        row = best[best.complexity == comp]
        if row.empty:
            continue
        y = float(row.loss.iloc[0])
        ax.annotate(f"C{comp}", (comp, y), xytext=off, textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color=RUBY if bool(row.has_terrain.iloc[0]) else "#4a4a4a")

    ax.set_yscale("log")
    ax.set_xlabel("Expression complexity")
    ax.set_ylabel(r"Mean squared error (kt$^2$)")
    ax.set_xlim(0, 26)
    ax.set_xticks(range(1, 26))
    ax.set_xticklabels([str(c) for c in range(1, 26)])
    # Mark the two complexities the search never populated.
    for absent in (2, 4):
        ax.get_xticklabels()[absent - 1].set_color("#b0b0b0")
    ax.grid(True, which="major", axis="both", color="#e4e4e4", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPATH), exist_ok=True)
    fig.savefig(OUTPATH, bbox_inches="tight")
    print(f"wrote {OUTPATH}")
    print(f"complexities plotted: {sorted(best.complexity.tolist())}")
    print(f"absent from every run: {[c for c in range(1, 26) if c not in set(best.complexity)]}")
    print(f"terrain first enters at complexity {knee}")


if __name__ == "__main__":
    main()
