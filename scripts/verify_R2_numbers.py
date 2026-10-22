#!/usr/bin/env python3
"""
verify_R2_numbers.py - recompute, from the archived data alone, the data-derived
numbers that revision 2 of WAF-D-26-0060 adds or restates
(including the October 2026 updates: the Taylor-expansion sentence in Sec. 4a and the
Fig. 2 and Fig. 3 captions).

Usage:   python verify_R2_numbers.py  <path to the AttenuationPaper repository>
Needs:   numpy and pandas (tested with pandas 2.2.3 and 3.0.5; no model fitting beyond
         one-parameter least squares)
Reads:   data/oos_predictions.csv, data/ph_decay_with_terrain_archive.csv,
         results/pysr_native_5m/all_runs_summary.csv and run_*/pareto_front.csv
Output:  one line per claim: where it appears, stated value, recomputed value,
         PASS or CHECK.  A CHECK line means the text and the archive disagree.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "."
oos = pd.read_csv(os.path.join(root, "data", "oos_predictions.csv"))
arc = pd.read_csv(os.path.join(root, "data", "ph_decay_with_terrain_archive.csv"), low_memory=False)
pdir = os.path.join(root, "results", "pysr_native_5m")
summ = pd.read_csv(os.path.join(pdir, "all_runs_summary.csv"))

names = arc.drop_duplicates("SID").set_index("SID")[["NAME", "SEASON"]]
oos = oos.join(names, on="SID")
oos["eK"] = oos.KD95_pred - oos.obs          # bias convention: predicted minus observed
oos["eC"] = oos.C11_pred - oos.obs


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


ROWS = []


def check(where, claim, stated, value, nd=None, tol=None):
    """nd: compare after rounding to nd decimals; tol: absolute tolerance; neither: exact."""
    if isinstance(stated, str):
        ok = stated == value
        shown = value
    else:
        if nd is not None:
            ok = abs(round(value, nd) - stated) < 1e-9
        elif tol is not None:
            ok = abs(value - stated) <= tol
        else:
            ok = value == stated
        shown = f"{value:.{max(nd or 0, 2)}f}" if isinstance(value, float) else str(value)
    ROWS.append((where, claim, str(stated), shown, "PASS" if ok else "CHECK"))


def storm(name, season):
    return oos[(oos.NAME == name) & (oos.SEASON == season)]


# ---------------------------------------------------------------- dataset identity
check("Sec. 2a", "native points", 453, len(oos))
check("Sec. 2a", "native storms", 121, oos.SID.nunique())

# ---------------------------------------------------------------- Table 2 (per-fold CV)
g = oos.groupby("fold")
fold = pd.DataFrame({
    "N": g.size(), "h": g.h_used.mean(), "t": g.t_used.mean(),
    "rK": g.eK.apply(rmse), "rC": g.eC.apply(rmse), "bK": g.eK.mean(), "bC": g.eC.mean()})
T2 = {1: (105, 199, 11.8, 14.34, 13.18, -0.22, -0.75),
      2: (87, 183, 9.7, 8.47, 7.46, -0.59, -0.35),
      3: (88, 171, 9.8, 9.82, 9.27, -0.68, -1.33),
      4: (87, 196, 8.8, 9.44, 8.94, -0.11, -0.92),
      5: (86, 237, 10.3, 7.24, 7.28, -0.06, -1.93)}
for k, (N, h, t, rK, rC, bK, bC) in T2.items():
    f = fold.loc[k]
    check("Table 2", f"fold {k} N", N, int(f.N))
    check("Table 2", f"fold {k} mean terrain (m)", h, float(f.h), nd=0)
    check("Table 2", f"fold {k} mean elapsed time (hr)", t, float(f.t), nd=1)
    check("Table 2", f"fold {k} RMSE KD95", rK, float(f.rK), nd=2)
    check("Table 2", f"fold {k} RMSE C11", rC, float(f.rC), nd=2)
    check("Table 2", f"fold {k} bias KD95", bK, float(f.bK), nd=2)
    check("Table 2", f"fold {k} bias C11", bC, float(f.bC), nd=2)
check("Table 2", "Mean row: mean terrain (m), mean of folds", 197, float(fold.h.mean()), nd=0)
check("Table 2", "Mean row: mean elapsed time (hr), mean of folds", 10.1, float(fold.t.mean()), nd=1)
check("Table 2 / Sec. 3b", "fold-mean RMSE KD95", 9.86, float(fold.rK.mean()), nd=2)
check("Table 2 / Sec. 3b", "fold-mean RMSE C11", 9.22, float(fold.rC.mean()), nd=2)
check("Table 2", "fold-mean bias KD95", -0.33, float(fold.bK.mean()), nd=2)
check("Table 2", "fold-mean bias C11", -1.06, float(fold.bC.mean()), nd=2)
check("Table 2", "Std row KD95 (sample std, ddof=1)", 2.69, float(fold.rK.std(ddof=1)), nd=2)
check("Table 2", "Std row C11 (sample std, ddof=1)", 2.38, float(fold.rC.std(ddof=1)), nd=2)

# ---------------------------------------------------------------- Sec. 3b text
check("Sec. 3b", "pooled RMSE KD95 (all 453 held-out points)", 10.36, rmse(oos.eK), nd=2)
check("Sec. 3b", "pooled RMSE C11 (all 453 held-out points)", 9.64, rmse(oos.eC), nd=2)
check("Sec. 3b", "fold-mean reduction (%)", 6.5, 100 * (fold.rK.mean() - fold.rC.mean()) / fold.rK.mean(), nd=1)
check("Sec. 3b", "folds in which C11 beats KD95", 4, int((fold.rC < fold.rK).sum()))

others = fold.loc[[2, 3, 4, 5]]
lo = min(others.rK.min(), others.rC.min())
hi = max(others.rK.max(), others.rC.max())
check("Sec. 3b", "other folds, lowest RMSE (either model)", 7.2, float(lo), nd=1)
check("Sec. 3b", "other folds, highest RMSE (either model)", 9.8, float(hi), nd=1)

three = {(n, s): storm(n, s).SID.unique() for n, s in [("GONI", 2020), ("SIBYL", 1995), ("BETTY", 1980)]}
sids = np.concatenate(list(three.values()))
f1 = oos[oos.fold == 1]
m3 = f1.SID.isin(sids)
check("Sec. 3b", "Goni, Sibyl, Betty all in fold 1", "yes",
      "yes" if all(oos[oos.SID.isin(s)].fold.eq(1).all() for s in three.values()) else "no")
check("Sec. 3b", "three storms' share of fold-1 squared error, C11 (%)", 47,
      100 * float((f1.eC[m3] ** 2).sum() / (f1.eC ** 2).sum()), nd=0)
check("Sec. 3b", "three storms' share of fold-1 squared error, KD95 (%)", 55,
      100 * float((f1.eK[m3] ** 2).sum() / (f1.eK ** 2).sum()), nd=0)
check("Sec. 3b", "fold 1 without the three storms, RMSE C11", 10.16, rmse(f1.eC[~m3]), nd=2)
check("Sec. 3b", "fold 1 without the three storms, RMSE KD95", 10.27, rmse(f1.eK[~m3]), nd=2)
check("Sec. 3b", "fold 1 still hardest without them (C11 and KD95 above every other fold)", "yes",
      "yes" if rmse(f1.eC[~m3]) > others.rC.max() and rmse(f1.eK[~m3]) > others.rK.max() else "no")
check("Sec. 3b", "fold with the longest mean elapsed time", 1, int(fold.t.idxmax()))
check("Sec. 3b", "other folds, shortest mean elapsed time (hr)", 8.8, float(others.t.min()), nd=1)
check("Sec. 3b", "other folds, longest mean elapsed time (hr)", 10.3, float(others.t.max()), nd=1)
check("Sec. 3b", "fold with C11's largest gain", 1, int((fold.rK - fold.rC).idxmax()))

per_storm = oos.groupby("SID").agg(V0=("V0_used", "first"), n=("obs", "size"), tmin=("t_used", "min"))
gon = storm("GONI", 2020).sort_values("t_used")
check("Sec. 3b", "Goni has the highest landfall intensity V0 of the 121 storms", "yes",
      "yes" if gon.V0_used.iloc[0] == per_storm.V0.max() and (per_storm.V0 == per_storm.V0.max()).sum() == 1 else "no")
pairs = list(zip(gon.obs.values[:-1], gon.obs.values[1:], np.diff(gon.t_used.values)))
check("Sec. 3b", "Goni 130 -> 75 kt within one 6-hr interval", "yes",
      "yes" if (130, 75, 6.0) in [(a, b, d) for a, b, d in pairs] else "no")
sib = storm("SIBYL", 1995).merge(arc[["SID", "ISO_TIME", "DIST2LAND"]], on=["SID", "ISO_TIME"]).sort_values("t_used")
check("Sec. 3b", "Sibyl first and peak native winds (kt)", "65 -> 90", f"{sib.obs.iloc[0]} -> {sib.obs.max()}")
check("Sec. 3b", "Sibyl distance to land at its 90-kt points (km)", "24, 46",
      ", ".join(str(int(d)) for d in sib[sib.obs == 90].DIST2LAND))
bet = storm("BETTY", 1980).sort_values("t_used")
drops = -np.diff(bet.obs.values)
check("Sec. 3b", "Betty largest 6-hr loss (kt)", 40, int(drops.max()))
i = int(drops.argmax())
check("Sec. 3b", "Betty mean terrain at the ends of that interval (m)", "474, 674",
      f"{bet.h_used.iloc[i]:.0f}, {bet.h_used.iloc[i + 1]:.0f}")

# ---------------------------------------------------------------- Sec. 3b case studies (out-of-fold)
ps = (oos.assign(K=oos.eK ** 2, C=oos.eC ** 2).groupby("SID")[["K", "C"]].mean()) ** 0.5   # per-storm RMSE
check("Sec. 3b", "storms with the lower out-of-fold RMSE for C11 (of 121)", 75, int((ps.C < ps.K).sum()))
for n, s, better in [("BETTY", 1980, True), ("GONI", 2020, True), ("KOPPU", 2015, False), ("FLO", 1993, False)]:
    sid = storm(n, s).SID.iloc[0]
    check("Sec. 3b", f"{n.title()} ({s}): C11 {'better' if better else 'worse'} than KD95 out of fold",
          "yes", "yes" if (ps.loc[sid, "C"] < ps.loc[sid, "K"]) == better else "no")

# ---------------------------------------------------------------- Table 3 and the Sec. 3d quadrature argument
T3 = {"TS (34-63 kt)": ((oos.V0_used >= 34) & (oos.V0_used <= 63), 190, 6.69, 6.16, 0.62, -1.09, 0.53),
      "Cat 1-2 (64-95 kt)": ((oos.V0_used >= 64) & (oos.V0_used <= 95), 128, 11.89, 11.31, -2.79, -2.11, 0.57),
      "Cat 3+ (>=96 kt)": (oos.V0_used >= 96, 135, 12.76, 11.71, 0.67, 0.04, 1.05),
      "Low (h < 100 m)": (oos.h_used < 100, 153, 11.50, 10.84, -1.75, -0.86, 0.66),
      "Medium (100-300 m)": ((oos.h_used >= 100) & (oos.h_used < 300), 200, 9.30, 8.33, -0.88, -1.79, 0.97),
      "High (h >= 300 m)": (oos.h_used >= 300, 100, 10.52, 10.15, 2.95, 0.17, 0.37)}
for lab, (m, N, rK, rC, bK, bC, d) in T3.items():
    s = oos[m]
    check("Table 3", f"{lab} N", N, len(s))
    check("Table 3", f"{lab} RMSE KD95", rK, rmse(s.eK), nd=2)
    check("Table 3", f"{lab} RMSE C11", rC, rmse(s.eC), nd=2)
    check("Table 3", f"{lab} bias KD95", bK, float(s.eK.mean()), nd=2)
    check("Table 3", f"{lab} bias C11", bC, float(s.eC.mean()), nd=2)
    check("Table 3", f"{lab} Delta (KD95 minus C11)", d, rmse(s.eK) - rmse(s.eC), nd=2)
hi_t = oos[oos.h_used >= 300]
scatter = float(np.sqrt(rmse(hi_t.eK) ** 2 - hi_t.eK.mean() ** 2))
check("Sec. 3d", "KD95 residual scatter over high terrain, 'about 10 kt'", 10, scatter, tol=0.5)
check("Sec. 3d", "RMSE drop from removing the 2.95-kt bias, 'about 0.4 kt'", 0.4, rmse(hi_t.eK) - scatter, tol=0.05)

# ---------------------------------------------------------------- Fig. S1 binned residuals (bins as in fig_s1_one_to_one.py)
edges = [0, 50, 100, 150, 200, 300, 450, 900]
check("Fig. S1", "all points inside the bin range 0-900 m", "yes",
      "yes" if (oos.h_used >= 0).all() and (oos.h_used < 900).all() else "no")
bK = [oos.eK[(oos.h_used >= a) & (oos.h_used < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
bC = [oos.eC[(oos.h_used >= a) & (oos.h_used < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
check("Fig. S1", "KD95 binned residual, 50-100 m", -2.2, float(bK[1]), nd=1)
check("Fig. S1", "KD95 binned residual, 100-150 m", -2.8, float(bK[2]), nd=1)
check("Fig. S1", "KD95 binned residual, 450-900 m", 5.8, float(bK[6]), nd=1)
check("Fig. S1", "C11 binned residuals, lowest", -2.7, float(min(bC)), nd=1)
check("Fig. S1", "C11 binned residuals, highest", 1.7, float(max(bC)), nd=1)

# ---------------------------------------------------------------- Fig. S2 caption
check("Fig. S2", "fold 1 improvement, KD95 minus C11 (kt)", 1.17, float(fold.rK[1] - fold.rC[1]), nd=2)
check("Fig. S2", "fold 5 margin, C11 minus KD95 (kt)", 0.03, float(fold.rC[5] - fold.rK[5]), nd=2)

# ---------------------------------------------------------------- Sec. S4 sampling
check("Sec. S4", "storms retaining only one or two points", 13, int((per_storm.n <= 2).sum()))
check("Sec. S4", "storms whose retained series begins at t = 6 hr", 8, int((per_storm.tmin == 6).sum()))
check("Sec. S4", "storms beginning at t = 0 or t = 6 only", "yes",
      "yes" if per_storm.tmin.isin([0, 6]).all() else "no")

# ---------------------------------------------------------------- Table 1 / Sec. S1 (PySR fronts, in-sample)
V0, t, h, y = oos.V0_used.values.astype(float), oos.t_used.values, oos.h_used.values, oos.obs.values
check("Table 1", "loss of V(t) = V0 (confirms same 453 points as PySR)", 452.9, float(np.mean((y - V0) ** 2)), nd=1)
for lab, x, stated_loss in [("C9 unit intercept, V0 - a V0^2 t", V0 ** 2 * t, 97.9),
                            ("C11 unit intercept, V0 - a V0 (V0 t + h)", V0 * (V0 * t + h), 90.3)]:
    a = float(np.sum(x * (V0 - y)) / np.sum(x * x))            # exact one-parameter least squares
    check("Table 1 / S1", f"{lab}: in-sample loss", stated_loss, float(np.mean((y - (V0 - a * x)) ** 2)), nd=1)
    if lab.startswith("C11"):
        check("Sec. 3a", "C11 least-squares coefficient a (x 1e-4)", 1.428853, a * 1e4, nd=6)

TERR = re.compile(r"h_mean|h_max|h_std")
runs = []
for p in sorted(glob.glob(os.path.join(pdir, "run_*", "pareto_front.csv"))):
    seed = int(re.search(r"seed(\d+)", p).group(1))
    fr = pd.read_csv(p).sort_values("complexity")
    terr = fr[fr.equation.str.contains(TERR)]
    first = int(terr.complexity.min())
    prev = fr[fr.complexity < first].loss.min()
    loss_at = dict(zip(fr.complexity, fr.loss))
    eq_at = dict(zip(fr.complexity, fr.equation))
    runs.append(dict(seed=seed, first=first, gain=100 * (prev - terr.loss.iloc[0]) / prev,
                     c9=loss_at.get(9, np.nan), c11=loss_at.get(11, np.nan),
                     c11_terr=bool(TERR.search(str(eq_at.get(11, ""))))))
R = pd.DataFrame(runs).set_index("seed")
check("Sec. 3a / S1", "runs with terrain at complexity 11", 9, int((R["first"] == 11).sum()))
check("Sec. 3a / S1", "runs with terrain first at complexity 13 (seeds)", "42, 47, 48",
      ", ".join(str(s) for s in R.index[R["first"] == 13]))
check("Sec. S1", "runs whose complexity-9 loss is 97.1 (fitted leading factor)", 7, int((R.c9.round(1) == 97.1).sum()))
check("Sec. S1", "lowest complexity-9 loss across runs", 96.2, float(R.c9.min()), nd=1)
check("Sec. S1", "runs with the unit-intercept C11 at loss 90.3 (seeds)", "49, 52",
      ", ".join(str(s) for s in R.index[R.c11_terr & (R.c11.round(1) == 90.3)]))
check("Sec. S1", "runs with complexity-11 terrain form at loss 89.3", 7, int((R.c11_terr & (R.c11.round(1) == 89.3)).sum()))
check("Sec. 3a / S1", "mean loss reduction at terrain entry, all 12 runs (%)", 6.0, float(R.gain.mean()), nd=1)
eleven = R.gain.drop(48)
check("Sec. 3a / S1", "range of that reduction in the eleven runs other than seed 48 (%)", "4.6-8.0",
      f"{eleven.min():.1f}-{eleven.max():.1f}")
check("Sec. 3a / S1", "reduction in seed 48 (%)", 0.8, float(R.gain[48]), nd=1)
check("Sec. S1", "seed 48 enters terrain as sqrt(h_mean) at complexity 13", "yes",
      "yes" if R.loc[48, "first"] == 13 else "no")

auto_c, auto_terrain = [], []
for _, s in summ.iterrows():
    fr = pd.read_csv(glob.glob(os.path.join(pdir, s.run_id, "pareto_front.csv"))[0])
    j = (fr.RMSE - s.RMSE).abs().idxmin()
    auto_c.append(int(fr.complexity[j]))
    auto_terrain.append(bool(TERR.search(s.equation)))
check("Sec. 2d / S1", "automatic selections that contain terrain", 0, int(sum(auto_terrain)))
check("Sec. 2d / S1", "complexity range of the automatic selections", "8-10", f"{min(auto_c)}-{max(auto_c)}")

# ---------------------------------------------------------------- updated R2 (October 2026): Sec. 4a, Fig. 2 and Fig. 3 captions
xc = V0 * (V0 * t + h)
a_ls = float(np.sum(xc * (V0 - y)) / np.sum(xc * xc))       # C11 least-squares coefficient, as above
xx = a_ls * (V0 * t + h)                                     # the argument of the Taylor expansion
check("Sec. 4a", "x = a(V0 t + h) over the 453 points: median", 0.11, float(np.median(xx)), nd=2)
check("Sec. 4a", "share of the 453 points with x below 0.3 (%)", 91, 100 * float((xx < 0.3).mean()), nd=0)
c9_eq = {}
for p in sorted(glob.glob(os.path.join(pdir, "run_*", "pareto_front.csv"))):
    fr = pd.read_csv(p)
    row = fr[fr.complexity == 9]
    if len(row):
        c9_eq[float(row.loss.iloc[0])] = str(row.equation.iloc[0])
best = c9_eq[min(c9_eq)]
check("Fig. 2 caption", "complexity-9 minimum is a rational expression, not Eq. (2)", "yes",
      "yes" if "/" in best and "exp" not in best else "no")
g = storm("GONI", 2020)
g_anchor = arc[(arc.NAME == "GONI") & (arc.SEASON == 2020) & (arc.t_used == 0)]
check("Fig. 3 caption", "Goni's retained series begins at t (hr)", 6, int(g.t_used.min()))
check("Fig. 3 caption", "Goni's t = 0 anchor record has no terrain data and is not retained", "yes",
      "yes" if len(g_anchor) == 1 and g_anchor.h_max.isna().all()
      and g_anchor.in_analysis.astype(str).str.lower().eq("false").all() else "no")

# ---------------------------------------------------------------- report
w = max(len(r[1]) for r in ROWS)
print(f"{'where':<16} {'claim':<{w}}  {'stated':>12}  {'recomputed':>12}  status")
for r in ROWS:
    print(f"{r[0]:<16} {r[1]:<{w}}  {r[2]:>12}  {r[3]:>12}  {r[4]}")
n_bad = sum(r[4] != "PASS" for r in ROWS)
print(f"\n{len(ROWS)} checks, {len(ROWS) - n_bad} PASS, {n_bad} CHECK")
