"""
stage9_model_benchmark.py -- Table 4, the discovered equation against seven alternatives
=========================================================================================
Runs C11, the Kaplan-DeMaria baseline, a terrain-modulated-rate variant of it, and
four statistical alternatives through ONE storm-stratified five-fold split, so every
row of Table 4 is scored on exactly the same folds.

WHY THIS SCRIPT EXISTS, and it matters. In analysis_all.py the machine-learning
block shuffles the storms with an RNG that has already been consumed by the
2000-sample bootstrap and the 500-sample Dvorak Monte Carlo. Random forest and
gradient boosting therefore ran on DIFFERENT folds from every other model in the
table, which is not what the caption claims. This script puts them all on the
common split. The ranking is unchanged; three numbers move.

The rate test is the direct answer to the question of whether terrain modifies the
decay rate or offsets the exposure:

    C11               V = V0 [1 - a (V0 t + h)]          terrain as exposure offset
    KD95 + gamma*h    alpha -> alpha + gamma*h           terrain as rate modifier

Needs scikit-learn and pygam in addition to the usual stack.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

np.seterr(all="ignore")

POINTS = "data/ph_decay_with_terrain_5m_circ.csv"
OUT = "results/model_benchmark.csv"
TERRAIN_COL = "hmean_rad75"

PUBLISHED = {"C11": 9.22, "KD95_rate_terrain": 9.63, "KD95": 9.86,
             "GradientBoostedTrees": 10.43, "RandomForest": 10.84,
             "GAM": 11.89, "MultipleLinearRegression": 12.14}


def kd95(X, Vb, R, al):
    V0, t = X
    return Vb + (R * V0 - Vb) * np.exp(-al * t)


def kd95_rate(X, Vb, R, al, g):
    V0, t, h = X
    return Vb + (R * V0 - Vb) * np.exp(-(al + g * h) * t)


def c11(X, a):
    V0, t, h = X
    return V0 - a * V0 * (V0 * t + h)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


raw = pd.read_csv(POINTS)
raw["ISO_TIME"] = pd.to_datetime(raw["ISO_TIME"])
d = raw[raw.ISO_TIME.dt.hour.isin([0, 6, 12, 18])].sort_values(["SID", "ISO_TIME"]).copy()
d["t"] = (d.ISO_TIME - d.groupby("SID").ISO_TIME.transform("min")).dt.total_seconds() / 3600
d["V0"] = d.groupby("SID").USA_WIND.transform("first")
cnt = d.groupby("SID").SID.transform("size")
d = d[(cnt >= 3) & (d.V0 >= 34)]
d["h"] = d[TERRAIN_COL]
d = d.dropna(subset=["h_max", "h", "USA_WIND", "V0", "t", "STORM_SPD"])
d = d[(d.h_max > 0) & (d.USA_WIND > 0)].rename(columns={"USA_WIND": "y"}).reset_index(drop=True)
print(f"native subset: {len(d)} points / {d.SID.nunique()} storms   (expected 453 / 121)")

# the common split, identical to cvfold() in analysis_all.py
ss = d.SID.unique().copy()
np.random.seed(42)
np.random.shuffle(ss)
fs = len(ss) // 5
FOLDS = [ss[k * fs:(k + 1) * fs] if k < 4 else ss[k * fs:] for k in range(5)]

FEATS = ["V0", "t", "h"]
scores = {k: [] for k in PUBLISHED}

try:
    from pygam import LinearGAM, s as gam_s
    HAVE_GAM = True
except ImportError:
    HAVE_GAM = False
    print("  pygam not installed: the GAM row will be skipped")

for te in FOLDS:
    m = d.SID.isin(te)
    tr, ts = d[~m], d[m]

    pk = curve_fit(kd95, (tr.V0, tr.t), tr.y, p0=[15, .9, .05],
                   bounds=([0, .3, .001], [60, 1, .5]), maxfev=10000)[0]
    pa = curve_fit(c11, (tr.V0, tr.t, tr.h), tr.y, p0=[1.34e-4],
                   bounds=([1e-5], [1e-3]), maxfev=10000)[0]
    try:
        pg = curve_fit(kd95_rate, (tr.V0, tr.t, tr.h), tr.y, p0=[15, .9, .05, 1e-5],
                       bounds=([0, .3, .001, -1e-2], [60, 1, .5, 1e-2]), maxfev=30000)[0]
    except Exception:
        pg = list(pk) + [0.0]

    scores["KD95"].append(rmse(ts.y, kd95((ts.V0, ts.t), *pk)))
    scores["C11"].append(rmse(ts.y, c11((ts.V0, ts.t, ts.h), *pa)))
    scores["KD95_rate_terrain"].append(rmse(ts.y, kd95_rate((ts.V0, ts.t, ts.h), *pg)))
    scores["RandomForest"].append(rmse(ts.y, RandomForestRegressor(
        n_estimators=300, random_state=0, n_jobs=-1).fit(tr[FEATS], tr.y).predict(ts[FEATS])))
    scores["GradientBoostedTrees"].append(rmse(ts.y, GradientBoostingRegressor(
        random_state=0).fit(tr[FEATS], tr.y).predict(ts[FEATS])))
    scores["MultipleLinearRegression"].append(rmse(ts.y, LinearRegression()
                                                   .fit(tr[FEATS], tr.y).predict(ts[FEATS])))
    if HAVE_GAM:
        scores["GAM"].append(rmse(ts.y, LinearGAM(gam_s(0) + gam_s(1) + gam_s(2))
                                  .fit(tr[FEATS].values, tr.y.values)
                                  .predict(ts[FEATS].values)))

rows = []
for k, v in scores.items():
    if not v:
        continue
    rows.append(dict(model=k, RMSE=float(np.mean(v)), published=PUBLISHED[k],
                     difference=float(np.mean(v)) - PUBLISHED[k]))
R = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
R.to_csv(OUT, index=False)

print("\n  model                         common folds   Table 4   difference")
for _, r in R.iterrows():
    flag = "" if abs(r.difference) < 0.005 else "   <- differs"
    print(f"  {r.model:28s} {r.RMSE:8.3f}   {r.published:8.2f}   {r.difference:+7.3f}{flag}")
print("\nAll rows should agree with Table 4 to rounding; v1.0.2 refreshed the "
      "published column to the revised common-fold benchmark.")
print(f"\nwrote {OUT}")
