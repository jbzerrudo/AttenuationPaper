"""
stage5_bias_by_radius.py -- terrain-conditional bias as a function of footprint radius
=======================================================================================
Produces the Bias_high column of Table 4(b) and the numbers quoted in Section 3e.

Run it after analysis_all.py. Nothing to edit: the paths below are already set.
Needs only pandas, numpy and scipy, so the same environment that runs
analysis_all.py will run this. Takes a few seconds.

WHAT IT DOES
------------
Rebuilds the native 6-hourly subset exactly as analysis_all.py does (453 points,
121 storms), then for each of the fourteen sampling radii runs the identical
storm-stratified five-fold cross-validation and reports:

  dRMSE   the C11 advantage over KD95, as the mean of the five fold RMSEs
          (this is the Delta column of Table 4b, and reproduces radius_sweep.csv)

  Bias_high_fixed   the pooled out-of-sample C11 bias over high terrain, with the
          strata held FIXED at the 75-km definition so the same 100 points enter
          the high-terrain bin at every radius. THIS IS THE TABLE 4(b) COLUMN.

  Bias_high_self    the same quantity with the strata defined by the radius being
          tested. Reported only so the difference is visible: it is NOT what the
          table shows, and it is non-monotone because the bin membership moves
          with the radius. Do not quote it.

VERIFY IT MATCHES
-----------------
The script first prints a self-check. It must read:

    self-check: KD95 9.862  C11 9.223   (published 9.86 / 9.22)

If those two numbers do not appear, stop: the input file or the environment
differs from the one that produced the manuscript, and nothing below is valid.

Then confirm these eight rows against Table 4(b):

    r_km    dRMSE   Bias_high_fixed
      25    0.41    +1.53
      50    0.50    +0.43
      70    0.62    +0.15
      75    0.64    +0.17
      90    0.67    +0.34
     110    0.69    +0.66
     150    0.67    +0.99
     200    0.58    +0.98

Output: bias_by_radius.csv in the same folder as the other result CSVs.
"""
import os, re
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
np.seterr(all='ignore')

# ---- PATHS (Jef's laptop) --------------------------------------------------
POINTS = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\CSV\ph_decay_with_terrain_5m_circ.csv"
OUTDIR = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\CSV"
OUT    = os.path.join(OUTDIR, "bias_by_radius.csv")

# The radius whose terrain bins define the fixed strata. Must match the radius
# adopted in the manuscript, so that the 75-km row of the table is the same
# number that appears in stratified_bias_native6h.csv.
STRAT_RADIUS_COL = 'hmean_rad75'

# ---- models, identical to analysis_all.py ----------------------------------
def kd95(X, Vb, R, al):
    V0, t = X
    return Vb + (R * V0 - Vb) * np.exp(-al * t)

def c11(X, a):
    V0, t, h = X
    return V0 - a * V0 * (V0 * t + h)

def fkd(s):
    try:
        return curve_fit(kd95, (s.V0, s.t), s.y, p0=[15, .9, .05],
                         bounds=([0, .3, .001], [60, 1, .5]), maxfev=10000)[0]
    except Exception:
        return np.array([38.95, 1, .0393])

def f11(s, h):
    try:
        return curve_fit(c11, (s.V0, s.t, s[h]), s.y, p0=[1.34e-4],
                         bounds=([1e-5], [1e-3]), maxfev=10000)[0]
    except Exception:
        return np.array([1.34e-4])

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

# ---- native 6-hourly subset, identical to build(raw, native=True) -----------
raw = pd.read_csv(POINTS)
raw['ISO_TIME'] = pd.to_datetime(raw['ISO_TIME'])
d = raw[raw.ISO_TIME.dt.hour.isin([0, 6, 12, 18])].sort_values(['SID', 'ISO_TIME']).copy()
# t and V0 are anchored HERE, on the hour-filtered set, before the validity drops
d['t']  = (d.ISO_TIME - d.groupby('SID').ISO_TIME.transform('min')).dt.total_seconds() / 3600
d['V0'] = d.groupby('SID').USA_WIND.transform('first')
c = d.groupby('SID').SID.transform('size')
d = d[(c >= 3) & (d.V0 >= 34)]
d = d.dropna(subset=['h_max', 'USA_WIND', 'V0', 't', 'STORM_SPD'])
d = d[(d.h_max > 0) & (d.USA_WIND > 0)].rename(columns={'USA_WIND': 'y'}).reset_index(drop=True)
print(f"native subset: {len(d)} points / {d.SID.nunique()} storms   (expected 453 / 121)")

RAD = sorted((int(m.group(1)), col) for col in d.columns
             for m in [re.fullmatch(r'hmean_rad(\d+)', str(col))]
             if m and d[col].notna().sum() > 0)
print(f"radii found: {[r for r, _ in RAD]}\n")

# ---- one cross-validated pass at a given radius ----------------------------
def run(km, hcol):
    df = d.dropna(subset=[hcol, STRAT_RADIUS_COL]).reset_index(drop=True)
    ss = df.SID.unique().copy()
    np.random.seed(42); np.random.shuffle(ss)
    fs = len(ss) // 5
    parts = []
    for k in range(5):
        te = ss[k * fs:(k + 1) * fs] if k < 4 else ss[k * fs:]
        m = df.SID.isin(te)
        tr, ts = df[~m], df[m].copy()
        ts['pk']  = kd95((ts.V0, ts.t), *fkd(tr))
        ts['p11'] = c11((ts.V0, ts.t, ts[hcol]), *f11(tr, hcol))
        ts['fold'] = k + 1
        parts.append(ts)
    P = pd.concat(parts)

    fold_kd = [rmse(g.y, g.pk)  for _, g in P.groupby('fold')]
    fold_c  = [rmse(g.y, g.p11) for _, g in P.groupby('fold')]
    row = dict(radius_km=km, N=len(P),
               KD95_RMSE=float(np.mean(fold_kd)),
               C11_RMSE=float(np.mean(fold_c)))
    row['dRMSE'] = row['KD95_RMSE'] - row['C11_RMSE']

    for tag, hs in [('fixed', STRAT_RADIUS_COL), ('self', hcol)]:
        for lab, msk in [('low',  P[hs] < 100),
                         ('med', (P[hs] >= 100) & (P[hs] < 300)),
                         ('high', P[hs] >= 300)]:
            s = P[msk]
            row[f'{lab}_N_{tag}']       = len(s)
            row[f'{lab}_C11bias_{tag}']  = float(np.mean(s.p11 - s.y)) if len(s) else np.nan
            row[f'{lab}_KD95bias_{tag}'] = float(np.mean(s.pk  - s.y)) if len(s) else np.nan
    return row

# ---- self-check at the adopted radius --------------------------------------
chk = run(75, STRAT_RADIUS_COL)
print(f"self-check: KD95 {chk['KD95_RMSE']:.3f}  C11 {chk['C11_RMSE']:.3f}"
      f"   (published 9.86 / 9.22)")
print(f"            high-terrain C11 bias at 75 km {chk['high_C11bias_fixed']:+.3f}"
      f"   (published +0.17)\n")

# ---- full sweep ------------------------------------------------------------
R = pd.DataFrame([run(km, h) for km, h in RAD])
R.to_csv(OUT, index=False)

print(" r_km     N    dRMSE   Bias_high_fixed   Bias_high_self   (KD95 high bias)")
for _, r in R.iterrows():
    print(f"{r.radius_km:5.0f} {r.N:5.0f}    {r.dRMSE:5.3f}         {r.high_C11bias_fixed:+6.2f}"
          f"           {r.high_C11bias_self:+6.2f}          {r.high_KD95bias_fixed:+6.2f}")

f = R.set_index('radius_km')['high_C11bias_fixed']
print(f"\nBias_high_fixed is minimized at {f.idxmin():.0f} km ({f.min():+.3f} kt); "
      f"below 0.25 kt over {', '.join(str(int(x)) for x in f[f < 0.25].index)} km.")
print(f"dRMSE peaks at {R.set_index('radius_km')['dRMSE'].idxmax():.0f} km "
      f"({R.dRMSE.max():.3f} kt).")
print(f"\nwrote {OUT}")
