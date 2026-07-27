"""
stage10_error_floor_mc.py -- how low could any decay model go? (Section 5, limitations)
========================================================================================
The aggregate RMSE improvement is modest in absolute terms, so the limitations
section asks what the achievable floor is given best-track uncertainty alone.

The experiment treats C11 as exactly true, generates synthetic winds from it,
perturbs them by errors of the magnitude expected of Dvorak estimates (a
storm-correlated bias of 7 kt and an independent per-point error of 5 kt), then
refits C11 to the perturbed data and scores it. Whatever RMSE survives is
irreducible: it is what a perfect model would still register against this record.

Reproduces the "achievable RMSE floor near 6.3 kt" of Section 5.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

np.seterr(all="ignore")

POINTS = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\CSV\ph_decay_with_terrain_5m_circ.csv"
TERRAIN_COL = "hmean_rad75"
SIGMA_STORM = 7.0     # storm-correlated bias, kt
SIGMA_POINT = 5.0     # independent per-point error, kt
N_REALISATIONS = 500
SEED = 42


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

a_hat = curve_fit(c11, (d.V0, d.t, d.h), d.y, p0=[1.34e-4],
                  bounds=([1e-5], [1e-3]), maxfev=10000)[0]
truth = c11((d.V0.values, d.t.values, d.h.values), *a_hat)
print(f"C11 fitted on the real data: a = {a_hat[0]:.6e}")

rng = np.random.RandomState(SEED)
sids = d.SID.unique()
rows = {s: d.index[d.SID == s].values for s in sids}

floors = []
for _ in range(N_REALISATIONS):
    y = truth.copy()
    for s in sids:
        ix = rows[s]
        y[ix] += rng.normal(0, SIGMA_STORM) + rng.normal(0, SIGMA_POINT, len(ix))
    y = np.clip(y, 15, None)
    V0p = pd.Series(y, index=d.index).groupby(d.SID).transform("first").values
    pa = curve_fit(c11, (V0p, d.t.values, d.h.values), y, p0=[1.34e-4],
                   bounds=([1e-5], [1e-3]), maxfev=10000)[0]
    floors.append(rmse(y, c11((V0p, d.t.values, d.h.values), *pa)))

floors = np.array(floors)
print(f"\nsigma_storm = {SIGMA_STORM:.0f} kt, sigma_point = {SIGMA_POINT:.0f} kt, "
      f"{N_REALISATIONS} realisations")
print(f"  achievable RMSE floor, median   {np.median(floors):.2f} kt   (paper: near 6.3)")
print(f"  90% interval                    [{np.percentile(floors, 5):.2f}, "
      f"{np.percentile(floors, 95):.2f}] kt")
print(f"\nFor comparison, C11 attains 9.22 kt out of sample, so roughly "
      f"{100 * np.median(floors) / 9.22:.0f}% of the residual is irreducible "
      f"best-track noise.")
