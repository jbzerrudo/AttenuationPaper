"""
Global (in-sample) fits of Section 2c and the dimensional two-coefficient
variant, v2 (archive release v1.0.2).

As archived in v1.0.1 this script pointed at a pre-revision points file that
is not shipped and passed the 50-km h_mean column rather than the reported
75-km predictor.  This version reads the released archive directly:

  data/ph_decay_with_terrain_archive.csv

NATIVE 6-hourly  : in_analysis == True (453 pts, 121 storms),
                   re-referenced t_used / V0_used, terrain = hmean_rad75.
3-hourly         : all overland rows (h_max > 0; 1100 pts, 174 storms),
                   original t_hours / V0 referencing, terrain = hmean_rad75.

Reproduces: KD95 global fit Vb = 39.1 kt, R = 1.00, alpha = 0.043 /hr,
in-sample RMSE 10.10 kt (5.20 m/s); the 3-hourly sensitivity fit
Vb = 39.0, R = 1.00, alpha = 0.039; the global C11 coefficient
a = 1.4289e-4; and the dimensional two-coefficient fit a = 1.42e-4,
b = 1.46e-4 (ratio 0.97; effective R at 200 m and 1000 m: 0.97, 0.85).

Run from the repository root:  python scripts/global_fits.py
"""
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
np.seterr(all='ignore')

def kd95(X, Vb, R, al): V0, t = X; return Vb + (R*V0 - Vb)*np.exp(-al*t)
def c9(X, a):   V0, t = X;       return V0 - a*V0**2*t
def c11(X, a):  V0, t, h = X;    return V0 - a*V0*(V0*t + h)
def c15(X, a, b): V0, t, h, s = X; return ((V0*t) + h)*a*(s + V0 + b) + V0
def c11b(X, a, b): V0, t, h = X; return V0*(1 - a*V0*t - b*h)
def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))
def r2(o, p):   return float(1 - np.sum((o - p)**2)/np.sum((o - np.mean(o))**2))

raw = pd.read_csv('data/ph_decay_with_terrain_archive.csv', parse_dates=['ISO_TIME'])

def build(native):
    if native:
        d = raw[raw.in_analysis == True].copy()
        d = d.rename(columns={'t_used': 't', 'V0_used': 'V0x'})
        d['V0'] = d['V0x']
    else:
        d = raw[raw.h_max > 0].copy()
        d['t'] = d['t_hours']
    d = d.dropna(subset=['hmean_rad75', 'USA_WIND', 'V0', 't', 'STORM_SPD'])
    d = d[d.USA_WIND > 0]
    return d.rename(columns={'USA_WIND': 'y', 'hmean_rad75': 'h',
                             'STORM_SPD': 'spd'})[['SID', 'V0', 't', 'y', 'h', 'spd']].reset_index(drop=True)

for tag, native in [("NATIVE 6-hourly (in_analysis, t_used/V0_used, hbar 75 km)", True),
                    ("3-hourly overland (t_hours/V0, hbar 75 km)", False)]:
    d = build(native)
    pk, _  = curve_fit(kd95, (d.V0, d.t), d.y, p0=[15, .9, .05], bounds=([0, .3, .001], [60, 1, .5]), maxfev=20000)
    p9, _  = curve_fit(c9,  (d.V0, d.t), d.y, p0=[1.5e-4], bounds=([1e-5], [1e-3]), maxfev=20000)
    p11, _ = curve_fit(c11, (d.V0, d.t, d.h), d.y, p0=[1.34e-4], bounds=([1e-5], [1e-3]), maxfev=20000)
    p15, _ = curve_fit(c15, (d.V0, d.t, d.h, d.spd), d.y, p0=[-1.75e-4, -35], bounds=([-1e-2, -100], [0, 0]), maxfev=20000)
    pb, _  = curve_fit(c11b, (d.V0, d.t, d.h), d.y, p0=[1.3e-4, 1.3e-4], bounds=([1e-6, 1e-6], [1e-2, 1e-2]), maxfev=40000)
    yk  = kd95((d.V0, d.t), *pk);  y9 = c9((d.V0, d.t), *p9)
    y11 = c11((d.V0, d.t, d.h), *p11); y15 = c15((d.V0, d.t, d.h, d.spd), *p15)
    print(f"\n===== {tag} : N={len(d)} pts, {d.SID.nunique()} storms =====")
    print(f"KD95:  Vb={pk[0]:.2f} R={pk[1]:.3f} alpha={pk[2]:.4f}  inRMSE={rmse(d.y, yk):.2f}  R2={r2(d.y, yk):.3f}")
    print(f"C9 :   a={p9[0]:.3e}  inRMSE={rmse(d.y, y9):.2f}  MSE(loss)={np.mean((d.y - y9)**2):.1f}  R2={r2(d.y, y9):.3f}")
    print(f"C11:   a={p11[0]:.6e}  inRMSE={rmse(d.y, y11):.2f}  MSE(loss)={np.mean((d.y - y11)**2):.1f}  R2={r2(d.y, y11):.3f}")
    print(f"C15:   a={p15[0]:.3e} b={p15[1]:.1f}  inRMSE={rmse(d.y, y15):.2f}  MSE(loss)={np.mean((d.y - y15)**2):.1f}")
    print(f"C11b:  a={pb[0]:.3e}/(kt hr)  b={pb[1]:.3e}/m  ratio={pb[0]/pb[1]:.2f}  "
          f"Reff200={1 - pb[1]*200:.3f} Reff1000={1 - pb[1]*1000:.3f}")
