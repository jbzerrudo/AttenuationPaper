"""
Land-fraction control test, v2.

v1 used a single 5-fold split and was dominated by split noise.  This version:
  (1) repeats the storm-stratified 5-fold CV over many random splits
  (2) bootstraps the fitted coefficients directly (storm-clustered), which is
      the low-noise way to ask whether elevation survives controlling for
      land fraction and vice versa.
"""
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
import warnings; warnings.filterwarnings('ignore')

NFOLD, NREP, NBOOT = 5, 60, 2000

d = pd.read_csv('data/ph_decay_with_terrain_5m_circ.csv', parse_dates=['ISO_TIME'])
t = pd.read_csv('data/terrain_sweeps.csv', parse_dates=['ISO_TIME'])
m = d.merge(t[['SID', 'ISO_TIME', 'npx_r50']], on=['SID', 'ISO_TIME'], how='left')
m['hour'] = m.ISO_TIME.dt.hour
m['lf'] = m.npx_r50 / m.npx_r50.max()


def subset(hcol, minpts=3):
    s = m[m.hour.isin([0, 6, 12, 18])]
    s = s[s[hcol].notna() & s.lf.notna() & (s.V0 >= 34)]
    c = s.groupby('SID').size()
    return s[s.SID.isin(c[c >= minpts].index)].reset_index(drop=True)


def f_M0(X, a):            V0, t_, h, lf = X; return V0 * (1 - a * V0 * t_)
def f_Mh(X, a, b):         V0, t_, h, lf = X; return V0 * (1 - a * V0 * t_ - b * h)
def f_Mlf(X, a, c):        V0, t_, h, lf = X; return V0 * (1 - a * V0 * t_ - c * lf)
def f_Mboth(X, a, b, c):   V0, t_, h, lf = X; return V0 * (1 - a * V0 * t_ - b * h - c * lf)
def f_KD95(X, Vb, R, al):  V0, t_, h, lf = X; return Vb + (R * V0 - Vb) * np.exp(-al * t_)

MODELS = {
    'KD95':  (f_KD95,  [39., 1., .04], ([0, .3, 0], [200, 1., 1.])),
    'M0':    (f_M0,    [1.4e-4], ([-1], [1])),
    'Mh':    (f_Mh,    [1.4e-4, 1.4e-4], ([-1, -1], [1, 1])),
    'Mlf':   (f_Mlf,   [1.4e-4, .05], ([-1, -1], [1, 1])),
    'Mboth': (f_Mboth, [1.4e-4, 1.4e-4, .05], ([-1, -1, -1], [1, 1, 1])),
}


def pack(df, hcol):
    return (df.V0.values.astype(float), df.t_hours.values.astype(float),
            df[hcol].values.astype(float), df.lf.values.astype(float))


def fit(name, df, hcol):
    f, p0, b = MODELS[name]
    return curve_fit(f, pack(df, hcol), df.USA_WIND.values.astype(float),
                     p0=p0, bounds=b, maxfev=200000)[0]


def repeated_cv(df, hcol):
    sids = np.array(sorted(df.SID.unique()))
    per_rep = {n: [] for n in MODELS}
    for rep in range(NREP):
        rng = np.random.default_rng(1000 + rep)
        u = sids.copy(); rng.shuffle(u)
        fmap = {s: i % NFOLD for i, s in enumerate(u)}
        fold = df.SID.map(fmap).values
        pooled = {n: np.full(len(df), np.nan) for n in MODELS}
        for k in range(NFOLD):
            tr, te = df[fold != k], df[fold == k]
            for n in MODELS:
                try:
                    p = fit(n, tr, hcol)
                    pooled[n][fold == k] = MODELS[n][0](pack(te, hcol), *p)
                except Exception:
                    pass
        o = df.USA_WIND.values
        for n in MODELS:
            per_rep[n].append(np.sqrt(np.nanmean((pooled[n] - o) ** 2)))
    return {n: np.array(v) for n, v in per_rep.items()}


def boot_coefs(df, hcol):
    sid = df.SID.values
    idx = {s: np.where(sid == s)[0] for s in np.unique(sid)}
    us = np.array(list(idx)); rng = np.random.default_rng(7)
    out = []
    for _ in range(NBOOT):
        pick = rng.choice(us, len(us), replace=True)
        j = np.concatenate([idx[s] for s in pick])
        try:
            out.append(fit('Mboth', df.iloc[j], hcol))
        except Exception:
            pass
    return np.array(out)


for hcol, tag in [('hmean_rad50', 'SCALE-MATCHED  h(50 km) vs lf(50 km)'),
                  ('hmean_rad75', 'PAPER PREDICTOR h(75 km) vs lf(50 km)')]:
    df = subset(hcol)
    r = df[hcol].corr(df.lf)
    print(f'\n{"="*78}\n{tag}    n={len(df)} pts, {df.SID.nunique()} storms,  corr(h,lf) = {r:+.3f}\n{"="*78}')

    cv = repeated_cv(df, hcol)
    print(f'  pooled out-of-sample RMSE over {NREP} random storm-stratified 5-fold splits')
    print(f'  {"model":6s} {"mean":>7s} {"sd":>6s}')
    for n in MODELS:
        print(f'  {n:6s} {cv[n].mean():7.3f} {cv[n].std():6.3f}')

    print(f'\n  paired gains across the same {NREP} splits (kt, positive = second model better)')
    for a, b in [('M0', 'Mh'), ('M0', 'Mlf'), ('Mlf', 'Mboth'), ('Mh', 'Mboth')]:
        g = cv[a] - cv[b]
        print(f'    {b:5s} over {a:5s}: {g.mean():+6.3f}  sd {g.std():5.3f}   '
              f'better in {100*(g>0).mean():5.1f}% of splits')

    bc = boot_coefs(df, hcol)
    names = ['a  (per kt hr)', 'b  (elevation, per m)', 'c  (land fraction, per unit)']
    print(f'\n  storm-clustered bootstrap of the JOINT fit (Mboth), {len(bc)} resamples')
    for i, nm in enumerate(names):
        v = bc[:, i]; lo, hi = np.percentile(v, [2.5, 97.5])
        sign = 'EXCLUDES 0' if lo * hi > 0 else 'includes 0'
        print(f'    {nm:30s} {v.mean():+.3e}  CI95 [{lo:+.3e}, {hi:+.3e}]  {sign}')
