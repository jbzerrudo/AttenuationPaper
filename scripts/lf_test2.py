"""
Land-fraction control test, v3 (archive release v1.0.2).

v1 used a single 5-fold split and was dominated by split noise.  v2 added
repeated splits and a storm-clustered coefficient bootstrap, but as archived
it read a pre-revision points file and referenced t_hours/V0 instead of the
re-referenced t_used/V0_used columns, so it did not reproduce the published
numbers.  This version reads the released archive directly:

  data/ph_decay_with_terrain_archive.csv   (in_analysis == True -> 453 pts, 121 storms)
  data/terrain_sweeps.csv                  (npx_r50 -> land fraction)
  data/fold_map.csv                        (the published 5-fold storm split)

and reproduces, in order:
  (A) the fold-mean RMSE quartet of Table 4 / Section 3e
      (time only 9.70, +land fraction 9.48, +hbar 9.25, both 9.31),
      the stratified bias decomposition, the land-fraction median (0.53),
      and the predictor correlations;
  (B) the paired gains over 60 repeated storm-stratified 5-fold splits
      (the -0.05 kt and +0.11 kt increments quoted in Section 3e);
  (C) the storm-clustered bootstrap of the joint fit
      (hbar coefficient 1.62e-4, CI [0.38, 2.97]e-4;
       land-fraction coefficient CI [-5.6, +3.7]e-2).

Run from the repository root:  python scripts/lf_test2.py
"""
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
import warnings; warnings.filterwarnings('ignore')

NFOLD, NREP, NBOOT = 5, 60, 2000

d = pd.read_csv('data/ph_decay_with_terrain_archive.csv', parse_dates=['ISO_TIME'])
t = pd.read_csv('data/terrain_sweeps.csv', parse_dates=['ISO_TIME'])
fmap = pd.read_csv('data/fold_map.csv', parse_dates=['ISO_TIME'])

m = d[d.in_analysis == True].merge(t[['SID', 'ISO_TIME', 'npx_r50']],
                                   on=['SID', 'ISO_TIME'], how='left')
m = m.merge(fmap, on=['SID', 'ISO_TIME'], how='left')
m['lf'] = m.npx_r50 / t.npx_r50.max()
assert m.fold.notna().all() and m.lf.notna().all() and len(m) == 453


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
    return (df.V0_used.values.astype(float), df.t_used.values.astype(float),
            df[hcol].values.astype(float), df.lf.values.astype(float))


def fit(name, df, hcol):
    f, p0, b = MODELS[name]
    return curve_fit(f, pack(df, hcol), df.USA_WIND.values.astype(float),
                     p0=p0, bounds=b, maxfev=200000)[0]


def cv_published_split(df, hcol):
    """Fold-mean RMSE and pooled out-of-fold predictions on data/fold_map.csv."""
    out = {}
    for n in MODELS:
        fold_rmse, pred = [], np.full(len(df), np.nan)
        for k in sorted(df.fold.unique()):
            tr, te = df[df.fold != k], df[df.fold == k]
            p = fit(n, tr, hcol)
            pr = MODELS[n][0](pack(te, hcol), *p)
            pred[(df.fold == k).values] = pr
            fold_rmse.append(np.sqrt(np.mean((pr - te.USA_WIND.values) ** 2)))
        out[n] = (np.mean(fold_rmse), pred)
    return out


def repeated_cv(df, hcol):
    sids = np.array(sorted(df.SID.unique()))
    per_rep = {n: [] for n in MODELS}
    for rep in range(NREP):
        rng = np.random.default_rng(1000 + rep)
        u = sids.copy(); rng.shuffle(u)
        fmap_ = {s: i % NFOLD for i, s in enumerate(u)}
        fold = df.SID.map(fmap_).values
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


print(f'ANALYSIS SUBSET: n={len(m)} pts, {m.SID.nunique()} storms')
print(f'land fraction: median = {m.lf.median():.4f}  '
      f'corr with hbar(75 km) = {m.lf.corr(m.hmean_rad75):+.4f}, '
      f'hbar(50 km) = {m.lf.corr(m.hmean_rad50):+.4f}, '
      f'h_max = {m.lf.corr(m.h_max):+.4f}')

for hcol, tag in [('hmean_rad50', 'SCALE-MATCHED  h(50 km) vs lf(50 km)'),
                  ('hmean_rad75', 'PAPER PREDICTOR h(75 km) vs lf(50 km)')]:
    r = m[hcol].corr(m.lf)
    print(f'\n{"="*78}\n{tag}    corr(h,lf) = {r:+.3f}\n{"="*78}')

    print(f'\n  (A) published 5-fold split (data/fold_map.csv), fold-mean RMSE')
    pub = cv_published_split(m, hcol)
    for n in MODELS:
        print(f'  {n:6s} {pub[n][0]:7.4f}')
    h75 = m.hmean_rad75.values
    print(f'      stratified mean bias (pred-obs) by terrain, and range:')
    for n in ['M0', 'Mlf', 'Mh']:
        e = pub[n][1] - m.USA_WIND.values
        b = [e[h75 < 100].mean(), e[(h75 >= 100) & (h75 < 300)].mean(), e[h75 >= 300].mean()]
        print(f'      {n:5s} low {b[0]:+5.2f}  med {b[1]:+5.2f}  high {b[2]:+5.2f}   range {max(b)-min(b):.2f}')

    cv = repeated_cv(m, hcol)
    print(f'\n  (B) pooled out-of-sample RMSE over {NREP} random storm-stratified 5-fold splits')
    print(f'  {"model":6s} {"mean":>7s} {"sd":>6s}')
    for n in MODELS:
        print(f'  {n:6s} {cv[n].mean():7.3f} {cv[n].std():6.3f}')
    print(f'\n      paired gains across the same {NREP} splits (kt, positive = second model better)')
    for a, b in [('M0', 'Mh'), ('M0', 'Mlf'), ('Mlf', 'Mboth'), ('Mh', 'Mboth')]:
        g = cv[a] - cv[b]
        lo, hi = np.percentile(g, [2.5, 97.5])
        print(f'        {b:5s} over {a:5s}: {g.mean():+6.3f}  CI95 [{lo:+.3f}, {hi:+.3f}]   '
              f'better in {100*(g>0).mean():5.1f}% of splits')

    bc = boot_coefs(m, hcol)
    names = ['a  (per kt hr)', 'b  (elevation, per m)', 'c  (land fraction, per unit)']
    print(f'\n  (C) storm-clustered bootstrap of the JOINT fit (Mboth), {len(bc)} resamples')
    for i, nm in enumerate(names):
        v = bc[:, i]; lo, hi = np.percentile(v, [2.5, 97.5])
        sign = 'EXCLUDES 0' if lo * hi > 0 else 'includes 0'
        print(f'        {nm:30s} {v.mean():+.3e}  CI95 [{lo:+.3e}, {hi:+.3e}]  {sign}')
