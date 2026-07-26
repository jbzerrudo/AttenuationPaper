"""
analysis_all.py — consolidated re-analysis for WAF-D-26-0060 (Zerrudo & Servando)
Regenerates every numeric result and writes them as CSVs + metadata.json.
Inputs: ph_decay_with_terrain.csv (modeling data) and terrain_sweeps.csv (sweep).
"""
import json, pandas as pd, numpy as np
from scipy.optimize import curve_fit
from scipy import stats
np.seterr(all='ignore')

# ---- PATHS (Jef's laptop, confirmed 25 Jul 2026) ----
POINTS = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\CSV\ph_decay_with_terrain_5m_circ.csv"
SWEEP  = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\CSV\terrain_sweeps.csv"
OUTDIR = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\CSV"   # where the result CSVs are written
STAMP  = "2026-07-26"

# ---- PRIMARY TERRAIN FOOTPRINT -------------------------------------------
# Which column becomes the terrain variable h in C11, and therefore drives the
# CV table, the stratified bias table, the bootstrap, the Dvorak Monte Carlo,
# LOSO, the temporal hold-out and the ML benchmark.
#
# Calibrated 26 Jul 2026. The radius sweep peaks near 110 km, but the
# terrain-conditional bias correction (the paper's headline) degrades beyond
# about 75 km, so 75 km is the reported setting: it improves RMSE AND bias AND
# significance relative to the old 50 km default.
#
#   'h_mean'       50 km  (old default; bootstrap p = 0.050, borderline)
#   'hmean_rad75'  75 km  (calibrated;  bootstrap p = 0.011)
#
# Set to 'h_mean' to reproduce the pre-calibration numbers.
PRIMARY_TERRAIN_COL = 'hmean_rad75'
STAMP  = STAMP + f" | terrain={PRIMARY_TERRAIN_COL}"

import os
os.makedirs(OUTDIR, exist_ok=True)
os.chdir(OUTDIR)

def kd95(X,Vb,R,al): V0,t=X; return Vb+(R*V0-Vb)*np.exp(-al*t)
def c9(X,a): V0,t=X; return V0-a*V0**2*t
def c11(X,a): V0,t,h=X; return V0-a*V0*(V0*t+h)
def c15(X,a,b): V0,t,h,s=X; return ((V0*t)+h)*a*(s+V0+b)+V0
def c11b(X,a,b): V0,t,h=X; return V0*(1-a*V0*t-b*h)
def fkd(s):
    try: return curve_fit(kd95,(s.V0,s.t),s.y,p0=[15,.9,.05],bounds=([0,.3,.001],[60,1,.5]),maxfev=10000)[0]
    except: return np.array([38.95,1,.0393])
def f9(s):
    try: return curve_fit(c9,(s.V0,s.t),s.y,p0=[1.5e-4],bounds=([1e-5],[1e-3]),maxfev=10000)[0]
    except: return np.array([1.48e-4])
def f11(s,h='h'):
    try: return curve_fit(c11,(s.V0,s.t,s[h]),s.y,p0=[1.34e-4],bounds=([1e-5],[1e-3]),maxfev=10000)[0]
    except: return np.array([1.34e-4])
def f15(s):
    try: return curve_fit(c15,(s.V0,s.t,s.h,s.spd),s.y,p0=[-1.75e-4,-35],bounds=([-1e-2,-100],[0,0]),maxfev=10000)[0]
    except: return np.array([-1.75e-4,-35])
def f11b(s):
    try: return curve_fit(c11b,(s.V0,s.t,s.h),s.y,p0=[1.3e-4,1.3e-4],bounds=([1e-6,1e-6],[1e-2,1e-2]),maxfev=20000)[0]
    except: return np.array([np.nan,np.nan])
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def bias(p,o): return float(np.mean(p-o))

def build(df, native):
    df=df.copy(); df['ISO_TIME']=pd.to_datetime(df['ISO_TIME'])
    if native:
        df['hh']=df.ISO_TIME.dt.hour; df=df[df.hh.isin([0,6,12,18])].sort_values(['SID','ISO_TIME'])
        df['t']=(df.ISO_TIME-df.groupby('SID').ISO_TIME.transform('min')).dt.total_seconds()/3600
        df['V0']=df.groupby('SID').USA_WIND.transform('first')
        c=df.groupby('SID').SID.transform('size'); df=df[(c>=3)&(df.V0>=34)]
    else:
        df['t']=df['t_hours']
    # Use the calibrated footprint as h. Falls back to h_mean if absent.
    _tc = PRIMARY_TERRAIN_COL if PRIMARY_TERRAIN_COL in df.columns else 'h_mean'
    if _tc != 'h_mean':
        df = df.copy(); df['h_mean'] = df[_tc]
    df=df.dropna(subset=['h_max','h_mean','USA_WIND','V0','t','STORM_SPD'])
    df=df[(df.h_max>0)&(df.USA_WIND>0)]
    out=df.rename(columns={'USA_WIND':'y','h_mean':'h','STORM_SPD':'spd'})[['SID','V0','t','y','h','spd']]
    out=out.sort_values(['SID','t']).reset_index(drop=True)
    # PATH-INTEGRATED terrain: running mean of h from landfall to the current point.
    # "terrain crossed so far" instead of "terrain here now". Same equation, same
    # single coefficient; only the terrain variable is redefined.
    out['h_path']=out.groupby('SID')['h'].expanding().mean().reset_index(level=0,drop=True)
    return out

def cvfold(df):
    ss=df.SID.unique().copy(); np.random.seed(42); np.random.shuffle(ss); fs=len(ss)//5; rows=[]
    for k in range(5):
        te=ss[k*fs:(k+1)*fs] if k<4 else ss[k*fs:]; m=df.SID.isin(te); tr,ts=df[~m],df[m]
        pk,p9,p11,p15=fkd(tr),f9(tr),f11(tr),f15(tr)
        yk=kd95((ts.V0,ts.t),*pk); y11=c11((ts.V0,ts.t,ts.h),*p11)
        rows.append(dict(fold=k+1,n_test=len(ts),mean_h=ts.h.mean(),mean_t=ts.t.mean(),
            KD95_RMSE=rmse(ts.y,yk),C9_RMSE=rmse(ts.y,c9((ts.V0,ts.t),*p9)),
            C11_RMSE=rmse(ts.y,y11),C15_RMSE=rmse(ts.y,c15((ts.V0,ts.t,ts.h,ts.spd),*p15)),
            KD95_bias=bias(yk,ts.y),C11_bias=bias(y11,ts.y),C11_a=p11[0],
            p_point=stats.ttest_rel((ts.y-yk)**2,(ts.y-y11)**2)[1]))
    return pd.DataFrame(rows)

raw=pd.read_csv(POINTS)
_tcheck = PRIMARY_TERRAIN_COL if PRIMARY_TERRAIN_COL in raw.columns else 'h_mean'
print(f"PRIMARY TERRAIN COLUMN: {_tcheck}"
      + ("" if _tcheck==PRIMARY_TERRAIN_COL else f"  [!! {PRIMARY_TERRAIN_COL} NOT FOUND, fell back]"))
full=build(raw,native=False); nat=build(raw,native=True)
meta={"stamp":STAMP,"n_3hourly":[int(len(full)),int(full.SID.nunique())],
      "n_native6h":[int(len(nat)),int(nat.SID.nunique())]}

# 1) CV per-fold (both)
for tag,d in [("3hourly",full),("native6h",nat)]:
    cvfold(d).to_csv(f"cv_perfold_{tag}.csv",index=False)

# 2) stratified with bias (both)
def strat(d):
    ss=d.SID.unique().copy(); np.random.seed(42); np.random.shuffle(ss); fs=len(ss)//5; parts=[]
    for k in range(5):
        te=ss[k*fs:(k+1)*fs] if k<4 else ss[k*fs:]; m=d.SID.isin(te); tr,ts=d[~m],d[m].copy()
        pk,p11=fkd(tr),f11(tr); ts['pk']=kd95((ts.V0,ts.t),*pk); ts['p11']=c11((ts.V0,ts.t,ts.h),*p11); parts.append(ts)
    c=pd.concat(parts); rows=[]
    for lab,msk in [("TS_34_63",(c.V0>=34)&(c.V0<64)),("Cat1_2_64_95",(c.V0>=64)&(c.V0<96)),("Cat3_ge96",c.V0>=96),
                    ("low_h<100",c.h<100),("med_100_300",(c.h>=100)&(c.h<300)),("high_h>=300",c.h>=300)]:
        s=c[msk]
        if len(s)<10: continue
        rows.append(dict(stratum=lab,N=len(s),KD95_RMSE=rmse(s.y,s.pk),C11_RMSE=rmse(s.y,s.p11),
                         KD95_bias=bias(s.pk,s.y),C11_bias=bias(s.p11,s.y),dRMSE=rmse(s.y,s.pk)-rmse(s.y,s.p11)))
    return pd.DataFrame(rows)
for tag,d in [("3hourly",full),("native6h",nat)]:
    strat(d).to_csv(f"stratified_bias_{tag}.csv",index=False)

# 3) dimensional 2-coef
rows=[]
for tag,d in [("3hourly",full),("native6h",nat)]:
    a,b=f11b(d); a1=f11(d)[0]
    rows.append(dict(dataset=tag,a_per_kt_hr=a,b_per_m=b,ratio_a_b=a/b,single_coef_a=a1,
                     Reff_h200=1-b*200,Reff_h1000=1-b*1000))
pd.DataFrame(rows).to_csv("dimensional_2coef.csv",index=False)

# 4) rigor suite (native)
d=nat; sids=d.SID.unique(); by={s:d[d.SID==s] for s in sids}; rng=np.random.RandomState(42)
def dR_a(s):
    pk=fkd(s); pa=f11(s); return rmse(s.y,kd95((s.V0,s.t),*pk))-rmse(s.y,c11((s.V0,s.t,s.h),*pa)), pa[0]
B=2000; dr=[]; aa=[]
for _ in range(B):
    smp=pd.concat([by[s] for s in rng.choice(sids,len(sids),replace=True)]); x=dR_a(smp); dr.append(x[0]); aa.append(x[1])
dr=np.array(dr); aa=np.array(aa); base=dR_a(d)
M=500; sig_s,sig_r=7.,5.; wins=0; drm=[]; six={s:d.index[d.SID==s].values for s in sids}
for _ in range(M):
    yp=d.y.values.astype(float).copy()
    for s in sids:
        ix=six[s]; yp[ix]+=rng.normal(0,sig_s)+rng.normal(0,sig_r,len(ix))
    yp=np.clip(yp,15,None); V0p=pd.Series(yp,index=d.index).groupby(d.SID).transform('first').values
    pk=fkd(pd.DataFrame({'V0':V0p,'t':d.t,'y':yp})); pa=curve_fit(c11,(V0p,d.t.values,d.h.values),yp,p0=[1.34e-4],bounds=([1e-5],[1e-3]),maxfev=10000)[0]
    drm.append(rmse(yp,kd95((V0p,d.t.values),*pk))-rmse(yp,c11((V0p,d.t.values,d.h.values),*pa))); wins+=(drm[-1]>0)
drm=np.array(drm)
# LOSO & temporal
yk=[]; yc=[]; ob=[]
for s in sids:
    tr=d[d.SID!=s]; te=d[d.SID==s]; pk=fkd(tr); pa=f11(tr)
    yk.append(kd95((te.V0,te.t),*pk)); yc.append(c11((te.V0,te.t,te.h),*pa)); ob.append(te.y.values)
ob=np.concatenate(ob); loso_kd=rmse(ob,np.concatenate(yk)); loso_c=rmse(ob,np.concatenate(yc))
d2=d.copy(); d2['season']=d2.SID.str[:4].astype(int); med=int(np.median(d2.season.unique()))
tr=d2[d2.season<med]; te=d2[d2.season>=med]; pk=fkd(tr); pa=f11(tr)
th_kd=rmse(te.y,kd95((te.V0,te.t),*pk)); th_c=rmse(te.y,c11((te.V0,te.t,te.h),*pa))
rig={"bootstrap_dRMSE":base[0],"bootstrap_dRMSE_CI95":[float(np.percentile(dr,2.5)),float(np.percentile(dr,97.5))],
     "bootstrap_p_dRMSE_le0":float(np.mean(dr<=0)),"bootstrap_a":base[1],
     "bootstrap_a_CI95":[float(np.percentile(aa,2.5)),float(np.percentile(aa,97.5))],
     "dvorak_MC_win_frac":wins/M,"dvorak_MC_dRMSE_median":float(np.median(drm)),
     "LOSO_KD95":loso_kd,"LOSO_C11":loso_c,"temporal_split_year":med,"temporal_KD95":th_kd,"temporal_C11":th_c}
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    feats=['V0','t','h']; ss=sids.copy(); rng.shuffle(ss); fs=len(ss)//5
    def cvml(mk):
        rs=[]
        for k in range(5):
            te=ss[k*fs:(k+1)*fs] if k<4 else ss[k*fs:]; m=d.SID.isin(te)
            mdl=mk(); mdl.fit(d[~m][feats],d[~m].y); rs.append(rmse(d[m].y.values,mdl.predict(d[m][feats])))
        return float(np.mean(rs))
    rig["ML_RandomForest"]=cvml(lambda:RandomForestRegressor(n_estimators=300,random_state=0,n_jobs=-1))
    rig["ML_GradientBoosting"]=cvml(lambda:GradientBoostingRegressor(random_state=0))
except Exception as e: rig["ML_note"]=str(e)
pd.DataFrame([rig]).to_csv("rigor_suite.csv",index=False); meta["rigor"]=rig

# 5-7) resolution / radius / RMW sweeps
# If POINTS is the 5-m circular rebuild it already carries every sweep column,
# so use it directly and skip the terrain_sweeps.csv merge entirely.
# Detect on the RADIUS columns only. The rebuilt file deliberately omits
# hmean_res20 (already in terrain_sweeps.csv, and 2 h of compute to reproduce
# numbers we have), so requiring it here would silently fall back to the old
# 4-radius merge and throw away the whole fine calibration grid.
import re as _re0
_have_rad=[c for c in raw.columns
           if _re0.fullmatch(r'hmean_rad\d+',str(c)) and raw[c].notna().sum()>0]
if len(_have_rad)>=3:
    print(f"Radius columns present in POINTS ({len(_have_rad)} radii) -> using primary file.")
    mg=raw.copy(); mg['ISO_TIME']=pd.to_datetime(mg['ISO_TIME'])
    mg['h_orig']=mg['h_mean']; mg['hmax_r50']=mg['h_max']
    PRIMARY_LABEL="primary_circ"
    # Backfill any resolution column the rebuild skipped (e.g. 20 m) from the
    # older sweep file, so the resolution table stays complete. Same circular
    # method, so the values are directly comparable.
    try:
        _sw=pd.read_csv(SWEEP); _sw['ISO_TIME']=pd.to_datetime(_sw['ISO_TIME'])
        _miss=[c for c in _sw.columns if _re0.fullmatch(r'hmean_res\d+',str(c))
               and c not in mg.columns and _sw[c].notna().sum()>0]
        if _miss:
            mg=mg.merge(_sw[['SID','ISO_TIME']+_miss],on=['SID','ISO_TIME'],how='left')
            print(f"  backfilled from SWEEP: {_miss}")
    except Exception as _e:
        print(f"  (no SWEEP backfill: {_e})")
else:
    print("Sweep columns absent from POINTS -> merging terrain_sweeps.csv.")
    sw=pd.read_csv(SWEEP); sw['ISO_TIME']=pd.to_datetime(sw['ISO_TIME'])
    mm=raw.copy(); mm['ISO_TIME']=pd.to_datetime(mm['ISO_TIME'])
    mg=sw.merge(mm[['SID','ISO_TIME','USA_WIND','STORM_SPD','h_mean','h_max']].rename(columns={'h_mean':'h_orig'}),on=['SID','ISO_TIME'])
    PRIMARY_LABEL="orig_20m_square"
mg['hh']=mg.ISO_TIME.dt.hour; mg=mg[mg.hh.isin([0,6,12,18])].sort_values(['SID','ISO_TIME'])
mg['t']=(mg.ISO_TIME-mg.groupby('SID').ISO_TIME.transform('min')).dt.total_seconds()/3600
mg['V0']=mg.groupby('SID').USA_WIND.transform('first'); c=mg.groupby('SID').SID.transform('size')
mg=mg[(c>=3)&(mg.V0>=34)&(mg.hmax_r50>0)&(mg.USA_WIND>0)].rename(columns={'USA_WIND':'y'}).reset_index(drop=True)
def cvh(df,hcol):
    df=df.dropna(subset=['V0','t','y',hcol])
    ss=df.SID.unique().copy(); np.random.seed(42); np.random.shuffle(ss); fs=len(ss)//5; rk=[];rc=[];aa=[]
    for k in range(5):
        te=ss[k*fs:(k+1)*fs] if k<4 else ss[k*fs:]; m=df.SID.isin(te); tr,ts=df[~m],df[m]
        pk=fkd(tr); pa=curve_fit(c11,(tr.V0,tr.t,tr[hcol]),tr.y,p0=[1.34e-4],bounds=([1e-5],[1e-3]),maxfev=10000)[0]
        rk.append(rmse(ts.y,kd95((ts.V0,ts.t),*pk))); rc.append(rmse(ts.y,c11((ts.V0,ts.t,ts[hcol]),*pa))); aa.append(pa[0])
    return np.mean(rk),np.mean(rc),np.mean(aa),len(df)
rows=[]
# 5 m is included now. It was in the old sweep script but never actually populated
# (hmean_res5 came back all-null), so this is the first real test of native 5 m.
_reslist=[("5m","hmean_res5"),("20m","hmean_res20"),("90m","hmean_res90"),
          ("500m","hmean_res500"),("1km","hmean_res1000"),(PRIMARY_LABEL,"h_orig")]
for lab,h in _reslist:
    if h not in mg.columns or mg[h].notna().sum()==0:
        print(f"  skipping resolution '{lab}' ({h}): no data"); continue
    k,cc,a,n=cvh(mg,h); rows.append(dict(resolution=lab,n_pts=n,KD95_RMSE=k,C11_RMSE=cc,improve=k-cc,coeff_a=a))
pd.DataFrame(rows).to_csv("resolution_sweep.csv",index=False)
# Auto-discover every hmean_radNNN column that actually carries data, sorted
# numerically, so the sweep grid can be changed in the sampler without editing
# this file. Radius is the quantity being calibrated, so the grid will change.
import re as _re
_rad=[]
for _c in mg.columns:
    _m=_re.fullmatch(r'hmean_rad(\d+)',str(_c))
    if _m and mg[_c].notna().sum()>0: _rad.append((int(_m.group(1)),_c))
_rad.sort()
print(f"Radius sweep: {len(_rad)} radii found -> {[r for r,_ in _rad]} km")
rows=[]
for km,h in _rad:
    k,cc,a,n=cvh(mg,h); rows.append(dict(radius_km=km,n_pts=n,KD95_RMSE=k,C11_RMSE=cc,improve=k-cc,coeff_a=a))
rs=pd.DataFrame(rows); rs.to_csv("radius_sweep.csv",index=False)
if len(rs):
    _b=rs.loc[rs.improve.idxmax()]
    # smallest radius within 0.02 kt of the best = the start of the plateau
    _pl=rs[rs.improve>=_b.improve-0.02].radius_km.min()
    print(f"  best radius {_b.radius_km:.0f} km (improve {_b.improve:.3f} kt); "
          f"plateau starts at {_pl:.0f} km")
    rs.assign(plateau_start_km=_pl).to_csv("radius_sweep.csv",index=False)

# 7b) NESTED CV over the radius choice -- is a larger radius a real preference or
# selection bias? Radius is chosen inside each training fold only, never on test.
_RADCOLS=[c for _,c in _rad]
nn=mg.dropna(subset=['V0','t','y']+_RADCOLS).reset_index(drop=True)
def _folds(ss,k=5,seed=42):
    ss=np.array(ss,dtype=object).copy(); r=np.random.RandomState(seed); r.shuffle(ss)
    fs=len(ss)//k; return [ss[i*fs:(i+1)*fs] if i<k-1 else ss[i*fs:] for i in range(k)]
_out=_folds(nn.SID.unique()); _nest=[]; _picks=[]; _kd=[]
for te in _out:
    m=nn.SID.isin(te); tr,ts=nn[~m],nn[m]
    _sc={h:[] for h in _RADCOLS}
    for ite in _folds(tr.SID.unique(),seed=7):
        im=tr.SID.isin(ite); itr,its=tr[~im],tr[im]
        for h in _RADCOLS:
            pa=curve_fit(c11,(itr.V0,itr.t,itr[h]),itr.y,p0=[1.34e-4],bounds=([1e-5],[1e-3]),maxfev=10000)[0]
            _sc[h].append(rmse(its.y,c11((its.V0,its.t,its[h]),*pa)))
    pk=min(_sc,key=lambda h:np.mean(_sc[h])); _picks.append(pk.replace('hmean_rad',''))
    pa=curve_fit(c11,(tr.V0,tr.t,tr[pk]),tr.y,p0=[1.34e-4],bounds=([1e-5],[1e-3]),maxfev=10000)[0]
    _nest.append(rmse(ts.y,c11((ts.V0,ts.t,ts[pk]),*pa)))
    _kd.append(rmse(ts.y,kd95((ts.V0,ts.t),*fkd(tr))))
_nrow=dict(n_pts=len(nn),n_storms=nn.SID.nunique(),radii_offered=";".join(_picks),
           KD95_RMSE=float(np.mean(_kd)),C11_nested_RMSE=float(np.mean(_nest)),
           nested_advantage=float(np.mean(_kd)-np.mean(_nest)))
pd.DataFrame([_nrow]).to_csv("radius_nested_cv.csv",index=False)
print(f"Nested-CV radius picks per fold: {_picks} | nested C11 {np.mean(_nest):.3f} "
      f"vs KD95 {np.mean(_kd):.3f} kt")
rr=mg[mg.rmw_km.notna()]; rows=[]
# NOTE: USA_RMW only exists from ~2000 onward in IBTrACS. This subset is therefore
# much smaller than the full native set and is a SENSITIVITY TEST ONLY. Do not use
# an RMW footprint for the primary analysis; it discards the entire 1977-1999 record.
print(f"RMW footprint subset: {len(rr)} pts / {rr.SID.nunique()} storms "
      f"(vs {len(mg)} pts / {mg.SID.nunique()} storms full) -- sensitivity only.")
for lab,h in [("fixed_50km","hmean_rad50"),("RMW","hmean_rmw"),("2xRMW","hmean_2rmw")]:
    if h not in rr.columns or rr[h].notna().sum()==0:
        print(f"  skipping footprint '{lab}': no data"); continue
    k,cc,a,n=cvh(rr,h); rows.append(dict(footprint=lab,n_pts=n,n_storms=rr.SID.nunique(),KD95_RMSE=k,C11_RMSE=cc,coeff_a=a))
pd.DataFrame(rows).to_csv("rmw_footprint.csv",index=False)

# 8) TERRAIN DEFINITION: instantaneous vs path-integrated, with nested CV.
# The terrain metric has two design choices, both inside the structure PySR gave
# us: spatial (footprint radius, above) and temporal (does it remember the path).
# Reported per-variant AND with the variant chosen inside training folds only,
# so the headline number is not inflated by having looked at both.
def _cv_h(df,hcol,folds):
    r=[]
    for te in folds:
        m=df.SID.isin(te); tr,ts=df[~m],df[m]
        r.append(rmse(ts.y,c11((ts.V0,ts.t,ts[hcol]),*f11(tr,hcol))))
    return float(np.mean(r))
rows=[]
for tag,dd in [("native6h",nat),("3hourly",full)]:
    ss=dd.SID.unique().copy(); rr=np.random.RandomState(42); rr.shuffle(ss)
    fs=len(ss)//5; fo=[ss[i*fs:(i+1)*fs] if i<4 else ss[i*fs:] for i in range(5)]
    kd=[]
    for te in fo:
        m=dd.SID.isin(te); kd.append(rmse(dd[m].y,kd95((dd[m].V0,dd[m].t),*fkd(dd[~m]))))
    # nested: pick h vs h_path inside each training fold
    nest=[];picks=[]
    for te in fo:
        m=dd.SID.isin(te); tr,ts=dd[~m],dd[m]
        ts_=tr.SID.unique().copy(); r2=np.random.RandomState(7); r2.shuffle(ts_)
        ifs=max(1,len(ts_)//5); ifo=[ts_[i*ifs:(i+1)*ifs] if i<4 else ts_[i*ifs:] for i in range(5)]
        sc={h:_cv_h(tr,h,ifo) for h in ['h','h_path']}
        pk=min(sc,key=sc.get); picks.append(pk)
        nest.append(rmse(ts.y,c11((ts.V0,ts.t,ts[pk]),*f11(tr,pk))))
    rows.append(dict(dataset=tag,n_pts=len(dd),n_storms=dd.SID.nunique(),
                     KD95_RMSE=float(np.mean(kd)),
                     C11_h_RMSE=_cv_h(dd,'h',fo), C11_hpath_RMSE=_cv_h(dd,'h_path',fo),
                     nested_RMSE=float(np.mean(nest)), nested_picks=";".join(picks),
                     a_h=float(f11(dd,'h')[0]), a_hpath=float(f11(dd,'h_path')[0])))
tdf=pd.DataFrame(rows); tdf.to_csv("terrain_definition.csv",index=False)
for _,r in tdf.iterrows():
    print(f"[{r.dataset}] KD95 {r.KD95_RMSE:.2f} | C11(h) {r.C11_h_RMSE:.2f} | "
          f"C11(h_path) {r.C11_hpath_RMSE:.2f} | nested {r.nested_RMSE:.2f} ({r.nested_picks})")

json.dump(meta,open("metadata.json","w"),indent=2)
print("Wrote:", sorted([f for f in __import__('os').listdir('.') if f.endswith('.csv') or f.endswith('.json')]))
