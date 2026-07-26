import numpy as np, pandas as pd
from scipy.optimize import curve_fit
np.seterr(all='ignore')
def kd95(X,Vb,R,al): V0,t=X; return Vb+(R*V0-Vb)*np.exp(-al*t)
def c9(X,a): V0,t=X; return V0-a*V0**2*t
def c11(X,a): V0,t,h=X; return V0-a*V0*(V0*t+h)
def c15(X,a,b): V0,t,h,s=X; return ((V0*t)+h)*a*(s+V0+b)+V0
def c11b(X,a,b): V0,t,h=X; return V0*(1-a*V0*t-b*h)
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def r2(o,p): return float(1-np.sum((o-p)**2)/np.sum((o-np.mean(o))**2))
def build(df,native):
    df=df.copy(); df['ISO_TIME']=pd.to_datetime(df['ISO_TIME'])
    if native:
        df['hh']=df.ISO_TIME.dt.hour; df=df[df.hh.isin([0,6,12,18])].sort_values(['SID','ISO_TIME'])
        df['t']=(df.ISO_TIME-df.groupby('SID').ISO_TIME.transform('min')).dt.total_seconds()/3600
        df['V0']=df.groupby('SID').USA_WIND.transform('first')
        c=df.groupby('SID').SID.transform('size'); df=df[(c>=3)&(df.V0>=34)]
    else:
        df['t']=df['t_hours']
    df=df.dropna(subset=['h_max','h_mean','USA_WIND','V0','t','STORM_SPD'])
    df=df[(df.h_max>0)&(df.USA_WIND>0)]
    return df.rename(columns={'USA_WIND':'y','h_mean':'h','STORM_SPD':'spd'})[['SID','V0','t','y','h','spd']].reset_index(drop=True)
# ---- PATH (Jef's laptop, confirmed 25 Jul 2026) ----
# Switch to the 5-m circular rebuild once it exists:
#   r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\CSV\ph_decay_with_terrain_5m_circ.csv"
POINTS = r"D:\2026\ATTENUATE\OUTS\ph_decay_with_terrain.csv"
raw=pd.read_csv(POINTS)
for tag,native in [("NATIVE 6-hourly",True),("3-hourly",False)]:
    d=build(raw,native)
    pk,_=curve_fit(kd95,(d.V0,d.t),d.y,p0=[15,.9,.05],bounds=([0,.3,.001],[60,1,.5]),maxfev=20000)
    p9,_=curve_fit(c9,(d.V0,d.t),d.y,p0=[1.5e-4],bounds=([1e-5],[1e-3]),maxfev=20000)
    p11,_=curve_fit(c11,(d.V0,d.t,d.h),d.y,p0=[1.34e-4],bounds=([1e-5],[1e-3]),maxfev=20000)
    p15,_=curve_fit(c15,(d.V0,d.t,d.h,d.spd),d.y,p0=[-1.75e-4,-35],bounds=([-1e-2,-100],[0,0]),maxfev=20000)
    pb,_=curve_fit(c11b,(d.V0,d.t,d.h),d.y,p0=[1.3e-4,1.3e-4],bounds=([1e-6,1e-6],[1e-2,1e-2]),maxfev=40000)
    yk=kd95((d.V0,d.t),*pk); y9=c9((d.V0,d.t),*p9); y11=c11((d.V0,d.t,d.h),*p11); y15=c15((d.V0,d.t,d.h,d.spd),*p15)
    print(f"\n===== {tag} : N={len(d)} pts, {d.SID.nunique()} storms =====")
    print(f"KD95:  Vb={pk[0]:.2f} R={pk[1]:.3f} alpha={pk[2]:.4f}  inRMSE={rmse(d.y,yk):.2f}  R2={r2(d.y,yk):.3f}")
    print(f"C9 :   a={p9[0]:.3e}  inRMSE={rmse(d.y,y9):.2f}  MSE(loss)={np.mean((d.y-y9)**2):.1f}  R2={r2(d.y,y9):.3f}")
    print(f"C11:   a={p11[0]:.3e}  inRMSE={rmse(d.y,y11):.2f}  MSE(loss)={np.mean((d.y-y11)**2):.1f}  R2={r2(d.y,y11):.3f}")
    print(f"C11b:  a={pb[0]:.3e}/(kt hr)  b={pb[1]:.3e}/m  ratio={pb[0]/pb[1]:.2f}  Reff200={1-pb[1]*200:.3f} Reff1000={1-pb[1]*1000:.3f}")
    print(f"C15:   a={p15[0]:.3e} b={p15[1]:.2f}  inRMSE={rmse(d.y,y15):.2f}  MSE(loss)={np.mean((d.y-y15)**2):.1f}  R2={r2(d.y,y15):.3f}")
    print(f"C1(V0 only) MSE={np.mean((d.y-d.V0)**2):.1f}")
