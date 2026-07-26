"""
terrain_equiv_subset.py  (Run B)
================================
Draws a random sample of points from across the archipelago and computes the
50-km mean elevation at 5 m and at 20 m ONLY, for the 5 m vs 20 m equivalence
(TOST) test. Same circular mask and NoData handling as the main sampler.

This one does read native 5 m, so it is slower per point, but only on ~150
random points. Run it after Run A (or in the background).
"""
import math, time
import numpy as np, pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

# ============ EDIT THESE ============
VRT    = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\DTM\Philippines_DTM_wgs84.vrt"
POINTS = r"D:\2026\ATTENUATE\OUTS\ph_decay_with_terrain.csv"   # needs LAT, LON, SID, ISO_TIME
OUT    = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\terrain_equiv_5m20m.csv"

NODATA       = -10000.0
NATIVE_RES_M = 5.0
RADIUS_KM    = 50.0
N_SUBSET     = 150     # random points; 100-200 is plenty for the equivalence test
SEED         = 7       # reproducible sample
# ====================================


def sample(src, lat, lon, radius_km, target_res_m):
    coslat = math.cos(math.radians(lat))
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * coslat)
    w, e, s, n = lon - dlon, lon + dlon, lat - dlat, lat + dlat
    b = src.bounds
    w, e = max(w, b.left), min(e, b.right)
    s, n = max(s, b.bottom), min(n, b.top)
    if w >= e or s >= n:
        return (np.nan, np.nan)
    win = from_bounds(w, s, e, n, src.transform)
    dec = max(1.0, target_res_m / NATIVE_RES_M)
    oh = max(1, int(round(win.height / dec)))
    ow = max(1, int(round(win.width / dec)))
    data = src.read(1, window=win, out_shape=(oh, ow),
                    resampling=Resampling.average).astype("float32")
    data[data == NODATA] = np.nan
    data[(data < -100) | (data > 3500)] = np.nan
    ys = np.linspace(n, s, oh); xs = np.linspace(w, e, ow)
    XX, YY = np.meshgrid(xs, ys)
    dist = np.sqrt(((YY - lat) * 111.0) ** 2 + ((XX - lon) * 111.0 * coslat) ** 2)
    vals = data[(dist <= radius_km) & ~np.isnan(data)]
    if vals.size == 0:
        return (np.nan, np.nan)
    return (float(vals.mean()), float(vals.max()))


def main():
    df = pd.read_csv(POINTS)
    df = df.sample(n=min(N_SUBSET, len(df)), random_state=SEED).reset_index(drop=True)
    print(f"Equivalence subset: {len(df)} random points (seed {SEED})")
    rows = []; t0 = time.time()
    with rasterio.open(VRT) as src:
        for i, r in df.iterrows():
            lat, lon = float(r["LAT"]), float(r["LON"])
            m5, x5 = sample(src, lat, lon, RADIUS_KM, 5.0)
            m20, _ = sample(src, lat, lon, RADIUS_KM, 20.0)
            rows.append({"SID": r["SID"], "ISO_TIME": r["ISO_TIME"], "LAT": lat, "LON": lon,
                         "hmean_5m": m5, "hmean_20m": m20, "hmax_50": x5})
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  {i+1}/{len(df)}  ({(time.time()-t0)/(i+1):.1f}s/pt)")
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    terr = out[out.hmax_50 > 0]
    d = (terr.hmean_5m - terr.hmean_20m)
    print(f"\nSaved {len(out)} rows to {OUT}")
    print(f"Terrain points: {len(terr)} | mean(5m-20m) diff = {d.mean():+.3f} m, max|diff| = {d.abs().max():.3f} m")


if __name__ == "__main__":
    main()
