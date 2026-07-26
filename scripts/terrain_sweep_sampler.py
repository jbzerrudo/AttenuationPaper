"""
terrain_sweep_sampler.py
========================
Samples the 5-m Philippine DTM VRT at every TC landfall point to produce the
resolution sweep, radius sweep, and RMW-scaled footprint for the WAF revision.

Fixes the original square-window bug: uses a TRUE circular mask (great-circle
distance) and respects NoData (-10000) so ocean is never averaged into h_mean.

Reads coarse levels from the region overviews (fast); native 5 m over a 50-km
footprint is the one slow level (see INCLUDE_5M).

Output: one CSV keyed by SID + ISO_TIME with h_mean for each sweep combination.
Run it in the OSGeo4W Shell / any env with rasterio + pandas + numpy.

Author: revision helper for J. Zerrudo
"""
import math, time
import numpy as np, pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

# ============ EDIT THESE ============
VRT    = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\DTM\Philippines_DTM_wgs84.vrt"
POINTS = r"D:\2026\ATTENUATE\OUTS\ph_decay_with_terrain.csv"   # needs LAT, LON, SID, ISO_TIME, USA_RMW
OUT    = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\CSV\terrain_sweeps.csv"

NODATA        = -10000.0
NATIVE_RES_M  = 5.0
TEST_N        = 0        # process only the first N points first to check sanity/timing; set 0 for ALL

# resolution sweep (fixed 50-km footprint) — effective resolutions in metres
RES_SWEEP_M   = [5, 20, 90, 500, 1000]
INCLUDE_5M    = False      # 5 m over 50 km is the SLOW level; set False to skip if timing is bad
RES_RADIUS_KM = 50.0

# radius sweep (fixed 90-m effective resolution) — radii in km
RADIUS_SWEEP_KM = [25, 50, 75, 100]
RADIUS_RES_M    = 90.0

# RMW-scaled footprint (fixed 90-m resolution); USA_RMW is in nautical miles
NM_TO_KM   = 1.852
RMW_FLOOR_KM = 25.0       # do not let tight-core storms collapse to a few pixels
DO_RMW     = True
# ====================================


def sample(src, lat, lon, radius_km, target_res_m):
    """Mean/max/std of bare-earth elevation within radius_km of (lat,lon),
    read at ~target_res_m, with a true circular mask and NoData handling."""
    coslat = math.cos(math.radians(lat))
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * coslat)
    w, e, s, n = lon - dlon, lon + dlon, lat - dlat, lat + dlat
    b = src.bounds
    w, e = max(w, b.left), min(e, b.right)
    s, n = max(s, b.bottom), min(n, b.top)
    if w >= e or s >= n:
        return (np.nan, np.nan, np.nan, 0)
    win = from_bounds(w, s, e, n, src.transform)
    dec = max(1.0, target_res_m / NATIVE_RES_M)
    oh = max(1, int(round(win.height / dec)))
    ow = max(1, int(round(win.width / dec)))
    # averaged, decimated read -> pulls from overviews for coarse levels (fast)
    data = src.read(1, window=win, out_shape=(oh, ow),
                    resampling=Resampling.average, boundless=False).astype("float32")
    data[data == NODATA] = np.nan
    data[(data < -100) | (data > 3500)] = np.nan     # kill NoData-contaminated coastal cells & spikes
    # circular mask on the read grid (equirectangular metres; <1% error at these scales)
    ys = np.linspace(n, s, oh); xs = np.linspace(w, e, ow)
    XX, YY = np.meshgrid(xs, ys)
    dist = np.sqrt(((YY - lat) * 111.0) ** 2 + ((XX - lon) * 111.0 * coslat) ** 2)
    vals = data[(dist <= radius_km) & ~np.isnan(data)]
    if vals.size == 0:
        return (np.nan, np.nan, np.nan, 0)
    return (float(vals.mean()), float(vals.max()), float(vals.std()), int(vals.size))


def main():
    df = pd.read_csv(POINTS)
    if TEST_N and TEST_N > 0:
        df = df.head(TEST_N).copy()
        print(f"*** TEST MODE: first {TEST_N} points. Set TEST_N=0 for the full run. ***")
    print(f"{len(df)} points | VRT: {VRT}")

    rows = []
    t0 = time.time()
    with rasterio.open(VRT) as src:
        for i, r in df.reset_index(drop=True).iterrows():
            lat, lon = float(r["LAT"]), float(r["LON"])
            out = {"SID": r["SID"], "ISO_TIME": r["ISO_TIME"], "LAT": lat, "LON": lon}
            # resolution sweep at fixed 50 km
            for res in RES_SWEEP_M:
                if res == 5 and not INCLUDE_5M:
                    out[f"hmean_res{res}"] = np.nan; continue
                hm, hx, hs, npx = sample(src, lat, lon, RES_RADIUS_KM, float(res))
                out[f"hmean_res{res}"] = hm
                if res == 90:  # keep max/std/count for the common anchor
                    out["hmax_r50"], out["hstd_r50"], out["npx_r50"] = hx, hs, npx
            # radius sweep at fixed 90 m
            for rad in RADIUS_SWEEP_KM:
                hm, hx, hs, npx = sample(src, lat, lon, float(rad), RADIUS_RES_M)
                out[f"hmean_rad{rad}"] = hm
            # RMW footprint
            if DO_RMW:
                rmw_nm = float(r.get("USA_RMW", 0) or 0)
                if rmw_nm > 0:
                    rmw_km = max(rmw_nm * NM_TO_KM, RMW_FLOOR_KM)
                    out["hmean_rmw"]  = sample(src, lat, lon, rmw_km, RADIUS_RES_M)[0]
                    out["hmean_2rmw"] = sample(src, lat, lon, 2 * rmw_km, RADIUS_RES_M)[0]
                    out["rmw_km"] = rmw_km
                else:
                    out["hmean_rmw"] = out["hmean_2rmw"] = out["rmw_km"] = np.nan
            rows.append(out)
            if (i + 1) % 25 == 0 or i == 0:
                dt = time.time() - t0
                print(f"  {i+1}/{len(df)}  ({dt/(i+1):.2f}s/point)")

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    print(f"\nSaved {len(res)} rows to {OUT}")
    # quick sanity: anchor 50 km / 90 m mean should be positive and plausible
    print("Sanity (hmean at 50 km, 90 m): "
          f"min={res['hmean_res90'].min():.0f}  median={res['hmean_res90'].median():.0f}  "
          f"max={res['hmean_res90'].max():.0f} m")


if __name__ == "__main__":
    main()
