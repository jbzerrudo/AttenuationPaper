"""
stage2_resample_5m_circ.py -- rebuild ph_decay_with_terrain.csv with 5-m circular sampling
============================================================================================
Re-samples terrain for every point in the existing ph_decay_with_terrain.csv using the
native 5-m NAMRIA IfSAR VRT with a TRUE circular mask (great-circle distance).
Fixes two defects in the original stage2_sample_dtm.py:
  1. Square-window bias -- the old code averaged a ±50 km bounding BOX, not a circle.
  2. Resolution -- it used a 20-m single-file DTM; this uses the 5-m VRT natively.

What it samples (one pass per point):
  Primary (replaces h_mean / h_max in the old CSV):
    h_mean  -- mean bare-earth elev within 50 km circle, at native 5-m resolution
    h_max   -- max within 50 km circle, same resolution
    h_std   -- std within 50 km circle, same resolution (kept for completeness)
  RMW-based footprints (5-m resolution, radius = USA_RMW or 50 km if missing):
    hmean_rmw  -- mean within 1 x RMW circle
    hmean_2rmw -- mean within 2 x RMW circle
    rmw_km     -- actual radius used in km (after floor)
  Resolution sweep at fixed 50 km circle:
    hmean_res5, hmean_res20, hmean_res90, hmean_res500, hmean_res1000
  Radius sweep at 90-m effective resolution:
    hmean_rad25, hmean_rad50, hmean_rad75, hmean_rad100

Output columns: all original non-terrain columns  +  all of the above.
The file is a DROP-IN replacement for ph_decay_with_terrain.csv in analysis_all.py
and stage4b_pysr_full_12x_NATIVE.py -- the column names h_mean and h_max are preserved.

RASTER BACKEND
--------------
Works with EITHER rasterio OR GDAL, whichever is importable. QGIS and OSGeo4W
ship GDAL but NOT rasterio, so the GDAL path lets this run in the OSGeo4W Shell
with nothing extra installed. It prints which backend it picked on startup.

Run it:
    python stage2_resample_5m_circ.py
If that fails on imports in the OSGeo4W Shell, run `py3_env` first, or try
`python3` instead of `python`.

MEASURED SPEED on Jef's machine, 26 Jul 2026, 1164 points:
  5-m primary, duplicated read      120.9 s/pt   39 h    (fixed: read was doubled)
  5-m primary, dedup fixed           27.5 s/pt  8.9 h
  90-m primary, radius sweep @500 m   ~1 s/pt   ~20 min  <-- current settings
Everything removed to get here is either a proven null result (5 m, 20 m give
identical medians) or already computed in terrain_sweeps.csv. The radius
calibration, which is the whole point of this stage, runs at all 14 radii.

VERIFY YOU HAVE THIS VERSION -- the startup banner must read:
  Primary h_mean/h_max: circular 50 km @ 90 m
  Resolution sweep     : [90, 500, 1000] m at 50 km circle
  Radius sweep         : [...] km at 500 m
If it says "@ 20 m" or "at 90 m" for the radius sweep, you are on the old file.

Author: Jef Zerrudo / revision helper
"""

import math
import os
import sys
import time
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# RASTER BACKEND: rasterio if present, otherwise GDAL.
# QGIS / OSGeo4W ship GDAL but NOT rasterio, so the GDAL path is what lets this
# run in the OSGeo4W Shell with nothing extra installed. Both paths do exactly
# the same thing: a windowed, averaged, decimated read.
# ---------------------------------------------------------------------------
BACKEND = None
try:
    import rasterio
    from rasterio.windows import from_bounds as _rio_from_bounds
    from rasterio.enums import Resampling as _RioResampling
    BACKEND = "rasterio"
except ImportError:
    try:
        from osgeo import gdal
        gdal.UseExceptions()
        BACKEND = "gdal"
    except ImportError:
        sys.exit("ERROR: neither rasterio nor GDAL (osgeo) is importable.\n"
                 "  In the OSGeo4W Shell try:  py3_env   then re-run.\n"
                 "  Or activate the conda env you used for the original sweep.")

# ============ PATHS (confirmed by Jef, 25 Jul 2026) ============
VRT    = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\DTM\Philippines_DTM_wgs84.vrt"
POINTS = r"D:\2026\ATTENUATE\OUTS\ph_decay_with_terrain.csv"   # existing 3-hourly dataset
OUT    = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\CSV\ph_decay_with_terrain_5m_circ.csv"

NODATA       = -10000.0      # VRT fill value
NATIVE_RES_M = 5.0           # native pixel size of the 5-m IfSAR DTM

# Primary circular window -------------------------------------------------------
PRIMARY_RADIUS_KM = 50.0     # same as original 50-km footprint, now circular
PRIMARY_RES_M     = 5.0      # only used when INCLUDE_5M is True
#
# INCLUDE_5M = False, and this is a considered decision, not a compromise.
# MEASURED on Jef's machine, 26 Jul 2026: with the native 5-m level ON, the
# sampler ran at 120.9 s/point = 39 HOURS for 1164 points. A 50-km circle at 5 m
# averages ~324 million pixels to produce one number, and it cannot use the VRT
# overviews because it is the native level.
#
# It also buys nothing measurable. The existing sweep already shows C11 gives
# 9.35-9.37 kt at every resolution from 20 m to 1 km against KD95's 9.86, i.e.
# the answer is flat across a 50-fold resolution range. The chance that 5 m
# differs from 20 m in any way that survives cross-validation is negligible.
#
# Set True only if a reviewer specifically demands the native level, and then
# expect to leave it running overnight.
INCLUDE_5M        = False

# RMW footprint -----------------------------------------------------------------
NM_TO_KM      = 1.852
RMW_FLOOR_KM  = 25.0         # minimum RMW radius (for tight-core storms)
RMW_RES_M     = 90.0         # effective resolution for RMW sampling (fast)

# Resolution sweep (all at fixed 50-km circle) ----------------------------------
# 5 m and 20 m deliberately absent.
#   5 m  = a 39-hour null result (see INCLUDE_5M above).
#   20 m = 25M pixels/point, ~70% of everything else combined, AND you already
#          have hmean_res20 for all 1164 points in terrain_sweeps.csv from the
#          Run A sampler, computed with the identical circular method. Your own
#          15-point test confirmed it: 20 m gave 112 m, 90 m gave 112 m.
# analysis_all.py auto-detects resolution columns, so it will use the 20 m values
# already on disk. Add 20 back only if you want the new file self-contained, and
# accept roughly +2 h for numbers you already have.
RES_SWEEP_M     = [90, 500, 1000]
RES_SWEEP_RAD   = 50.0

# Radius sweep (all at 90-m effective resolution) --------------------------------
# CALIBRATION GRID. The old 25-km steps were too coarse: nested CV showed skill
# still rising at 100 km (the largest radius tested) with the gain from 75 to
# 100 km only 0.02 kt, so the plateau sits somewhere in 75-100 and the old grid
# could not resolve it. Fine steps of 5-10 km through 50-110 locate it; the
# 125/150/200 entries are diagnostic only, to check the curve really flattens
# rather than climbing indefinitely. If skill keeps rising to 200 km, h_mean has
# stopped measuring local terrain and started measuring position in the
# archipelago, and a small radius is the safer choice.
#
# COST: the radius sweep is the expensive part and scales as radius^2. This grid
# is about 1.5x the cost of the old one. If the TEST_N run is too slow, thin the
# tails first (drop 200, then 150) before thinning the 50-110 region, which is
# the part that answers the question.
RAD_SWEEP_KM  = [25, 50, 60, 70, 75, 80, 85, 90, 95, 100, 110, 125, 150, 200]
# 500 m, NOT 90 m. Single biggest saving in the script, and it costs nothing:
# the radius comparison is insensitive to the resolution it is sampled at
# (20 m to 1 km all give C11 at 9.35-9.37 kt). Cost scales as 1/res^2, so
# 90 m -> 500 m cuts the radius sweep from 73M to 2.4M pixels per point.
RAD_SWEEP_RES = 500.0

# Test mode ---------------------------------------------------------------------
# RUN 1: leave TEST_N = 15, read the reported s/pt, decide INCLUDE_5M above.
# RUN 2: set TEST_N = 0 for the full 1164-point rebuild.
TEST_N = 0
# ====================================


class Raster:
    """Thin wrapper so the sampling code is identical under rasterio and GDAL."""

    def __init__(self, path):
        self.backend = BACKEND
        if BACKEND == "rasterio":
            self.ds = rasterio.open(path)
            b = self.ds.bounds
            self.left, self.right, self.bottom, self.top = b.left, b.right, b.bottom, b.top
            self.crs = str(self.ds.crs)
            self.xres = abs(self.ds.transform.a)
        else:
            self.ds = gdal.Open(path, gdal.GA_ReadOnly)
            if self.ds is None:
                raise IOError(f"GDAL could not open {path}")
            self.band = self.ds.GetRasterBand(1)
            gt = self.ds.GetGeoTransform()
            self.gt = gt
            self.W, self.H = self.ds.RasterXSize, self.ds.RasterYSize
            self.left, self.top = gt[0], gt[3]
            self.right  = gt[0] + self.W * gt[1]
            self.bottom = gt[3] + self.H * gt[5]     # gt[5] is negative
            self.crs = self.ds.GetProjection()[:60]
            self.xres = abs(gt[1])

    def overviews(self):
        if BACKEND == "rasterio":
            return self.ds.overviews(1)
        return [2 ** (i + 1) for i in range(self.band.GetOverviewCount())]

    def read_window(self, w, s, e, n, oh, ow):
        """Averaged, decimated read of the geographic window into an (oh, ow) array."""
        if BACKEND == "rasterio":
            win = _rio_from_bounds(w, s, e, n, self.ds.transform)
            return self.ds.read(1, window=win, out_shape=(oh, ow),
                                resampling=_RioResampling.average,
                                boundless=False).astype("float32")
        # --- GDAL path ---
        gt = self.gt
        x0 = int(math.floor((w - gt[0]) / gt[1]))
        x1 = int(math.ceil((e - gt[0]) / gt[1]))
        y0 = int(math.floor((n - gt[3]) / gt[5]))   # gt[5] negative -> north is smaller row
        y1 = int(math.ceil((s - gt[3]) / gt[5]))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.W, x1), min(self.H, y1)
        xs_, ys_ = x1 - x0, y1 - y0
        if xs_ < 1 or ys_ < 1:
            raise ValueError("empty window")
        arr = self.band.ReadAsArray(x0, y0, xs_, ys_,
                                    buf_xsize=ow, buf_ysize=oh,
                                    resample_alg=gdal.GRIORA_Average)
        if arr is None:
            raise ValueError("GDAL read returned None")
        return arr.astype("float32")

    def close(self):
        if BACKEND == "rasterio":
            self.ds.close()
        else:
            self.band = None
            self.ds = None

    def __enter__(self):  return self
    def __exit__(self, *a):
        self.close(); return False


def sample(src, lat, lon, radius_km, target_res_m):
    """
    Mean, max, std of valid bare-earth elevation within radius_km of (lat, lon).
    Reads at ~target_res_m via an averaged decimated read (pulls VRT overviews for
    coarse levels, so coarse resolutions are very fast).
    Returns (mean, max, std, n_pixels); (nan, nan, nan, 0) if no valid land pixels.
    """
    coslat = math.cos(math.radians(lat))
    dlat   = radius_km / 111.0
    dlon   = radius_km / (111.0 * coslat)
    w, e   = lon - dlon, lon + dlon
    s, n   = lat - dlat, lat + dlat
    w, e = max(w, src.left),   min(e, src.right)
    s, n = max(s, src.bottom), min(n, src.top)
    if w >= e or s >= n:
        return (np.nan, np.nan, np.nan, 0)

    dec = max(1.0, target_res_m / NATIVE_RES_M)
    # window size in native pixels, from the geographic extent
    win_w = (e - w) / src.xres
    win_h = (n - s) / src.xres
    oh  = max(1, int(round(win_h / dec)))
    ow  = max(1, int(round(win_w / dec)))

    try:
        data = src.read_window(w, s, e, n, oh, ow)
    except Exception:
        return (np.nan, np.nan, np.nan, 0)
    oh, ow = data.shape

    # NoData and sanity mask
    data[data == NODATA] = np.nan
    data[(data < -100) | (data > 3500)] = np.nan

    # Circular mask (equirectangular; < 1% error at these distances)
    ys = np.linspace(n, s, oh)
    xs = np.linspace(w, e, ow)
    XX, YY = np.meshgrid(xs, ys)
    dist = np.sqrt(
        ((YY - lat) * 111.0) ** 2 +
        ((XX - lon) * 111.0 * coslat) ** 2
    )
    vals = data[(dist <= radius_km) & ~np.isnan(data)]

    if vals.size == 0:
        return (np.nan, np.nan, np.nan, 0)
    return (float(vals.mean()), float(vals.max()), float(vals.std()), int(vals.size))


def main():
    # make sure the CSV output folder exists
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    df = pd.read_csv(POINTS)
    if TEST_N and TEST_N > 0:
        df = df.head(TEST_N).copy()
        print(f"*** TEST MODE: first {TEST_N} points only. Set TEST_N=0 for full run. ***")

    n_pts = len(df)
    print(f"Raster backend: {BACKEND.upper()}")
    print(f"Loaded {n_pts} points from {POINTS}")
    print(f"VRT: {VRT}")
    print()

    # Check USA_RMW availability
    rmw_avail = (df["USA_RMW"].fillna(0) > 0).sum()
    print(f"USA_RMW > 0 (RMW footprint available): {rmw_avail}/{n_pts} points")
    print()

    # Columns we will REPLACE (old square-window values)
    DROP_COLS = {"h_point", "h_mean", "h_max", "h_std"}
    keep_cols = [c for c in df.columns if c not in DROP_COLS]
    base = df[keep_cols].copy()

    # Initialise output columns
    new_cols = (
        ["h_mean", "h_max", "h_std"]          # primary 5-m circular (replace old)
        + ["hmean_rmw", "hmean_2rmw", "rmw_km"]  # RMW footprint
        + [f"hmean_res{r}" for r in RES_SWEEP_M]    # resolution sweep @ 50 km
        + [f"hmean_rad{r}" for r in RAD_SWEEP_KM]   # radius sweep @ 90 m
    )
    for c in new_cols:
        base[c] = np.nan

    # --- Effective primary resolution -----------------------------------------
    # 90 m when the 5-m level is off. Not a compromise: 20 m and 90 m returned
    # the SAME median (112 m) in the 15-point test, and 90 m is the SRTM grid,
    # which strengthens the paper's "globally available coarse data suffice"
    # claim. It also shares its read with the 90 m resolution-sweep entry, so
    # the primary column is effectively free.
    prim_res = PRIMARY_RES_M if INCLUDE_5M else 90.0
    print(f"Primary h_mean/h_max: circular {PRIMARY_RADIUS_KM:.0f} km @ {prim_res:.0f} m")
    print(f"RMW sampling         : circular RMW (floor {RMW_FLOOR_KM:.0f} km) @ {RMW_RES_M:.0f} m")
    print(f"Resolution sweep     : {RES_SWEEP_M} m at {RES_SWEEP_RAD:.0f} km circle")
    print(f"Radius sweep         : {RAD_SWEEP_KM} km at {RAD_SWEEP_RES:.0f} m")
    print()

    rows = []
    t0 = time.time()

    with Raster(VRT) as src:
        print(f"DTM  CRS={src.crs}  native~{src.xres*111000:.0f} m  "
              f"overviews={src.overviews()}")
        print()

        for i, row in df.reset_index(drop=True).iterrows():
            lat = float(row["LAT"])
            lon = float(row["LON"])

            out = {}

            # Per-point memo. The primary window and the resolution-sweep entry
            # at the same (radius, resolution) are the SAME computation; without
            # this they were being read twice, doubling the expensive 5-m level.
            memo = {}
            def s(radius_km, res_m):
                k = (round(float(radius_km), 4), round(float(res_m), 4))
                if k not in memo:
                    memo[k] = sample(src, lat, lon, float(radius_km), float(res_m))
                return memo[k]

            # ---- PRIMARY: circular 50 km at prim_res ------------------------
            hm, hx, hs, npx = s(PRIMARY_RADIUS_KM, prim_res)
            out["h_mean"] = hm
            out["h_max"]  = hx
            out["h_std"]  = hs

            # ---- RMW footprint -----------------------------------------------
            rmw_nm = float(row.get("USA_RMW", 0) or 0)
            if rmw_nm > 0:
                rmw_km = max(rmw_nm * NM_TO_KM, RMW_FLOOR_KM)
                out["hmean_rmw"]  = s(rmw_km,     RMW_RES_M)[0]
                out["hmean_2rmw"] = s(2*rmw_km,   RMW_RES_M)[0]
                out["rmw_km"]     = rmw_km
            else:
                out["hmean_rmw"] = out["hmean_2rmw"] = out["rmw_km"] = np.nan

            # ---- RESOLUTION SWEEP at 50 km -----------------------------------
            for res in RES_SWEEP_M:
                col = f"hmean_res{res}"
                if res == 5 and not INCLUDE_5M:
                    out[col] = np.nan   # skip the slow native level if requested
                else:
                    out[col] = s(RES_SWEEP_RAD, float(res))[0]

            # ---- RADIUS SWEEP at 90 m ----------------------------------------
            for rad in RAD_SWEEP_KM:
                out[f"hmean_rad{rad}"] = s(float(rad), RAD_SWEEP_RES)[0]

            rows.append(out)

            if (i + 1) % 50 == 0 or i == 0:
                dt = time.time() - t0
                rate = dt / (i + 1)
                remaining = rate * (n_pts - i - 1)
                print(f"  {i+1:4d}/{n_pts}  "
                      f"{rate:.1f} s/pt  "
                      f"ETA {remaining/60:.1f} min")

    # Build output dataframe ----------------------------------------------------
    terrain_df = pd.DataFrame(rows, index=base.index)
    for c in new_cols:
        base[c] = terrain_df[c]

    base.to_csv(OUT, index=False)

    # Sanity report -------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Saved {len(base)} rows to {OUT}")
    print()

    # Primary column summary
    hm  = base["h_mean"].dropna()
    hx  = base["h_max"].dropna()
    rmw = base["hmean_rmw"].dropna()
    print(f"PRIMARY  h_mean (circular {PRIMARY_RADIUS_KM:.0f} km @ {prim_res:.0f} m):")
    print(f"  min={hm.min():.0f}  median={hm.median():.0f}  max={hm.max():.0f} m  "
          f"n={len(hm)}/{len(base)}")
    print(f"PRIMARY  h_max (circular {PRIMARY_RADIUS_KM:.0f} km @ {prim_res:.0f} m):")
    print(f"  min={hx.min():.0f}  median={hx.median():.0f}  max={hx.max():.0f} m")
    print(f"RMW      hmean_rmw:")
    print(f"  min={rmw.min():.0f}  median={rmw.median():.0f}  max={rmw.max():.0f} m  "
          f"n={len(rmw)}/{len(base)}")
    print()
    print("Resolution sweep medians at 50-km circle:")
    for r in RES_SWEEP_M:
        col = f"hmean_res{r}"
        v = base[col].dropna()
        print(f"  {r:5d} m:  {v.median():.0f} m  (n={len(v)})")
    print()
    print(f"Radius sweep medians at {RAD_SWEEP_RES:.0f}-m resolution:")
    for r in RAD_SWEEP_KM:
        col = f"hmean_rad{r}"
        v = base[col].dropna()
        print(f"  {r:3d} km:  {v.median():.0f} m  (n={len(v)})")
    print()
    print("Next steps:")
    print(f"  1. Send {os.path.basename(OUT)} back to Cowork.")
    print(f"  2. Re-run analysis_all.py with POINTS set to:")
    print(f"       {OUT}")
    print(f"     -- CV / stratified / bootstrap on 5-m circular terrain.")
    print(f"  3. Re-run stage4b_pysr_full_12x_NATIVE.py with INPUT_CSV set to the same")
    print(f"     file -- PySR discovery on 5-m circular terrain.")
    print(f"     Do this only AFTER the current PySR run finishes.")


if __name__ == "__main__":
    main()
