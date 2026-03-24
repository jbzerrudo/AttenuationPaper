"""
Stage 2: Sample DTM Terrain Statistics Along TC Overland Tracks
===============================================================
For each post-landfall timestep in ph_landfall_decay.csv,
opens a window from the 20m Philippine DTM and computes:
  - h_point:  elevation directly under TC center
  - h_max:    max elevation within sampling radius
  - h_mean:   mean elevation within sampling radius
  - h_std:    std deviation of elevation (terrain roughness proxy)

Input:  ph_landfall_decay.csv  (from Stage 1)
        dtm_phil.tif           (20m Philippine DTM, EPSG:4326)
Output: ph_decay_with_terrain.csv

Author: Jef / Claude pipeline
"""

import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import sys
from pathlib import Path

# ──────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────
INPUT_CSV = r"D:\2026\SYNTC\ATTENUATE\OUTS\ph_landfall_decay.csv"
DTM_PATH = r"D:\2025\SYNTC\PreJUNE2025\GEVNEW\SOURCE\NEWSTORMS\DEM\dtm_phil.tif"
OUTPUT_CSV = r"D:\2026\SYNTC\ATTENUATE\OUTS\ph_decay_with_terrain.csv"

# Sampling radius in km — terrain statistics computed within this radius
# of each TC center position. 50 km captures the inner core interaction.
SAMPLE_RADIUS_KM = 50

# Approximate degrees per km at Philippine latitudes (~12°N)
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 108.0  # cos(12°) * 111


def diagnose_dtm(dtm_path):
    """Print DTM metadata for verification before processing."""
    print(f"Opening DTM: {dtm_path}")
    with rasterio.open(dtm_path) as src:
        print(f"  CRS:        {src.crs}")
        print(f"  Size:       {src.width} x {src.height} pixels")
        print(f"  Resolution: {src.res[0]:.6f}° x {src.res[1]:.6f}°")
        print(f"  Bounds:     W={src.bounds.left:.4f} E={src.bounds.right:.4f}")
        print(f"              S={src.bounds.bottom:.4f} N={src.bounds.top:.4f}")
        print(f"  Bands:      {src.count}")
        print(f"  Dtype:      {src.dtypes[0]}")
        print(f"  NoData:     {src.nodata}")

        res_m = src.res[0] * KM_PER_DEG_LON * 1000
        print(f"  ~Pixel size: {res_m:.1f} m")

        # Sanity check: sample a known location
        # Manila: ~14.5°N, 121.0°E — should be ~5-30m
        # Cordillera: ~16.5°N, 120.9°E — should be ~1000-2500m
        test_points = [
            ("Manila (coastal)", 14.55, 121.00, 0, 50),
            ("Cordillera (mountain)", 16.50, 120.90, 500, 2800),
            ("Sierra Madre (mountain)", 16.00, 122.10, 300, 2000),
        ]
        print("\n  Diagnostic point samples:")
        for name, lat, lon, expect_lo, expect_hi in test_points:
            try:
                row, col = src.index(lon, lat)
                # Read a small 5x5 window around the point
                window = rasterio.windows.Window(col - 2, row - 2, 5, 5)
                data = src.read(1, window=window)
                val = data[2, 2]  # center pixel
                mean_val = np.nanmean(data[data != src.nodata]) if src.nodata is not None else np.nanmean(data)
                status = "OK" if expect_lo <= val <= expect_hi else "CHECK!"
                print(f"    {name}: {val:.0f} m (mean 5x5: {mean_val:.0f} m) [{status}]")
            except Exception as e:
                print(f"    {name}: FAILED - {e}")

    print()


def sample_terrain(src, lat, lon, radius_km):
    """
    Sample terrain statistics within radius_km of (lat, lon).
    Uses windowed read to avoid loading full 18GB raster.

    Returns dict with h_point, h_max, h_mean, h_std, or NaNs on failure.
    """
    result = {'h_point': np.nan, 'h_max': np.nan, 'h_mean': np.nan, 'h_std': np.nan}

    # Convert radius to degrees
    dlat = radius_km / KM_PER_DEG_LAT
    dlon = radius_km / KM_PER_DEG_LON

    # Bounding box for the window
    west = lon - dlon
    east = lon + dlon
    south = lat - dlat
    north = lat + dlat

    # Clip to DTM bounds
    west = max(west, src.bounds.left)
    east = min(east, src.bounds.right)
    south = max(south, src.bounds.bottom)
    north = min(north, src.bounds.top)

    if west >= east or south >= north:
        return result

    try:
        window = from_bounds(west, south, east, north, src.transform)

        # Clamp window to valid raster extent
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

        if window.width < 1 or window.height < 1:
            return result

        data = src.read(1, window=window).astype(float)

        # Mask nodata
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

        # Also mask unreasonable values (< -100 m or > 3500 m for Philippines)
        data[(data < -100) | (data > 3500)] = np.nan

        valid = data[~np.isnan(data)]
        if len(valid) == 0:
            return result

        # Point value: center pixel
        cy, cx = data.shape[0] // 2, data.shape[1] // 2
        h_point = data[cy, cx]
        if np.isnan(h_point):
            # Fall back to nearest valid pixel
            h_point = valid[0] if len(valid) > 0 else np.nan

        result['h_point'] = h_point
        result['h_max'] = np.nanmax(valid)
        result['h_mean'] = np.nanmean(valid)
        result['h_std'] = np.nanstd(valid)

    except Exception as e:
        print(f"    Warning: sampling failed at ({lat:.2f}, {lon:.2f}): {e}")

    return result


def main():
    # ── Diagnose DTM first ──
    diagnose_dtm(DTM_PATH)

    # ── Load Stage 1 output ──
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")
    print(f"Storms: {df['SID'].nunique()}")

    # ── Sample terrain ──
    print(f"\nSampling terrain (radius={SAMPLE_RADIUS_KM} km) ...")
    terrain_cols = ['h_point', 'h_max', 'h_mean', 'h_std']
    for col in terrain_cols:
        df[col] = np.nan

    with rasterio.open(DTM_PATH) as src:
        total = len(df)
        for i, (idx, row) in enumerate(df.iterrows()):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Processing row {i+1}/{total} ...")

            terrain = sample_terrain(src, row['LAT'], row['LON'], SAMPLE_RADIUS_KM)
            for col in terrain_cols:
                df.at[idx, col] = terrain[col]

    # ── Report ──
    valid = df.dropna(subset=['h_mean'])
    print(f"\n{'='*60}")
    print(f"STAGE 2 SUMMARY")
    print(f"{'='*60}")
    print(f"Rows with valid terrain data: {len(valid)} / {len(df)}")
    print(f"\nTerrain statistics across all post-landfall points:")
    for col in terrain_cols:
        vals = df[col].dropna()
        print(f"  {col:10s}: mean={vals.mean():.0f}  std={vals.std():.0f}  "
              f"min={vals.min():.0f}  max={vals.max():.0f}")

    # ── Save ──
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nOutput saved to: {OUTPUT_CSV}")
    print(f"Ready for Stage 3 (Kaplan & DeMaria baseline fit).")


if __name__ == "__main__":
    main()
