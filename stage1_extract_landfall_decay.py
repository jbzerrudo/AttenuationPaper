"""
Stage 1: Extract Philippine Landfall TC Decay Segments
======================================================
Filters IBTrACS PAR data for storms making Philippine landfall,
extracts overland wind decay segments, and computes derived variables.

Input:  PAR_1977_2023.csv (IBTrACS extract for PAR domain)
Output: ph_landfall_decay.csv (one row per overland timestep per storm)

Author: Jef / Claude pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────
INPUT_CSV = r"D:\2026\SYNTC\ATTENUATE\DATA\PAR_1977_2023.csv"          # adjust path as needed
OUTPUT_CSV = r"D:\2026\SYNTC\ATTENUATE\OUTS\ph_landfall_decay.csv"

# Philippine bounding box
PH_LAT_MIN, PH_LAT_MAX = 4.5, 21.5
PH_LON_MIN, PH_LON_MAX = 116.0, 127.5

# Overland threshold: DIST2LAND (km) below which we consider the TC over land
# 30 km allows for small island gaps in the archipelago
DIST2LAND_THRESH = 30

# Minimum wind speed at landfall (kt) — TS threshold per Kaplan & DeMaria
MIN_V0 = 34

# Minimum number of post-landfall data points per storm
MIN_POINTS = 3

# Columns to keep from IBTrACS
KEEP_COLS = [
    'SID', 'NAME', 'SEASON',
    'ISO_TIME', 'LAT', 'LON',
    'USA_WIND',
    'LANDFALL', 'DIST2LAND',
    'STORM_SPD', 'STORM_DR',
    'USA_RMW',
]


def load_and_filter(path):
    """Load CSV, keep only needed columns, filter to PH bounding box."""
    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False, usecols=KEEP_COLS)
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])

    # Replace sentinel values with NaN
    df.replace({-999.9: np.nan, ' ': np.nan}, inplace=True)

    # Filter to Philippine bounding box
    ph = df[
        (df['LAT'] >= PH_LAT_MIN) & (df['LAT'] <= PH_LAT_MAX) &
        (df['LON'] >= PH_LON_MIN) & (df['LON'] <= PH_LON_MAX)
    ].copy()

    print(f"  Total rows: {len(df)}")
    print(f"  Rows in PH box: {len(ph)}")
    print(f"  Unique storms in PH box: {ph['SID'].nunique()}")
    return ph


def extract_overland_segment(storm_df):
    """
    Given a single storm's rows (sorted by time, PH box only),
    find the first landfall and extract the overland decay segment.
    
    Returns a DataFrame of overland timesteps, or None if unusable.
    """
    storm = storm_df.sort_values('ISO_TIME').reset_index(drop=True)

    # Find first landfall row (LANDFALL == 0)
    lf_rows = storm[storm['LANDFALL'] == 0]
    if len(lf_rows) == 0:
        return None

    first_lf_idx = lf_rows.index[0]
    v0 = storm.loc[first_lf_idx, 'USA_WIND']

    # Must have valid wind at landfall, at least TS strength
    if pd.isna(v0) or v0 <= 0 or v0 < MIN_V0:
        return None

    # Track forward from landfall
    post = storm.loc[first_lf_idx:]

    # Collect overland rows: DIST2LAND <= threshold
    # Allow 1 consecutive water row (island gap), stop at 2+ water rows
    overland_indices = []
    water_count = 0
    for idx, row in post.iterrows():
        if row['DIST2LAND'] <= DIST2LAND_THRESH:
            overland_indices.append(idx)
            water_count = 0
        else:
            water_count += 1
            if water_count >= 2:
                break
            overland_indices.append(idx)  # single water gap (inter-island)

    if len(overland_indices) < 2:
        return None

    segment = storm.loc[overland_indices].copy()

    # Must have valid USA_WIND for all rows
    segment = segment[segment['USA_WIND'] > 0].copy()
    if len(segment) < MIN_POINTS:
        return None

    # ── Derived variables ──
    t0 = segment['ISO_TIME'].iloc[0]
    segment['t_hours'] = (segment['ISO_TIME'] - t0).dt.total_seconds() / 3600.0
    segment['V0'] = v0
    segment['V_norm'] = segment['USA_WIND'] / v0  # normalized decay (1.0 at landfall)

    return segment


def main():
    ph = load_and_filter(INPUT_CSV)

    # Get unique storms with at least one LANDFALL=0 and valid USA_WIND
    candidate_sids = ph[ph['LANDFALL'] == 0]['SID'].unique()
    print(f"\nStorms with LANDFALL=0 in PH box: {len(candidate_sids)}")

    # Extract decay segments
    segments = []
    skipped = 0
    for sid in candidate_sids:
        storm_df = ph[ph['SID'] == sid]
        seg = extract_overland_segment(storm_df)
        if seg is not None:
            segments.append(seg)
        else:
            skipped += 1

    print(f"  Usable decay segments: {len(segments)}")
    print(f"  Skipped (no data / below TS / too few points): {skipped}")

    # Combine all segments
    result = pd.concat(segments, ignore_index=True)

    # ── Summary statistics ──
    storms = result.groupby('SID').agg(
        NAME=('NAME', 'first'),
        SEASON=('SEASON', 'first'),
        V0=('V0', 'first'),
        n_points=('USA_WIND', 'count'),
        duration_h=('t_hours', 'max'),
        V_final=('USA_WIND', 'last'),
    )

    print(f"\n{'='*60}")
    print(f"STAGE 1 SUMMARY")
    print(f"{'='*60}")
    print(f"Total storms:        {len(storms)}")
    print(f"Total data points:   {len(result)}")
    print(f"Points per storm:    {storms.n_points.mean():.1f} (median {storms.n_points.median():.0f})")
    print(f"Duration (hours):    {storms.duration_h.mean():.1f} mean, {storms.duration_h.median():.0f} median")
    print(f"\nLandfall intensity distribution:")
    print(f"  TS  (34-63 kt):   {((storms.V0>=34)&(storms.V0<=63)).sum()}")
    print(f"  Cat 1-2 (64-95):  {((storms.V0>=64)&(storms.V0<=95)).sum()}")
    print(f"  Cat 3+ (>=96):    {(storms.V0>=96).sum()}")
    print(f"\nSeason range: {storms.SEASON.min()} - {storms.SEASON.max()}")

    # ── Output columns ──
    out_cols = [
        'SID', 'NAME', 'SEASON',
        'ISO_TIME', 'LAT', 'LON',
        'USA_WIND', 'V0', 'V_norm', 't_hours',
        'DIST2LAND', 'LANDFALL',
        'STORM_SPD', 'STORM_DR',
        'USA_RMW',
    ]
    result[out_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\nOutput saved to: {OUTPUT_CSV}")
    print(f"Ready for Stage 2 (DTM sampling).")


if __name__ == "__main__":
    main()
