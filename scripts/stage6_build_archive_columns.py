"""
stage6_build_archive_columns.py -- add the reproducibility columns to the archived dataset
===========================================================================================
Section 2a of the manuscript states that elapsed time and landfall intensity are
re-referenced to the native 6-hourly subset, and that both quantities plus a flag
marking the retained records accompany the archived dataset. This script writes
them.

Input : ph_decay_with_terrain_5m_circ.csv   (output of stage2_resample_5m_circ.py)
Output: ph_decay_with_terrain_archive.csv   (same rows, three extra columns)

The three columns are:

  in_analysis   True on the 453 records that constitute the native 6-hourly
                primary dataset, False on the other 711. Selecting on this column
                reproduces the analysis subset exactly.

  t_used        Elapsed time in hours as the models see it: zero at the first
                record of each storm's overland segment that falls at a synoptic
                hour. This is NOT t_hours, which is anchored at the landfall
                record and is therefore 0 or 3 hr larger.

  V0_used       Landfall intensity as the models see it: the wind at that same
                anchor record. This is NOT the V0 column, which is the wind at
                the landfall record. The two differ for the 43 storms that made
                landfall at an off-synoptic hour.

Both t_used and V0_used are computed on the hour-filtered set BEFORE the removal
of offshore and no-data points, which is why eight storms have min(t_used) = 6
rather than 0. They are populated wherever they are defined and left blank on
off-synoptic records.

Run it after stage2_resample_5m_circ.py. Needs only pandas and numpy.
"""
import os
import numpy as np
import pandas as pd

CSVDIR = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\CSV"
POINTS = os.path.join(CSVDIR, "ph_decay_with_terrain_5m_circ.csv")
OUT    = os.path.join(CSVDIR, "ph_decay_with_terrain_archive.csv")

TERRAIN_COL = 'hmean_rad75'   # the predictor adopted in the manuscript

raw = pd.read_csv(POINTS)
raw['ISO_TIME'] = pd.to_datetime(raw['ISO_TIME'])

# ---- reproduce build(raw, native=True) from analysis_all.py -----------------
d = raw[raw.ISO_TIME.dt.hour.isin([0, 6, 12, 18])].sort_values(['SID', 'ISO_TIME']).copy()
d['t_used']  = (d.ISO_TIME - d.groupby('SID').ISO_TIME.transform('min')).dt.total_seconds() / 3600
d['V0_used'] = d.groupby('SID').USA_WIND.transform('first')
cnt = d.groupby('SID').SID.transform('size')
d = d[(cnt >= 3) & (d.V0_used >= 34)]

keep = d.dropna(subset=['h_max', TERRAIN_COL, 'USA_WIND', 'V0_used', 't_used', 'STORM_SPD'])
keep = keep[(keep.h_max > 0) & (keep.USA_WIND > 0)]

# ---- attach to the full record ---------------------------------------------
out = raw.merge(d[['SID', 'ISO_TIME', 't_used', 'V0_used']], on=['SID', 'ISO_TIME'], how='left')
key = set(zip(keep.SID, keep.ISO_TIME))
out['in_analysis'] = [(s, t) in key for s, t in zip(out.SID, out.ISO_TIME)]

out.to_csv(OUT, index=False)

# ---- verification ----------------------------------------------------------
n = int(out.in_analysis.sum())
s = out.loc[out.in_analysis, 'SID'].nunique()
sub = out[out.in_analysis]
print(f"in_analysis : {n} records / {s} storms      (expected 453 / 121)")
print(f"mean t_used : {sub.t_used.mean():.3f} hr           (expected 10.146)")
print(f"mean t_hours: {sub.t_hours.mean():.3f} hr           (expected 11.570)")
print(f"V0_used differs from V0 on {int((sub.V0_used != sub.V0).sum())} records "
      f"/ {sub.loc[sub.V0_used != sub.V0, 'SID'].nunique()} storms   (expected 182 / 43)")
print(f"storms with min(t_used) > 0: {int((sub.groupby('SID').t_used.min() > 0).sum())}"
      f"   (expected 8, all at t = 6 hr)")
print(f"\nwrote {OUT}   ({len(out)} rows, {len(out.columns)} columns)")
