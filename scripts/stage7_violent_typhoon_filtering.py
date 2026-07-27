"""
stage7_violent_typhoon_filtering.py -- the >=106 kt overland ceiling (Section 4c, Fig. 6)
==========================================================================================
Reproduces every number in Section 4c from the IBTrACS PAR extract.

The question: do violent-typhoon winds survive a crossing of the archipelago?
The RSMC Tokyo 10-minute field is used throughout, so no conversion between the
JTWC and RSMC Tokyo averaging periods is required.

Output: violent_typhoon_filtering.csv  (the >=106 kt points, for Fig. 6)
        plus a printed verification block against the published numbers.

Note on Fig. 6: panel (b) and every number below come from this script. Panel (a)
draws these points over a web basemap, so a redraw will not be byte-identical to
the archived PNG. The data layer is fully specified here.
"""
import os
import numpy as np
import pandas as pd

PAR = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\CSV\PAR_1977_2023.csv"
OUT = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\CSV\violent_typhoon_filtering.csv"

THRESH = 106.0   # JMA violent-typhoon level, 10-min sustained
NEAR_KM = 30.0   # "close to a coast"

d = pd.read_csv(PAR, low_memory=False)
d.columns = [c.strip() for c in d.columns]
d["ISO_TIME"] = pd.to_datetime(d["ISO_TIME"], errors="coerce")
for c in ("TOK_WIND", "USA_WIND", "DIST2LAND", "LAT", "LON"):
    d[c] = pd.to_numeric(d[c], errors="coerce")

hi = d[d.TOK_WIND >= THRESH].copy()
hi["synoptic"] = hi.ISO_TIME.dt.hour.isin([0, 6, 12, 18])
hi["overland"] = hi.DIST2LAND == 0
hi["near_coast"] = hi.DIST2LAND <= NEAR_KM
hi.sort_values(["SID", "ISO_TIME"]).to_csv(OUT, index=False)

ov = hi[hi.overland]
syn = hi[hi.synoptic]

print(f"season range                 {int(d.SEASON.min())}-{int(d.SEASON.max())}")
print(f">= {THRESH:.0f} kt (10-min) points    {len(hi):4d} / {hi.SID.nunique()} storms"
      f"        (paper 445 / 52)")
print(f"  centre over land           {len(ov):4d} points / {ov.SID.nunique()} storms"
      f"   (paper 2 / 2)")
print(f"  within {NEAR_KM:.0f} km of a coast     {int(hi.near_coast.sum()):4d}"
      f"                       (paper 7)")
print(f"  median longitude           {hi.LON.median():6.1f} E"
      f"                  (paper 128.7)")
print(f"  east of 126 E              {100 * (hi.LON > 126).mean():6.0f} %"
      f"                    (paper 70)")
print(f"  west of 120 E              {int((hi.LON < 120).sum()):4d}"
      f"                        (paper 0)")
print("\nthe overland points:")
print(ov[["SID", "NAME", "SEASON", "ISO_TIME", "LAT", "LON",
          "TOK_WIND", "DIST2LAND", "LANDFALL", "synoptic"]].to_string(index=False))

# Interpolation exposure. Section 2a treats off-synoptic entries as interpolated,
# so the subsection has to survive being restricted to synoptic times.
print(f"\nrestricted to synoptic times: {len(syn)} points / {syn.SID.nunique()} storms, "
      f"{int((syn.DIST2LAND == 0).sum())} overland, median lon {syn.LON.median():.1f} E, "
      f"{int((syn.LON < 120).sum())} west of 120 E")
for _, r in ov.iterrows():
    s = d[d.SID == r.SID].sort_values("ISO_TIME").set_index("ISO_TIME").TOK_WIND
    p, n = s.get(r.ISO_TIME - pd.Timedelta("3h")), s.get(r.ISO_TIME + pd.Timedelta("3h"))
    if pd.notna(p) and pd.notna(n):
        mid = (p + n) / 2
        tag = "IS the midpoint of its neighbours" if abs(r.TOK_WIND - mid) < 1e-9 \
            else "is not the midpoint"
        print(f"  {r.NAME} {r.ISO_TIME:%Y-%m-%d %H%M}Z: {r.TOK_WIND:.0f} kt, "
              f"neighbours {p:.0f}/{n:.0f}, midpoint {mid:.0f} -> {tag}")

print(f"\nwrote {OUT}")
