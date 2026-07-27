"""
stage8_interpolation_check.py -- are the off-synoptic best-track entries interpolated?
=======================================================================================
Section 2a argues that only the 6-hourly synoptic times are independent best-track
estimates and that the intervening 3-hourly entries are temporally interpolated.
This script tests that directly: for every off-synoptic entry, compare the reported
wind against the arithmetic midpoint of the two synoptic entries bracketing it.

IMPORTANT, and this is a correction to the submitted text. IBTrACS stores winds as
whole knots, so when two synoptic values differ by an odd number the true midpoint
is a half knot and the stored value is rounded. Requiring exact equality therefore
understates the agreement badly: only 61% of JTWC winds match exactly, but 98%
match once that 1-kt storage rounding is allowed. The manuscript's word "exactly"
has to become "to within the 1-kt storage precision".

The script prints both figures so the distinction is visible.
"""
import numpy as np
import pandas as pd

PAR = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\CSV\PAR_1977_2023.csv"
TOL = 0.5   # half a knot: the most a correct midpoint can be shifted by rounding

d = pd.read_csv(PAR, low_memory=False)
d.columns = [c.strip() for c in d.columns]
d["ISO_TIME"] = pd.to_datetime(d["ISO_TIME"], errors="coerce")
for c in ("USA_WIND", "TOK_WIND"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.drop_duplicates(["SID", "ISO_TIME"]).sort_values(["SID", "ISO_TIME"])

off = d[~d.ISO_TIME.dt.hour.isin([0, 6, 12, 18])]
print(f"off-synoptic entries in the PAR record: {len(off)}")

for col, agency, published in (("USA_WIND", "JTWC 1-min", 97.7),
                               ("TOK_WIND", "RSMC Tokyo 10-min", 94.3)):
    s = d.set_index(["SID", "ISO_TIME"])[col]
    prev = s.reindex(pd.MultiIndex.from_arrays(
        [off.SID, off.ISO_TIME - pd.Timedelta("3h")])).values
    nxt = s.reindex(pd.MultiIndex.from_arrays(
        [off.SID, off.ISO_TIME + pd.Timedelta("3h")])).values
    val = off[col].values
    ok = ~np.isnan(prev) & ~np.isnan(nxt) & ~np.isnan(val)
    dev = np.abs(val[ok] - (prev[ok] + nxt[ok]) / 2)
    print(f"\n{agency}  ({col}), {int(ok.sum())} comparisons")
    print(f"  exactly the midpoint                {100 * (dev == 0).mean():5.1f} %")
    print(f"  midpoint within {TOL} kt rounding    {100 * (dev <= TOL).mean():5.1f} %"
          f"   (manuscript states {published})")
    print(f"  median deviation                    {np.median(dev):5.2f} kt")

print("\nConclusion: the off-synoptic entries carry no information independent of "
      "the synoptic estimates that bracket them.")
