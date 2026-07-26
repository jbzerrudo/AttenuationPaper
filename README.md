# How Much Do Mountains Matter? Terrain-Dependent Tropical Cyclone Wind Decay in the Philippines

Code, data and results supporting Zerrudo and Servando, submitted to *Weather and Forecasting* (WAF-D-26-0060).

The paper revisits the Kaplan and DeMaria (1995) inland wind decay model for the
Philippine archipelago and reports a terrain-aware alternative discovered by
symbolic regression:

```
V(t) = V0 * [1 - a * (V0 * t + h)]        a = 1.43e-4
```

where `h` is the mean bare-earth elevation within a 75 km circle centred on the
storm. On 453 native 6-hourly overland points from 121 storms (1977 to 2022),
storm-stratified five-fold cross-validation gives a mean RMSE of 9.22 kt against
9.86 kt for the constant-rate baseline.

## Reproducing the analysis

Run the scripts in this order. Each has its paths set at the top of the file.

| # | Script | Produces |
|---|--------|----------|
| 1 | `scripts/stage1_extract_landfall_decay.py` | overland decay segments from the IBTrACS PAR extract |
| 2 | `scripts/stage2_resample_5m_circ.py` | terrain sampled from the IfSAR DTM at every radius and resolution |
| 3 | `scripts/stage6_build_archive_columns.py` | the archive columns `in_analysis`, `t_used`, `V0_used` |
| 4 | `scripts/global_fits.py` | global fits, both cadences |
| 5 | `scripts/analysis_all.py` | cross-validation, stratified bias, bootstrap, sweeps |
| 6 | `scripts/stage5_bias_by_radius.py` | terrain-conditional bias by footprint radius, Table 5(b) |
| 7 | `scripts/stage4b_pysr_full_12x_NATIVE.py` | the twelve-seed symbolic regression search |

Steps 1 and 2 need the source data described below. Steps 3 to 6 need only
pandas, numpy and scipy and run in seconds to minutes. Step 7 needs PySR and
takes two to four hours.

Set `TEST_N = 0` in `stage2_resample_5m_circ.py` for a full run. It ships at 0,
but check it, because a value of 15 processes only the first fifteen points.

## Source data, not redistributed here

**IBTrACS**, release v04r01, downloaded 15 March 2026, subset to the Philippine
Area of Responsibility as `PAR_1977_2023.csv`. Available from NOAA NCEI at
https://www.ncei.noaa.gov/products/international-best-track-archive .
`USA_WIND` is the JTWC 1-minute sustained wind.

**NAMRIA IfSAR digital terrain model**, 5 m bare earth, mosaicked to a WGS84
virtual raster. Distributed by the National Mapping and Resource Information
Authority of the Philippines and not redistributable here.

## What is in `data/`

`ph_decay_with_terrain_5m_circ.csv` is the output of step 2: 1164 overland
records over 174 storms at the IBTrACS 3-hourly cadence, with terrain sampled at
three resolutions and fourteen footprint radii.

`ph_decay_with_terrain_archive.csv` is the same file plus the three columns the
analysis actually consumes. **Use this one.** The reason it exists is that the
modelling subset is not recoverable from the raw columns alone:

- `in_analysis` marks the 453 records of the native 6-hourly primary dataset.
  Filtering on it reproduces the analysis subset exactly.
- `t_used` is elapsed time as the models see it, zero at the first record of each
  storm's overland segment that falls at a synoptic hour. It is **not** `t_hours`,
  which is anchored at the landfall record and is 0 or 3 hr larger.
- `V0_used` is landfall intensity as the models see it, the wind at that same
  anchor record. It is **not** the `V0` column. The two differ on 182 of the 453
  records across the 43 storms that made landfall at an off-synoptic hour, by up
  to 15 kt.

Both `t_used` and `V0_used` are evaluated before the removal of offshore and
no-data points, which is why eight storms begin at `t_used` = 6 hr rather than 0.
Fitting with `t_hours` and `V0` in their place gives 10.45 kt and 9.88 kt instead
of the published 9.86 kt and 9.22 kt.

`fold_map.csv` gives the cross-validation fold of each of the 453 records, so the
storm-stratified split can be reproduced without rerunning the shuffle.

`oos_predictions.csv` gives the held-out prediction of both models at every
record, with the coefficients fitted in each fold, so Tables 2 and 3 can be
checked arithmetically without refitting anything.

## What is in `results/`

Cross-validation tables at both cadences, stratified RMSE and bias, the
dimensional two-coefficient fit, the resolution and radius sweeps, the nested
cross-validation over radius, the RMW footprint sensitivity test, the robustness
suite (storm-clustered bootstrap, Dvorak Monte Carlo, leave-one-storm-out,
temporal hold-out, machine-learning benchmarks), `bias_by_radius.csv` behind
Table 5(b), and the Pareto fronts from all twelve symbolic-regression seeds.

## A note on the terrain predictor

The reported analyses use `hmean_rad75`, a 75 km circle read at an effective
500 m spacing. Not 5 m, not 20 m, and not the 50 km `h_mean` column. The 5 m
level is present in the sampler but disabled: it costs 39 hours for the full file
and the resolution sweep shows the answer is flat to within 0.02 kt from 20 m to
1 km.

## Citation

See `CITATION.cff`. Please cite both the paper and the archived version of this
repository.

## License

Code is released under the MIT License. Derived data files are released under
CC BY 4.0. Neither IBTrACS nor the NAMRIA DTM is redistributed here; both remain
under the terms of their respective providers.
