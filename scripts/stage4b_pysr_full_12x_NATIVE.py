"""
Stage 4b (NATIVE 6-hourly): PySR — rediscover the decay equation on synoptic-only data
========================================================================================
Same as stage4b_pysr_full_12x.py, but the DISCOVERY search is run on the native
6-hourly synoptic best-track points only (0/6/12/18 UTC), with the interpolated
3-hourly entries removed. This is the interpolation-free re-search that answers
Reviewer 1 (Major 2) at the discovery level, not just the refit level.

Two things differ from the original stage4b:
  1. prepare_data() filters to synoptic times and recomputes t_hours and V0 from
     the native points (t = 0 at the first native overland synoptic time; V0 = the
     wind at that time). This matches the manuscript's native dataset exactly
     (459 points, 121 storms).
  2. The KD95 reference is set to the native global fit for a fair on-screen comparison.

ITERATION COUNT -- read this before changing it
-----------------------------------------------
niterations is 300, matching the original run and matching what the manuscript
already states. An earlier draft of this script used 2000 purely so the paper
could claim a bigger search. That is a cosmetic reason, not a scientific one, and
it costs ~12-24 h of wall clock for 12 seeds.

The evidence says 300 is enough: in the 25 July trial, seed 42 had already
surfaced C11 = V0 - a*V0*(V0*t + h_mean) with a = 1.44e-4 in its Hall of Fame at
15% of a 2000-iteration budget, i.e. around 300 iterations. The form is easy to
find; it does not need a deep search. Reporting that 12 independent seeds all
land on it after a modest search is a STRONGER claim than reporting that one
exhaustive search found it.

At 300 iterations expect roughly 10-20 min per seed, so about 2-4 h for all 12.

Input:  ph_decay_with_terrain_5m_circ.csv  (the 5-m circular rebuild from
        stage2_resample_5m_circ.py). The script does the native 6-hourly
        filtering itself, so point it at the full file.
Output: pysr_results_native_5m/run_YYYYMMDD_HHMMSS_seedNN/  (+ all_runs_summary.csv)

Author: Jef Zerrudo (I used Claude AI to optimize this code)
"""

import pandas as pd
import numpy as np
from pysr import PySRRegressor
from datetime import datetime
import warnings
import time
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────
# ---- PATHS (Jef's laptop, confirmed 25 Jul 2026) ----
# Default input is the 5-m circular rebuild from stage2_resample_5m_circ.py.
# To fall back to the old 20-m square dataset, use:
#   r"D:\2026\ATTENUATE\OUTS\ph_decay_with_terrain.csv"
INPUT_CSV   = r"D:\2026\ATTENUATE\RevisionsFolder\Dataset\CSV\ph_decay_with_terrain_5m_circ.csv"
OUTPUT_BASE = r"D:\2026\ATTENUATE\OUTS\pysr_results_native_5m"

# Number of seeds. 12 is the manuscript claim. If you are time-boxed, drop to 6:
# six independent seeds agreeing is still a defensible robustness statement, and
# the manuscript text just has to say six.
N_RUNS = 12

# ---- CALIBRATED TERRAIN FOOTPRINT ----------------------------------------
# Which column PySR sees as h_mean. The paper now reports a 75-km circular
# footprint (calibrated 26 Jul 2026). Discovery MUST use the same variable the
# paper validates on, or the equation is found on one footprint and reported on
# another. Set to None to keep the file's own 50-km h_mean.
TERRAIN_COL = 'hmean_rad75'

# KD95 reference, refit on THIS dataset (453 pts / 121 storms, circular 75 km).
# On-screen comparison only; the authoritative numbers come from analysis_all.py.
KD95_NATIVE = dict(Vb=39.13, R=1.0, alpha=0.0427)

PYSR_CONFIG = dict(
    niterations=300,           # see header. 300 matches the manuscript and the
                               # original run; C11 surfaces well inside this budget.
    populations=40,
    population_size=60,
    maxsize=25,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "sqrt", "abs"],
    extra_sympy_mappings={},
    loss="loss(prediction, target) = (prediction - target)^2",
    model_selection="best",
    temp_equation_file=True,
    verbosity=1,
    progress=True,
)


def prepare_data(df):
    # ---- SUBSTITUTE THE CALIBRATED TERRAIN COLUMN -------------------------
    df = df.copy()
    if TERRAIN_COL:
        if TERRAIN_COL in df.columns:
            print(f"Terrain feature: {TERRAIN_COL} "
                  f"({df[TERRAIN_COL].notna().sum()} non-null) -> used as h_mean")
            df['h_mean'] = df[TERRAIN_COL]
        else:
            raise SystemExit(
                f"ERROR: TERRAIN_COL '{TERRAIN_COL}' not found in {INPUT_CSV}.\n"
                f"  Available hmean_rad* columns: "
                f"{[c for c in df.columns if c.startswith('hmean_rad')]}\n"
                f"  Either point INPUT_CSV at the rebuilt file, or set TERRAIN_COL = None.")
    else:
        print("Terrain feature: h_mean as-is (50 km)")
    # -----------------------------------------------------------------------

    # ---- NATIVE 6-HOURLY FILTER (interpolation-free) ----
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
    df['hh'] = df['ISO_TIME'].dt.hour
    df = df[df['hh'].isin([0, 6, 12, 18])].sort_values(['SID', 'ISO_TIME'])
    # re-anchor time and landfall intensity to the native points only
    df['t_hours'] = (df['ISO_TIME'] - df.groupby('SID')['ISO_TIME'].transform('min')
                     ).dt.total_seconds() / 3600.0
    df['V0'] = df.groupby('SID')['USA_WIND'].transform('first')
    cnt = df.groupby('SID')['SID'].transform('size')
    df = df[(cnt >= 3) & (df['V0'] >= 34)]
    # -----------------------------------------------------

    feature_defs = {
        't_hours':    'Hours after landfall',
        'V0':         'Wind speed at landfall (kt)',
        'h_max':      'Max terrain height within 50km (m)',
        'h_mean':     'Mean terrain height, calibrated 75-km circular footprint (m)',
        'STORM_SPD':  'Translational speed (kt)',
    }

    available = [f for f in feature_defs if f in df.columns]
    clean = df.dropna(subset=available + ['USA_WIND']).copy()
    clean = clean[clean['h_max'] > 0].copy()
    clean = clean[clean['USA_WIND'] > 0].copy()

    X = clean[available]
    y = clean['USA_WIND']

    print(f"Features ({len(available)}):")
    for f in available:
        desc = feature_defs.get(f, '')
        vals = X[f]
        print(f"  {f:15s}: {desc}")
        print(f"    range [{vals.min():.1f}, {vals.max():.1f}], "
              f"mean={vals.mean():.1f}, std={vals.std():.1f}")

    print(f"\nTarget: USA_WIND (kt)")
    print(f"  range [{y.min():.0f}, {y.max():.0f}], "
          f"mean={y.mean():.1f}, std={y.std():.1f}")
    print(f"\nNATIVE 6-hourly data points: {len(clean)}  (expected 453)")
    print(f"Unique storms:               {clean['SID'].nunique()}  (expected 121)")
    if len(clean) != 453 or clean['SID'].nunique() != 121:
        print("  !! WARNING: does not match analysis_all.py. Check INPUT_CSV and TERRAIN_COL.")

    return X, y, clean


def evaluate_all_equations(model, X, y):
    equations = model.equations_
    if equations is None:
        return None

    ss_tot = np.sum((y.values - np.mean(y.values)) ** 2)
    r2_list, rmse_list, mae_list = [], [], []

    for i in range(len(equations)):
        try:
            y_pred = model.predict(X, index=i)
            residuals = y.values - y_pred
            ss_res = np.sum(residuals ** 2)
            r2 = 1 - ss_res / ss_tot
            rmse = np.sqrt(np.mean(residuals ** 2))
            mae = np.mean(np.abs(residuals))
        except Exception:
            r2, rmse, mae = np.nan, np.nan, np.nan
        r2_list.append(r2); rmse_list.append(rmse); mae_list.append(mae)

    equations['R2'] = r2_list
    equations['RMSE'] = rmse_list
    equations['MAE'] = mae_list
    return equations


def report_results(model, X, y, output_dir, run_id, elapsed_s):
    equations = evaluate_all_equations(model, X, y)

    # KD95 baseline = native global fit
    Vb_kd, R_kd, alpha_kd = KD95_NATIVE['Vb'], KD95_NATIVE['R'], KD95_NATIVE['alpha']
    V0 = X['V0'].values
    t = X['t_hours'].values
    y_pred_kd = Vb_kd + (R_kd * V0 - Vb_kd) * np.exp(-alpha_kd * t)
    ss_tot = np.sum((y.values - np.mean(y.values)) ** 2)
    res_kd = y.values - y_pred_kd
    r2_kd = 1 - np.sum(res_kd ** 2) / ss_tot
    rmse_kd = np.sqrt(np.mean(res_kd ** 2))
    mae_kd = np.mean(np.abs(res_kd))

    print(f"\n{'='*80}\nPySR RESULTS (NATIVE) — RUN {run_id}\n{'='*80}")
    if equations is not None:
        print(f"\n{'Idx':>3s}  {'Cplx':>4s}  {'Loss':>10s}  {'R2':>7s}  "
              f"{'RMSE':>8s}  {'MAE':>8s}  Equation")
        print(f"{'-'*3}  {'-'*4}  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*40}")
        for i, row in equations.iterrows():
            print(f"[{i:2d}]  {row['complexity']:4.0f}  {row['loss']:10.4f}  "
                  f"{row['R2']:7.4f}  {row['RMSE']:8.2f}  {row['MAE']:8.2f}  {row['equation']}")

    best_eq = model.sympy()
    y_pred_best = model.predict(X)
    res_best = y.values - y_pred_best
    r2_best = 1 - np.sum(res_best ** 2) / ss_tot
    rmse_best = np.sqrt(np.mean(res_best ** 2))
    mae_best = np.mean(np.abs(res_best))

    print(f"\n  Best: V(t) = {best_eq}")
    print(f"  PySR:  R2={r2_best:.4f}  RMSE={rmse_best:.2f} kt  MAE={mae_best:.2f} kt")
    print(f"  KD95:  R2={r2_kd:.4f}  RMSE={rmse_kd:.2f} kt  MAE={mae_kd:.2f} kt")
    print(f"  RMSE improvement: {(rmse_kd-rmse_best)/rmse_kd*100:.1f}%")
    print(f"  Elapsed: {elapsed_s:.1f} s")

    if equations is not None:
        equations.to_csv(f"{output_dir}/pareto_front.csv", index=False)
    with open(f"{output_dir}/pareto_front_metrics.txt", 'w') as f:
        f.write(f"PySR Pareto Front (NATIVE 6-hourly) | Run: {run_id}\n{'='*90}\n")
        f.write(f"N = {len(y)} data points | Elapsed: {elapsed_s:.1f} s\n")
        f.write(f"KD95 (native global): R2={r2_kd:.4f}  RMSE={rmse_kd:.2f}  MAE={mae_kd:.2f}\n\n")
        if equations is not None:
            for i, row in equations.iterrows():
                f.write(f"[{i:2d}]  {row['complexity']:4.0f}  {row['loss']:10.4f}  "
                        f"{row['R2']:7.4f}  {row['RMSE']:8.2f}  {row['MAE']:8.2f}  {row['equation']}\n")
        f.write(f"\nBest: V(t) = {best_eq}\n")
        f.write(f"R2={r2_best:.4f}  RMSE={rmse_best:.2f}  MAE={mae_best:.2f}\n")
        f.write(f"RMSE improvement over KD95: {(rmse_kd-rmse_best)/rmse_kd*100:.1f}%\n")
    with open(f"{output_dir}/best_equation_full.txt", 'w') as f:
        f.write(f"Run: {run_id}\nV(t) = {best_eq}\n")
        f.write(f"PySR:  R2={r2_best:.4f}  RMSE={rmse_best:.2f}  MAE={mae_best:.2f}\n")
        f.write(f"KD95:  R2={r2_kd:.4f}  RMSE={rmse_kd:.2f}  MAE={mae_kd:.2f}\n")
        f.write(f"Improvement: {(rmse_kd-rmse_best)/rmse_kd*100:.1f}%\n")
        f.write(f"N={len(y)}  Elapsed={elapsed_s:.1f}s\n")

    return r2_best, rmse_best, mae_best, str(best_eq)


def single_run(X, y, run_number, seed):
    import os
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}_seed{seed}"
    output_dir = f"{OUTPUT_BASE}/{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*80}\n# RUN {run_number} | {run_id}\n{'#'*80}")
    config = PYSR_CONFIG.copy()
    config['random_state'] = seed
    model = PySRRegressor(**config)

    t_start = time.time()
    model.fit(X, y)
    elapsed = time.time() - t_start

    r2, rmse, mae, eq = report_results(model, X, y, output_dir, run_id, elapsed)
    return {'run': run_number, 'seed': seed, 'run_id': run_id,
            'R2': r2, 'RMSE': rmse, 'MAE': mae, 'equation': eq, 'elapsed_s': elapsed}


def main():
    import os
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    seeds = list(range(42, 42 + N_RUNS))

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")
    X, y, clean = prepare_data(df)

    all_results = []
    for i, seed in enumerate(seeds, 1):
        all_results.append(single_run(X, y, i, seed))
        # write the summary after EVERY seed, so a run that is cut short
        # still leaves usable results on disk
        pd.DataFrame(all_results).to_csv(f"{OUTPUT_BASE}/all_runs_summary.csv",
                                         index=False)
        print(f"\n>>> {i}/{N_RUNS} seeds done. "
              f"all_runs_summary.csv updated (safe to stop here).\n")

    summary = pd.DataFrame(all_results)
    summary_file = f"{OUTPUT_BASE}/all_runs_summary.csv"
    summary.to_csv(summary_file, index=False)

    print(f"\n{'='*80}\nSUMMARY OF ALL {N_RUNS} NATIVE RUNS\n{'='*80}")
    for _, row in summary.iterrows():
        print(f"{row['run']:3.0f}  seed {row['seed']:4.0f}  R2={row['R2']:7.4f}  "
              f"RMSE={row['RMSE']:7.2f}  {row['equation']}")
    print(f"\nMean R2:   {summary['R2'].mean():.4f} +/- {summary['R2'].std():.4f}")
    print(f"Mean RMSE: {summary['RMSE'].mean():.2f} +/- {summary['RMSE'].std():.2f} kt")
    print(f"Total time: {summary['elapsed_s'].sum()/60:.1f} min")
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
