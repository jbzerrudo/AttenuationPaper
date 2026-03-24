"""
Stage 4b: PySR — Discover Full Decay Equation with Terrain (12x runs)
======================================================================
Full 1,164-point dataset. PySR discovers:
    V(t) = f(V0, t, h_max, h_mean, STORM_SPD, ...)

Input:  ph_decay_with_terrain.csv (from Stage 2)
Output: pysr_results_full/run_YYYYMMDD_HHMMSS_seedNN/
            pareto_front.csv
            pareto_front_metrics.txt
            best_equation_full.txt
        pysr_results_full/all_runs_summary.csv

Author: Jef / Claude pipeline
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
INPUT_CSV = r"D:\2026\SYNTC\ATTENUATE\OUTS\ph_decay_with_terrain.csv"
OUTPUT_BASE = r"D:\2026\SYNTC\ATTENUATE\OUTS\pysr_results_full"

PYSR_CONFIG = dict(
    niterations=300,
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
    feature_defs = {
        't_hours':    'Hours after landfall',
        'V0':         'Wind speed at landfall (kt)',
        'h_max':      'Max terrain height within 50km (m)',
        'h_mean':     'Mean terrain height within 50km (m)',
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
    print(f"\nClean data points: {len(clean)}")
    print(f"Unique storms:     {clean['SID'].nunique()}")

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

        r2_list.append(r2)
        rmse_list.append(rmse)
        mae_list.append(mae)

    equations['R2'] = r2_list
    equations['RMSE'] = rmse_list
    equations['MAE'] = mae_list

    return equations


def report_results(model, X, y, output_dir, run_id, elapsed_s):
    equations = evaluate_all_equations(model, X, y)

    # Kaplan & DeMaria baseline
    Vb_kd, R_kd, alpha_kd = 38.95, 1.0, 0.0393
    V0 = X['V0'].values
    t = X['t_hours'].values
    y_pred_kd = Vb_kd + (R_kd * V0 - Vb_kd) * np.exp(-alpha_kd * t)
    ss_tot = np.sum((y.values - np.mean(y.values)) ** 2)
    res_kd = y.values - y_pred_kd
    r2_kd = 1 - np.sum(res_kd ** 2) / ss_tot
    rmse_kd = np.sqrt(np.mean(res_kd ** 2))
    mae_kd = np.mean(np.abs(res_kd))

    print(f"\n{'='*80}")
    print(f"PySR RESULTS — RUN {run_id}")
    print(f"{'='*80}")

    if equations is not None:
        print(f"\n{'Idx':>3s}  {'Cplx':>4s}  {'Loss':>10s}  {'R²':>7s}  "
              f"{'RMSE':>8s}  {'MAE':>8s}  Equation")
        print(f"{'-'*3}  {'-'*4}  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*40}")

        for i, row in equations.iterrows():
            print(f"[{i:2d}]  {row['complexity']:4.0f}  {row['loss']:10.4f}  "
                  f"{row['R2']:7.4f}  {row['RMSE']:8.2f}  {row['MAE']:8.2f}  "
                  f"{row['equation']}")

    best_eq = model.sympy()
    y_pred_best = model.predict(X)
    res_best = y.values - y_pred_best
    ss_res = np.sum(res_best ** 2)
    r2_best = 1 - ss_res / ss_tot
    rmse_best = np.sqrt(np.mean(res_best ** 2))
    mae_best = np.mean(np.abs(res_best))

    print(f"\n  Best: V(t) = {best_eq}")
    print(f"  PySR:  R²={r2_best:.4f}  RMSE={rmse_best:.2f} kt  MAE={mae_best:.2f} kt")
    print(f"  KD95:  R²={r2_kd:.4f}  RMSE={rmse_kd:.2f} kt  MAE={mae_kd:.2f} kt")
    print(f"  RMSE improvement: {(rmse_kd-rmse_best)/rmse_kd*100:.1f}%")
    print(f"  Elapsed: {elapsed_s:.1f} s")

    # Save Pareto front CSV
    if equations is not None:
        equations.to_csv(f"{output_dir}/pareto_front.csv", index=False)

    # Save metrics text
    with open(f"{output_dir}/pareto_front_metrics.txt", 'w') as f:
        f.write(f"PySR Pareto Front — V(t) Prediction | Run: {run_id}\n")
        f.write(f"{'='*90}\n")
        f.write(f"N = {len(y)} data points | Elapsed: {elapsed_s:.1f} s\n")
        f.write(f"random_state = {PYSR_CONFIG.get('random_state', 'varies')}\n\n")
        f.write(f"KD95 baseline: R²={r2_kd:.4f}  RMSE={rmse_kd:.2f}  MAE={mae_kd:.2f}\n\n")
        f.write(f"{'Idx':>3s}  {'Cplx':>4s}  {'Loss':>10s}  {'R²':>7s}  "
                f"{'RMSE':>8s}  {'MAE':>8s}  Equation\n")
        f.write(f"{'-'*3}  {'-'*4}  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*50}\n")
        if equations is not None:
            for i, row in equations.iterrows():
                f.write(f"[{i:2d}]  {row['complexity']:4.0f}  {row['loss']:10.4f}  "
                        f"{row['R2']:7.4f}  {row['RMSE']:8.2f}  {row['MAE']:8.2f}  "
                        f"{row['equation']}\n")
        f.write(f"\nBest: V(t) = {best_eq}\n")
        f.write(f"R²={r2_best:.4f}  RMSE={rmse_best:.2f}  MAE={mae_best:.2f}\n")
        f.write(f"RMSE improvement over KD95: {(rmse_kd-rmse_best)/rmse_kd*100:.1f}%\n")

    # Save best equation
    with open(f"{output_dir}/best_equation_full.txt", 'w') as f:
        f.write(f"Run: {run_id}\n")
        f.write(f"V(t) = {best_eq}\n")
        f.write(f"PySR:  R²={r2_best:.4f}  RMSE={rmse_best:.2f}  MAE={mae_best:.2f}\n")
        f.write(f"KD95:  R²={r2_kd:.4f}  RMSE={rmse_kd:.2f}  MAE={mae_kd:.2f}\n")
        f.write(f"Improvement: {(rmse_kd-rmse_best)/rmse_kd*100:.1f}%\n")
        f.write(f"N={len(y)}  Elapsed={elapsed_s:.1f}s\n")

    return r2_best, rmse_best, mae_best, str(best_eq)


def single_run(X, y, run_number, seed):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}_seed{seed}"
    output_dir = f"{OUTPUT_BASE}/{run_id}"

    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*80}")
    print(f"# RUN {run_number} | {run_id}")
    print(f"{'#'*80}")

    config = PYSR_CONFIG.copy()
    config['random_state'] = seed

    model = PySRRegressor(**config)

    t_start = time.time()
    model.fit(X, y)
    elapsed = time.time() - t_start

    r2, rmse, mae, eq = report_results(model, X, y, output_dir, run_id, elapsed)

    return {'run': run_number, 'seed': seed, 'run_id': run_id,
            'R2': r2, 'RMSE': rmse, 'MAE': mae,
            'equation': eq, 'elapsed_s': elapsed}


def main():
    import os
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    N_RUNS = 12
    seeds = list(range(42, 42 + N_RUNS))

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    X, y, clean = prepare_data(df)

    all_results = []
    for i, seed in enumerate(seeds, 1):
        result = single_run(X, y, i, seed)
        all_results.append(result)

    # Summary
    summary = pd.DataFrame(all_results)
    summary_file = f"{OUTPUT_BASE}/all_runs_summary.csv"
    summary.to_csv(summary_file, index=False)

    print(f"\n{'='*80}")
    print(f"SUMMARY OF ALL {N_RUNS} RUNS")
    print(f"{'='*80}")
    print(f"\n{'Run':>3s}  {'Seed':>4s}  {'R²':>7s}  {'RMSE':>7s}  {'MAE':>7s}  "
          f"{'Time(s)':>7s}  Equation")
    print(f"{'-'*3}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*40}")
    for _, row in summary.iterrows():
        print(f"{row['run']:3.0f}  {row['seed']:4.0f}  {row['R2']:7.4f}  "
              f"{row['RMSE']:7.2f}  {row['MAE']:7.2f}  "
              f"{row['elapsed_s']:7.1f}  {row['equation']}")

    best_idx = summary['R2'].idxmax()
    print(f"\nBest run: R²={summary['R2'].max():.4f} "
          f"(run {summary.loc[best_idx, 'run']:.0f}, seed {summary.loc[best_idx, 'seed']:.0f})")
    print(f"Mean R²:   {summary['R2'].mean():.4f} ± {summary['R2'].std():.4f}")
    print(f"Mean RMSE: {summary['RMSE'].mean():.2f} ± {summary['RMSE'].std():.2f} kt")
    print(f"Total time: {summary['elapsed_s'].sum():.0f} s "
          f"({summary['elapsed_s'].sum()/60:.1f} min)")
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
