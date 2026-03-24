"""
Stage 3: Kaplan & DeMaria (1995) Baseline Fit
==============================================
Fits the classic exponential decay model:
    V(t) = Vb + (R*V0 - Vb) * exp(-alpha * t)

Two fitting modes:
  A) Global fit: single alpha, Vb, R for all storms
  B) Per-storm fit: individual alpha per storm (target for PySR in Stage 4)

Input:  ph_decay_with_terrain.csv (from Stage 2)
Output: ph_decay_fitted.csv       (adds KD95 predictions + per-storm alpha)
        kd95_per_storm.csv        (one row per storm: alpha, R, Vb, terrain stats)

Author: Jef / Claude pipeline
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────
INPUT_CSV = r"D:\2026\SYNTC\ATTENUATE\OUTS\ph_decay_with_terrain.csv"
OUTPUT_FITTED = r"D:\2026\SYNTC\ATTENUATE\OUTS\ph_decay_fitted.csv"
OUTPUT_PER_STORM = r"D:\2026\SYNTC\ATTENUATE\OUTS\kd95_per_storm.csv"


def kd95_model(t, Vb, R, alpha):
    """Kaplan & DeMaria (1995) exponential decay."""
    return Vb + (R - Vb) * np.exp(-alpha * t)


def fit_global(df):
    """Fit a single global KD95 model to all storms."""
    # For global fit, we use V_norm (normalized by V0)
    # V_norm(t) = Vb_n + (R_n - Vb_n) * exp(-alpha * t)
    # where Vb_n = Vb/V0, R_n = R (reduction factor at t=0)

    t = df['t_hours'].values
    V = df['USA_WIND'].values
    V0 = df['V0'].values

    # Fit in absolute wind space: V(t) = Vb + (R*V0 - Vb) * exp(-alpha*t)
    def model_abs(t_v0, Vb, R, alpha):
        t, v0 = t_v0
        return Vb + (R * v0 - Vb) * np.exp(-alpha * t)

    try:
        popt, pcov = curve_fit(
            model_abs,
            (t, V0),
            V,
            p0=[15.0, 0.9, 0.05],
            bounds=([0, 0.3, 0.001], [50, 1.0, 0.5]),
            maxfev=10000
        )
        Vb, R, alpha = popt
        V_pred = model_abs((t, V0), *popt)
        residuals = V - V_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((V - np.mean(V)) ** 2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean(residuals ** 2))

        return {
            'Vb': Vb, 'R': R, 'alpha': alpha,
            'R2': r2, 'RMSE': rmse,
            'V_pred': V_pred
        }
    except Exception as e:
        print(f"  Global fit failed: {e}")
        return None


def fit_per_storm(storm_df, global_params):
    """
    Fit alpha per storm, using global Vb and R as starting points.
    This produces the per-storm alpha that PySR will model.
    """
    t = storm_df['t_hours'].values
    V = storm_df['USA_WIND'].values
    V0 = storm_df['V0'].iloc[0]

    # Fix R from global fit, fit only Vb and alpha per storm
    R_fixed = global_params['R']

    def model_fixed_R(t, Vb, alpha):
        return Vb + (R_fixed * V0 - Vb) * np.exp(-alpha * t)

    try:
        popt, _ = curve_fit(
            model_fixed_R,
            t, V,
            p0=[global_params['Vb'], global_params['alpha']],
            bounds=([0, 0.001], [60, 1.0]),
            maxfev=5000
        )
        Vb_s, alpha_s = popt
        V_pred = model_fixed_R(t, *popt)
        residuals = V - V_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((V - np.mean(V)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = np.sqrt(np.mean(residuals ** 2))

        return {
            'Vb': Vb_s, 'alpha': alpha_s,
            'R2': r2, 'RMSE': rmse,
            'V_pred': V_pred
        }
    except:
        return None


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows, {df['SID'].nunique()} storms")

    # ── A) Global fit ──
    print(f"\n{'='*60}")
    print("A) GLOBAL KAPLAN & DeM ARIA FIT")
    print(f"{'='*60}")
    gfit = fit_global(df)
    if gfit:
        print(f"  Vb    = {gfit['Vb']:.2f} kt  (background wind)")
        print(f"  R     = {gfit['R']:.4f}      (landfall reduction factor)")
        print(f"  alpha = {gfit['alpha']:.4f} /hr (decay rate)")
        print(f"  R²    = {gfit['R2']:.4f}")
        print(f"  RMSE  = {gfit['RMSE']:.2f} kt")
        df['V_pred_global'] = gfit['V_pred']
    else:
        print("  FAILED — check data")
        return

    # ── B) Per-storm fit ──
    print(f"\n{'='*60}")
    print("B) PER-STORM ALPHA FIT")
    print(f"{'='*60}")

    storm_results = []
    df['V_pred_storm'] = np.nan
    df['alpha_storm'] = np.nan

    for sid in df['SID'].unique():
        mask = df['SID'] == sid
        storm_df = df[mask]

        sfit = fit_per_storm(storm_df, gfit)
        if sfit is None:
            continue

        df.loc[mask, 'V_pred_storm'] = sfit['V_pred']
        df.loc[mask, 'alpha_storm'] = sfit['alpha']

        # Aggregate terrain stats for this storm
        row = {
            'SID': sid,
            'NAME': storm_df['NAME'].iloc[0],
            'SEASON': storm_df['SEASON'].iloc[0],
            'V0': storm_df['V0'].iloc[0],
            'alpha': sfit['alpha'],
            'Vb': sfit['Vb'],
            'R2_storm': sfit['R2'],
            'RMSE_storm': sfit['RMSE'],
            'n_points': len(storm_df),
            'duration_h': storm_df['t_hours'].max(),
            'STORM_SPD_mean': storm_df['STORM_SPD'].mean(),
        }

        # Terrain stats (from Stage 2) — aggregate per storm
        for tcol in ['h_point', 'h_max', 'h_mean', 'h_std']:
            if tcol in storm_df.columns:
                row[f'{tcol}_mean'] = storm_df[tcol].mean()
                row[f'{tcol}_max'] = storm_df[tcol].max()

        # USA_RMW at landfall (if available)
        rmw_val = storm_df['USA_RMW'].iloc[0]
        row['RMW_landfall'] = rmw_val if rmw_val > 0 else np.nan

        storm_results.append(row)

    sdf = pd.DataFrame(storm_results)
    n_fitted = len(sdf)
    n_failed = df['SID'].nunique() - n_fitted

    print(f"  Successfully fitted: {n_fitted} storms")
    print(f"  Failed:              {n_failed} storms")
    print(f"\n  Per-storm alpha distribution:")
    print(f"    Mean:   {sdf['alpha'].mean():.4f} /hr")
    print(f"    Median: {sdf['alpha'].median():.4f} /hr")
    print(f"    Std:    {sdf['alpha'].std():.4f} /hr")
    print(f"    Min:    {sdf['alpha'].min():.4f} /hr")
    print(f"    Max:    {sdf['alpha'].max():.4f} /hr")
    print(f"\n  Per-storm R²:")
    print(f"    Mean:   {sdf['R2_storm'].mean():.4f}")
    print(f"    Median: {sdf['R2_storm'].median():.4f}")

    # ── Compare global vs per-storm ──
    valid = df.dropna(subset=['V_pred_storm'])
    rmse_global = np.sqrt(np.mean((valid['USA_WIND'] - valid['V_pred_global']) ** 2))
    rmse_storm = np.sqrt(np.mean((valid['USA_WIND'] - valid['V_pred_storm']) ** 2))
    print(f"\n  Global RMSE:    {rmse_global:.2f} kt")
    print(f"  Per-storm RMSE: {rmse_storm:.2f} kt")

    # ── Save ──
    df.to_csv(OUTPUT_FITTED, index=False)
    sdf.to_csv(OUTPUT_PER_STORM, index=False)
    print(f"\nOutput saved to:")
    print(f"  {OUTPUT_FITTED}  (all timesteps with predictions)")
    print(f"  {OUTPUT_PER_STORM}  (one row per storm — PySR input)")
    print(f"\nReady for Stage 4 (PySR symbolic regression on alpha).")


if __name__ == "__main__":
    main()
