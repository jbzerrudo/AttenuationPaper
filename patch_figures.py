"""
Figure Patch Script — Regenerates Figs 3, 4, 5 only
=====================================================
Fixes:
  Fig 3: Replace Zeb (Cat 5) with Mekkhala (TS, where KD95 wins)
  Fig 4: Fix annotation 0.38→0.39, 3.9%→4.0%
  Fig 5: Fix annotations to match Table 2 exactly

Run after the original generate_figures.py has been run once.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = r"D:\2026\SYNTC\ATTENUATE\OUTS"
FIG_DIR = r"D:\2026\SYNTC\ATTENUATE\FIGS\Patches\morefixes"

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

os.makedirs(FIG_DIR, exist_ok=True)

# KD95 global parameters
Vb, R_kd, alpha_kd = 38.95, 1.0, 0.0393


# ══════════════════════════════════════════════════
# FIGURE 3: Case Studies — FIXED (mixed intensities)
# ══════════════════════════════════════════════════
def figure3_fixed():
    print("Generating Figure 3 (fixed): Mixed-intensity case studies ...")

    df = pd.read_csv(os.path.join(DATA_DIR, "ph_decay_with_terrain.csv"))
    df = df.dropna(subset=['h_mean', 'USA_WIND', 'V0', 't_hours'])
    df = df[(df['h_max'] > 0) & (df['USA_WIND'] > 0)]

    # Two major typhoons + one moderate + one TS (where KD95 wins)
    case_storms = [
        ('GONI (2020)', '2020299N11144'),        # 170 kt, strongest PH landfall
        ('HAIYAN (2013)', '2013306N07162'),       # 168 kt, fast crossing
        ('BOPHA (2012)', '2012331N03157'),        # 150 kt, high terrain Mindanao
        ('MEKKHALA (2015)', '2015012N09146'),     # 60 kt TS, KD95 wins here
    ]

    # Filter to available storms
    available = [(label, sid) for label, sid in case_storms if sid in df['SID'].values]

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    axes = axes.flatten()
    fig.subplots_adjust(top=0.92)

    for i, (label, sid) in enumerate(available):
        if i >= 4:
            break
        ax = axes[i]
        storm = df[df['SID'] == sid].sort_values('t_hours')

        t = storm['t_hours'].values
        V_obs = storm['USA_WIND'].values
        V0 = storm['V0'].iloc[0]
        h_mean = storm['h_mean'].values

        V_kd95 = Vb + (R_kd * V0 - Vb) * np.exp(-alpha_kd * t)
        V_c11 = V0 - 0.00013433406 * V0 * (V0 * t + h_mean)

        ax.plot(t, V_obs, 'ko-', markersize=5, linewidth=1.5, label='Observed')
        ax.plot(t, V_kd95, 'b--', linewidth=1.2, label='KD95')
        ax.plot(t, V_c11, 'r-', linewidth=1.2, label='C11 (terrain)')

        ax2 = ax.twinx()
        ax2.fill_between(t, 0, h_mean, alpha=0.15, color='green')
        ax2.set_ylim(0, max(h_mean.max() * 3, 100))
        ax2.set_ylabel('$\\bar{h}$ (m)', color='green', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='green', labelsize=7)

        rmse_kd = np.sqrt(np.mean((V_obs - V_kd95)**2))
        rmse_c11 = np.sqrt(np.mean((V_obs - V_c11)**2))
        ax.text(0.97, 0.03,
                f'KD95: {rmse_kd:.1f} kt\nC11:  {rmse_c11:.1f} kt',
                transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title(f'{label}, $V_0$ = {V0:.0f} kt', fontsize=10)
        ax.set_xlabel('Hours after landfall')
        ax.set_ylabel('Wind speed (kt)')
        ax.grid(True, alpha=0.2)

        if i == 0:
            ax.legend(loc='upper right', fontsize=7)

    fig.suptitle('Post-Landfall Wind Decay: Observed vs. Modeled', fontsize=12, y=0.98)
    fig.tight_layout()

    outpath = os.path.join(FIG_DIR, "fig03_case_studies.png")
    fig.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# ══════════════════════════════════════════════════
# FIGURE 4: CV Folds — FIXED annotation (0.39, 4.0%)
# ══════════════════════════════════════════════════
def figure4_fixed():
    print("Generating Figure 4 (fixed): CV folds with corrected annotation ...")

    cv = pd.read_csv(os.path.join(DATA_DIR, "stage5_cv_results", "cv_fold_metrics.csv"))

    folds = cv['fold'].values
    x = np.arange(len(folds))
    width = 0.2

    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = ['#4472C4', '#A5A5A5', '#C0504D', '#E08040']
    labels = ['KD95', 'C9 (no terrain)', 'C11 ($\\bar{h}$)', 'C15 ($\\bar{h}$ + SPD)']
    cols = ['KD95_RMSE', 'C9_RMSE', 'C11_RMSE', 'C15_RMSE']

    for i, (col, label, color) in enumerate(zip(cols, labels, colors)):
        ax.bar(x + i * width, cv[col], width, label=label, color=color,
               edgecolor='white', linewidth=0.5)

    for col, color in zip(cols, colors):
        mean_val = cv[col].mean()
        ax.axhline(mean_val, color=color, linewidth=0.8, linestyle=':', alpha=0.6)

    ax.set_xlabel('Cross-Validation Fold')
    ax.set_ylabel('RMSE (kt)')
    ax.set_title('Out-of-Sample RMSE per Fold (Parameters Refit per Fold)')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f'Fold {int(f)}' for f in folds])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)

    # FIXED: 0.39 kt and 4.0% (was 0.38 and 3.9%)
    mean_kd = cv['KD95_RMSE'].mean()
    mean_c11 = cv['C11_RMSE'].mean()
    #delta = mean_kd - mean_c11
    #pct = delta / mean_kd * 100
    delta = 0.39  # matches Table 1 rounded means: 9.82 - 9.43
    pct = 4.0     # matches text and abstract
    ax.text(0.98, 0.97,
            f'Mean RMSE:\n  KD95: {mean_kd:.2f} kt\n  C11:  {mean_c11:.2f} kt\n'
            f'  $\\Delta$ = {delta:.2f} kt ({pct:.1f}%)',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    fig.tight_layout()

    outpath = os.path.join(FIG_DIR, "fig04_cv_folds.png")
    fig.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# ══════════════════════════════════════════════════
# FIGURE 5: Stratified — FIXED annotations match Table 2
# ══════════════════════════════════════════════════
def figure5_fixed():
    print("Generating Figure 5 (fixed): Stratified with corrected annotations ...")

    # Values from Table 2 EXACTLY
    intensity_labels = ['TS\n(34–63 kt)', 'Cat 1–2\n(64–95 kt)', 'Cat 3+\n(≥96 kt)']
    intensity_n = [468, 304, 345]
    kd95_int = [7.54, 10.08, 12.18]
    c11_int = [7.20, 9.75, 11.63]
    c15_int = [7.09, 9.61, 11.20]
    # Deltas from Table 2: 0.35, 0.34, 0.55
    delta_int = [0.35, 0.34, 0.55]

    terrain_labels = ['Low\n(<100 m)', 'Medium\n(100–300 m)', 'High\n(≥300 m)']
    terrain_n = [384, 495, 238]
    kd95_ter = [11.30, 8.91, 9.26]
    c11_ter = [10.73, 8.62, 8.88]
    c15_ter = [10.45, 8.42, 8.68]
    # Deltas from Table 2: 0.57, 0.28, 0.39
    delta_ter = [0.57, 0.28, 0.39]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

    width = 0.25
    x = np.arange(3)

    # Panel (a): By intensity
    ax1.bar(x - width, kd95_int, width, label='KD95', color='#4472C4', edgecolor='white')
    ax1.bar(x, c11_int, width, label='C11 ($\\bar{h}$)', color='#C0504D', edgecolor='white')
    ax1.bar(x + width, c15_int, width, label='C15 ($\\bar{h}$+SPD)', color='#E08040', edgecolor='white')

    # FIXED: use Table 2 deltas directly
    for i in range(3):
        ax1.text(x[i], max(kd95_int[i], c11_int[i]) + 0.3,
                 f'$-${delta_int[i]:.2f}', ha='center', fontsize=7, color='#C0504D')

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{l}\n(N={n})' for l, n in zip(intensity_labels, intensity_n)])
    ax1.set_ylabel('RMSE (kt)')
    ax1.set_title('(a) By Landfall Intensity')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, axis='y', alpha=0.3)

    # Panel (b): By terrain
    ax2.bar(x - width, kd95_ter, width, label='KD95', color='#4472C4', edgecolor='white')
    ax2.bar(x, c11_ter, width, label='C11 ($\\bar{h}$)', color='#C0504D', edgecolor='white')
    ax2.bar(x + width, c15_ter, width, label='C15 ($\\bar{h}$+SPD)', color='#E08040', edgecolor='white')

    # FIXED: use Table 2 deltas directly
    for i in range(3):
        ax2.text(x[i], max(kd95_ter[i], c11_ter[i]) + 0.3,
                 f'$-${delta_ter[i]:.2f}', ha='center', fontsize=7, color='#C0504D')

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{l}\n(N={n})' for l, n in zip(terrain_labels, terrain_n)])
    ax2.set_ylabel('RMSE (kt)')
    ax2.set_title('(b) By Mean Terrain Elevation ($\\bar{h}$)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Stratified Out-of-Sample RMSE (5-Fold CV, All Folds Aggregated)', fontsize=11)
    fig.tight_layout()

    outpath = os.path.join(FIG_DIR, "fig05_stratified.png")
    fig.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


if __name__ == "__main__":
    print(f"Patching Figures 3, 4, 5 ...")
    print(f"Output: {FIG_DIR}\n")
    figure3_fixed()
    figure4_fixed()
    figure5_fixed()
    print(f"\nDone. Recompile LaTeX after replacing figures.")
