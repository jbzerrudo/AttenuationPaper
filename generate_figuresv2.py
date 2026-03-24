"""
Attenuate Manuscript — Figure Generation Scripts
=================================================
Generates Figures 1–5 for the WAF manuscript.

All input data from: D:\2026\SYNTC\ATTENUATE\OUTS
Output figures to:   D:\2026\SYNTC\ATTENUATE\FIGS

Requirements: matplotlib, numpy, pandas, cartopy (for Fig 1), rasterio (for Fig 1)
Install if needed:  pip install cartopy rasterio

Author: Jef / Claude pipeline
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────
DATA_DIR = r"D:\2026\SYNTC\ATTENUATE\OUTS"
FIG_DIR = r"D:\2026\SYNTC\ATTENUATE\FIGS\NEW"
DTM_PATH = r"D:\2025\SYNTC\PreJUNE2025\GEVNEW\SOURCE\NEWSTORMS\DEM\dtm_phil.tif"

# AMS-friendly settings
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


# ══════════════════════════════════════════════════
# FIGURE 1: Study Area Map — TC Tracks over DTM
# ══════════════════════════════════════════════════
def figure1_map():
    """
    Map of Philippine terrain with 174 TC post-landfall tracks overlaid.
    Color-coded by landfall intensity.
    """
    print("Generating Figure 1: Study area map ...")

    df = pd.read_csv(os.path.join(DATA_DIR, "ph_decay_with_terrain.csv"))

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import rasterio
        from rasterio.enums import Resampling
        has_cartopy = True
    except ImportError:
        has_cartopy = False
        print("  WARNING: cartopy or rasterio not available. Making simplified map.")

    fig, ax = plt.subplots(
        figsize=(7, 9),
        subplot_kw={'projection': ccrs.PlateCarree()} if has_cartopy else {}
    )

    if has_cartopy:
        ax.set_extent([116, 128, 4, 22], crs=ccrs.PlateCarree())

        # Try to load DTM (downsampled for plotting)
        try:
            with rasterio.open(DTM_PATH) as src:
                # Downsample to ~1km for plotting
                scale = 50  # read every 50th pixel
                data = src.read(
                    1,
                    out_shape=(src.height // scale, src.width // scale),
                    resampling=Resampling.average
                )
                # Mask nodata and ocean
                data = data.astype(float)
                data[data < 0] = np.nan
                data[data > 3500] = np.nan

                extent = [src.bounds.left, src.bounds.right,
                          src.bounds.bottom, src.bounds.top]
                im = ax.imshow(
                    data, origin='upper', extent=extent,
                    cmap='terrain', vmin=0, vmax=2500, alpha=0.6,
                    transform=ccrs.PlateCarree()
                )
                cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02, label='Elevation (m)')
        except Exception as e:
            print(f"  Could not load DTM: {e}. Using cartopy terrain.")

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--')
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax.set_xlim(116, 128)
        ax.set_ylim(4, 22)
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        ax.grid(True, alpha=0.3)

    # Plot TC tracks color-coded by V0
    cmap_tracks = plt.cm.YlOrRd
    norm = plt.Normalize(34, 170)

    for sid in df['SID'].unique():
        storm = df[df['SID'] == sid].sort_values('t_hours')
        v0 = storm['V0'].iloc[0]
        color = cmap_tracks(norm(v0))
        transform = ccrs.PlateCarree() if has_cartopy else ax.transData

        if has_cartopy:
            ax.plot(storm['LON'], storm['LAT'], color=color, linewidth=0.6,
                    alpha=0.7, transform=ccrs.PlateCarree())
            ax.plot(storm['LON'].iloc[0], storm['LAT'].iloc[0], 'o',
                    color=color, markersize=2, transform=ccrs.PlateCarree())
        else:
            ax.plot(storm['LON'], storm['LAT'], color=color, linewidth=0.6, alpha=0.7)
            ax.plot(storm['LON'].iloc[0], storm['LAT'].iloc[0], 'o',
                    color=color, markersize=2)

    # Colorbar for tracks
    sm = plt.cm.ScalarMappable(cmap=cmap_tracks, norm=norm)
    sm.set_array([])
    cbar2 = plt.colorbar(sm, ax=ax, shrink=0.3, pad=0.06, label='Landfall $V_0$ (kt)')

    ax.set_title(f'Post-Landfall TC Tracks (N = {df.SID.nunique()} storms, 1977–2022)')

    outpath = os.path.join(FIG_DIR, "fig01_study_area.png")
    fig.savefig(outpath)
    plt.close()
    print(f"  Saved: {outpath}")


# ══════════════════════════════════════════════════
# FIGURE 2: Pareto Front — Loss vs Complexity
# ══════════════════════════════════════════════════
def figure2_pareto():
    print("Generating Figure 2: Pareto front ...")

    pf = pd.read_csv(os.path.join(DATA_DIR, "pysr_results_full", "pareto_front.csv"))

    has_terrain = pf['equation'].apply(
        lambda eq: any(t in str(eq) for t in ['h_max', 'h_mean'])
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Continuous line through all points
    ax.plot(pf['complexity'], pf['loss'], '-', color='gray',
            linewidth=1, alpha=0.4)

    # No-terrain markers
    no_terr = pf[~has_terrain]
    ax.plot(no_terr['complexity'], no_terr['loss'], 'o', color='gray',
            markersize=6, label='No terrain')

    # Terrain markers
    terr = pf[has_terrain]
    ax.plot(terr['complexity'], terr['loss'], 's', color='darkred',
            markersize=7, label='Terrain ($\\bar{h}$) present')

    # Label key complexities with equation numbers from manuscript
    labels = {
        7: 'C7',
        9: 'C9',
        11: 'C11',
        15: 'C15',
    }
    for cplx, label in labels.items():
        row = pf[pf['complexity'] == cplx]
        if len(row) > 0:
            x = row['complexity'].iloc[0]
            y = row['loss'].iloc[0]
            ax.annotate(label, xy=(x, y), xytext=(0, 25),
                        textcoords='offset points', fontsize=8,
                        ha='center', fontweight='bold')

    ax.set_xlabel('Expression Complexity (number of terms)')
    ax.set_ylabel('Mean Squared Error (kt$^2$)')
    ax.set_title('PySR Pareto Front: Loss vs. Complexity')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    outpath = os.path.join(FIG_DIR, "fig02_pareto_front.png")
    fig.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# ══════════════════════════════════════════════════
# FIGURE 3: Case Study Decay Curves
# ══════════════════════════════════════════════════
def figure3_case_studies():
    """
    Observed vs predicted decay curves for selected typhoons.
    Shows KD95, C9, and C11 predictions.
    """
    print("Generating Figure 3: Case study decay curves ...")

    df = pd.read_csv(os.path.join(DATA_DIR, "ph_decay_with_terrain.csv"))
    df = df.dropna(subset=['h_mean', 'USA_WIND', 'V0', 't_hours'])
    df = df[df['USA_WIND'] > 0]

    # KD95 parameters (global fit)
    Vb, R, alpha = 38.95, 1.0, 0.0393

    # Select case studies: strong, medium, and one with high terrain
    case_storms = {
        'GONI (2020)': '2020298N07131',       # 170 kt, strongest PH landfall
        'HAIYAN (2013)': '2013306N07149',      # 168 kt, fast crossing
        'BOPHA (2012)': '2012335N05135',       # 150 kt, Mindanao high terrain
        'KOPPU (2015)': '2015289N12138',       # slow mover if available
    }

    # Filter to storms that exist in data
    available = {}
    for label, sid in case_storms.items():
        if sid in df['SID'].values:
            available[label] = sid
    if len(available) == 0:
        # Fallback: pick top 4 by V0
        top4 = df.groupby('SID').first().nlargest(4, 'V0')
        for i, (sid, row) in enumerate(top4.iterrows()):
            available[f"{row['NAME']} ({int(row['SEASON'])})"] = sid

    n_storms = len(available)
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=False)
    axes = axes.flatten()

    for i, (label, sid) in enumerate(available.items()):
        if i >= 4:
            break
        ax = axes[i]
        storm = df[df['SID'] == sid].sort_values('t_hours')

        t = storm['t_hours'].values
        V_obs = storm['USA_WIND'].values
        V0 = storm['V0'].iloc[0]
        h_mean = storm['h_mean'].values

        # KD95
        V_kd95 = Vb + (R * V0 - Vb) * np.exp(-alpha * t)

        # C9 (no terrain)
        V_c9 = V0 - 0.00014813543 * V0**2 * t

        # C11 (terrain)
        V_c11 = V0 - 0.00013433406 * V0 * (V0 * t + h_mean)

        ax.plot(t, V_obs, 'ko-', markersize=5, linewidth=1.5, label='Observed')
        ax.plot(t, V_kd95, 'b--', linewidth=1.2, label='KD95')
        ax.plot(t, V_c11, 'r-', linewidth=1.2, label='C11 (terrain)')

        # Show terrain on secondary axis
        ax2 = ax.twinx()
        ax2.fill_between(t, 0, h_mean, alpha=0.15, color='green')
        ax2.set_ylim(0, max(h_mean.max() * 3, 100))
        ax2.set_ylabel('$\\bar{h}$ (m)', color='green', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='green', labelsize=7)

        # RMSE annotations
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

    # Hide unused subplots
    for j in range(len(available), 4):
        axes[j].set_visible(False)

    fig.suptitle('Post-Landfall Wind Decay: Observed vs. Modeled', fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.92)
    fig.tight_layout()

    outpath = os.path.join(FIG_DIR, "fig03_case_studies.png")
    fig.savefig(outpath)
    plt.close()
    print(f"  Saved: {outpath}")


# ══════════════════════════════════════════════════
# FIGURE 4: Cross-Validation RMSE by Fold
# ══════════════════════════════════════════════════
def figure4_cv_folds():
    """
    Grouped bar chart of RMSE per fold for KD95, C9, C11, C15.
    """
    print("Generating Figure 4: Cross-validation by fold ...")

    cv = pd.read_csv(os.path.join(DATA_DIR, "stage5_cv_results", "cv_fold_metrics.csv"))

    folds = cv['fold'].values
    x = np.arange(len(folds))
    width = 0.2

    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = ['#4472C4', '#A5A5A5', '#C0504D', '#E08040']
    labels = ['KD95', 'C9 (no terrain)', 'C11 ($\\bar{h}$)', 'C15 ($\\bar{h}$ + SPD)']
    cols = ['KD95_RMSE', 'C9_RMSE', 'C11_RMSE', 'C15_RMSE']

    for i, (col, label, color) in enumerate(zip(cols, labels, colors)):
        bars = ax.bar(x + i * width, cv[col], width, label=label, color=color,
                      edgecolor='white', linewidth=0.5)

    # Add mean lines
    for col, color, ls in zip(cols, colors, ['-', '--', '-', '--']):
        mean_val = cv[col].mean()
        ax.axhline(mean_val, color=color, linewidth=0.8, linestyle=':', alpha=0.6)

    ax.set_xlabel('Cross-Validation Fold')
    ax.set_ylabel('RMSE (kt)')
    ax.set_title('Out-of-Sample RMSE per Fold (Parameters Refit per Fold)')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f'Fold {int(f)}' for f in folds])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)

    # Annotate means
    mean_kd = cv['KD95_RMSE'].mean()
    mean_c11 = cv['C11_RMSE'].mean()
    ax.text(0.98, 0.97,
            f'Mean RMSE:\n  KD95: {mean_kd:.2f} kt\n  C11:  {mean_c11:.2f} kt\n'
            f'  Δ = {mean_kd - mean_c11:.2f} kt ({(mean_kd - mean_c11)/mean_kd*100:.1f}%)',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    fig.tight_layout()

    outpath = os.path.join(FIG_DIR, "fig04_cv_folds.png")
    fig.savefig(outpath)
    plt.close()
    print(f"  Saved: {outpath}")


# ══════════════════════════════════════════════════
# FIGURE 5: Stratified Performance
# ══════════════════════════════════════════════════
def figure5_stratified():
    """
    Two-panel figure: (a) RMSE by intensity class, (b) RMSE by terrain class.
    KD95 vs C11 vs C15.
    """
    print("Generating Figure 5: Stratified performance ...")

    # Data from cv_summary.txt (hardcoded from your results)
    # By intensity
    intensity_labels = ['TS\n(34–63 kt)', 'Cat 1–2\n(64–95 kt)', 'Cat 3+\n(≥96 kt)']
    intensity_n = [468, 304, 345]
    kd95_int = [7.54, 10.08, 12.18]
    c11_int = [7.20, 9.75, 11.63]
    c15_int = [7.09, 9.61, 11.20]

    # By terrain
    terrain_labels = ['Low\n(<100 m)', 'Medium\n(100–300 m)', 'High\n(≥300 m)']
    terrain_n = [384, 495, 238]
    kd95_ter = [11.30, 8.91, 9.26]
    c11_ter = [10.73, 8.62, 8.88]
    c15_ter = [10.45, 8.42, 8.68]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

    width = 0.25
    x = np.arange(3)

    # Panel (a): By intensity
    ax1.bar(x - width, kd95_int, width, label='KD95', color='#4472C4', edgecolor='white')
    ax1.bar(x, c11_int, width, label='C11 ($\\bar{h}$)', color='#C0504D', edgecolor='white')
    ax1.bar(x + width, c15_int, width, label='C15 ($\\bar{h}$+SPD)', color='#E08040', edgecolor='white')

    # Add improvement annotations
    for i in range(3):
        diff = kd95_int[i] - c11_int[i]
        ax1.text(x[i], max(kd95_int[i], c11_int[i]) + 0.3,
                 f'−{diff:.2f}', ha='center', fontsize=7, color='#C0504D')

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

    for i in range(3):
        diff = kd95_ter[i] - c11_ter[i]
        ax2.text(x[i], max(kd95_ter[i], c11_ter[i]) + 0.3,
                 f'−{diff:.2f}', ha='center', fontsize=7, color='#C0504D')

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{l}\n(N={n})' for l, n in zip(terrain_labels, terrain_n)])
    ax2.set_ylabel('RMSE (kt)')
    ax2.set_title('(b) By Mean Terrain Elevation ($\\bar{h}$)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Stratified Out-of-Sample RMSE (5-Fold CV, All Folds Aggregated)', fontsize=11)
    fig.tight_layout()

    outpath = os.path.join(FIG_DIR, "fig05_stratified.png")
    fig.savefig(outpath)
    plt.close()
    print(f"  Saved: {outpath}")


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Input:  {DATA_DIR}")
    print(f"Output: {FIG_DIR}")
    print(f"{'='*60}\n")

    figure1_map()
    figure2_pareto()
    figure3_case_studies()
    figure4_cv_folds()
    figure5_stratified()

    print(f"\n{'='*60}")
    print(f"All figures saved to {FIG_DIR}")
    print("Convert to PDF for AMS submission if needed:")
    print("  e.g., magick convert fig01_study_area.png fig01_study_area.pdf")
