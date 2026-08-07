"""Builds supplemental Fig. S1 (out-of-fold 1:1 diagnostic) from the archived
per-point predictions.  Added in v1.0.2.  Run from the repository root:
  python scripts/fig_s1_one_to_one.py
"""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

oos = pd.read_csv('data/oos_predictions.csv')
obs = oos.obs.values.astype(float)
h = oos.h_used.values.astype(float)
models = {'KD95': oos.KD95_pred.values.astype(float), 'C11': oos.C11_pred.values.astype(float)}

TER = [('Low (<100 m)', h < 100, '#74A9CF'),
       ('Medium (100-300 m)', (h >= 100) & (h < 300), '#2B6CB0'),
       ('High (>=300 m)', h >= 300, '#0A2F6B')]
MCOL = {'KD95': '#E8862D', 'C11': '#8C1D2F'}
INK, MUT = '#1A1A1A', '#6B6B6B'

def stats(pred):
    e = pred - obs
    return (np.sqrt(np.mean(e**2)), 1 - np.sum(e**2)/np.sum((obs-obs.mean())**2), e.mean())

mpl.rcParams.update({'font.size': 9.5, 'axes.edgecolor': '#BBBBBB', 'axes.linewidth': 0.8,
                     'xtick.color': MUT, 'ytick.color': MUT, 'text.color': INK,
                     'axes.labelcolor': INK, 'font.family': 'DejaVu Sans'})
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))

lims = [20, 145]
for ax, name in zip(axes[:2], ['KD95', 'C11']):
    pred = models[name]
    ax.plot(lims, lims, ls='--', lw=1, color='#999999', zorder=1)
    for lab, m, c in TER:
        ax.scatter(obs[m], pred[m], s=14, color=c, alpha=0.65, lw=0.3,
                   edgecolor='white', zorder=3, label=lab)
    r, r2, b = stats(pred)
    ax.text(0.03, 0.97, f'{name}\nRMSE = {r:.2f} kt\n$R^2$ = {r2:.2f}\nbias = {b:+.2f} kt',
            transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(fc='white', ec='#CCCCCC', lw=0.6, boxstyle='round,pad=0.35'))
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Observed wind (kt)'); ax.grid(alpha=0.22, lw=0.5)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Out-of-fold predicted wind (kt)')
axes[0].set_title('(a) Kaplan-DeMaria baseline', fontsize=10.5, loc='left')
axes[1].set_title('(b) Terrain equation C11', fontsize=10.5, loc='left')
axes[1].legend(loc='lower right', fontsize=8, frameon=True, framealpha=0.9,
               edgecolor='#CCCCCC', title='Mean terrain $\\bar{h}$', title_fontsize=8)

# (c) residual vs terrain with binned means
ax = axes[2]
bins = np.array([0, 50, 100, 150, 200, 300, 450, 900])
ctr = 0.5*(bins[:-1]+bins[1:])
ax.axhline(0, color='#999999', lw=1, ls='--', zorder=1)
for name in ['KD95', 'C11']:
    e = models[name] - obs
    ax.scatter(h, e, s=9, color=MCOL[name], alpha=0.28, lw=0, zorder=2)
    bm = [e[(h >= a) & (h < b)].mean() for a, b in zip(bins[:-1], bins[1:])]
    ax.plot(ctr, bm, '-o', color=MCOL[name], lw=2, ms=5, zorder=4,
            markeredgecolor='white', markeredgewidth=0.8)
ax.text(430, 6.4, 'KD95', color=MCOL['KD95'], fontsize=9.5, fontweight='bold')
ax.text(430, -4.9, 'C11', color=MCOL['C11'], fontsize=9.5, fontweight='bold')
ax.set_xlabel('Mean terrain elevation $\\bar{h}$ (m)')
ax.set_ylabel('Predicted - observed (kt)')
ax.set_title('(c) Residual vs terrain (binned means)', fontsize=10.5, loc='left')
ax.set_xlim(0, 900); ax.set_ylim(-32, 32); ax.grid(alpha=0.22, lw=0.5)
ax.set_axisbelow(True)

fig.suptitle('Out-of-fold predictions vs observations, 453 native 6-hourly points, 121 storms',
             fontsize=11, y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig('figures/fig_one_to_one_diagnostic.png', dpi=300, bbox_inches='tight')
print('saved; pooled stats:')
for k, v in models.items():
    print(k, ['%.3f' % x for x in stats(v)])
