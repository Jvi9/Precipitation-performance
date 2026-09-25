# -*- coding: utf-8 -*-
"""
Figure 8, scatter version: every station plotted at its actual elevation and
actual metric value (no binning), with a LOESS-smoothed trend line per
product, Spearman rho annotated on each panel, AND the station elevation
distribution blended into the SAME figure as a top marginal strip - not a
separate file.

LOESS (LOcally Estimated Scatterplot Smoothing): at each point along the
x-axis, fits a small local straight-line regression using only the nearest
neighborhood of data (weighted so closer points count more), and uses that
local fit's prediction as the curve's value there. Repeating this across
the x-range gives a smooth curve that can bend to follow the data's actual
shape, without assuming linearity everywhere the way a single regression
line would.

Spearman's rho: measures whether two variables move together in RANK order
(not necessarily a straight-line relationship) - ranges from -1 to +1.
|rho| < 0.3 = weak, 0.3-0.5 = moderate, > 0.5 = strong. The p-value answers
a different question: given the sample size, how likely is a correlation
this large to appear by chance if there were truly no relationship? p<0.05
(marked with *) is the conventional "probably not just noise" cutoff - a
high rho with no * is a sign the pattern may not be reliable.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

from configurations import PLOT_ORDER, DISPLAY_NAMES, PRODUCT_COLORS

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
geoparquet_path = os.path.join(wkDir, 'outputs', 'station_metrics_1mm_clipped.parquet')
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)


def simple_lowess(x, y, frac=0.6, n_points=100):
    """Minimal LOWESS using only numpy (no statsmodels dependency). Tricube-
    weighted local linear fit evaluated at n_points across the data range."""
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    n = len(x_sorted)
    k = max(int(np.ceil(frac * n)), 2)

    x_eval = np.linspace(x_sorted.min(), x_sorted.max(), n_points)
    y_eval = np.empty(n_points)

    for i, x0 in enumerate(x_eval):
        dist = np.abs(x_sorted - x0)
        idx = np.argsort(dist)[:k]
        d_max = dist[idx].max()
        weights = np.ones(k) if d_max == 0 else np.clip((1 - (dist[idx] / d_max) ** 3) ** 3, 0, None)
        xi, yi, wi = x_sorted[idx], y_sorted[idx], weights
        W = np.diag(wi)
        X = np.vstack([np.ones(k), xi]).T
        try:
            beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ yi)
            y_eval[i] = beta[0] + beta[1] * x0
        except np.linalg.LinAlgError:
            y_eval[i] = np.average(yi, weights=wi)
    return x_eval, y_eval


def interpret_rho(rho, pval):
    strength = 'weak' if abs(rho) < 0.3 else ('moderate' if abs(rho) < 0.5 else 'strong')
    sig = 'statistically significant (p<0.05)' if pval < 0.05 else 'NOT statistically significant'
    direction = 'increases' if rho > 0 else 'decreases'
    return strength, sig, direction


gdf = gpd.read_parquet(geoparquet_path)

print("="*90)
print("WHAT THESE NUMBERS MEAN")
print("="*90)
print("LOESS trend line: a smooth curve fit locally to nearby points (not a")
print("straight-line assumption) - shows the SHAPE the data suggests without")
print("imposing a formula. Bendy/flat LOESS lines with low |rho| = weak signal;")
print("a smooth, consistent slope with high |rho| and a '*' = a real, reliable trend.")
print()
print("Spearman rho: -1 to +1, how consistently the metric rises/falls with")
print("elevation (rank-based, not just straight-line fit). |rho|<0.3 weak,")
print("0.3-0.5 moderate, >0.5 strong. '*' means p<0.05 (unlikely to be pure chance")
print("given n=70 stations); no '*' means treat the pattern with real caution.")
print("="*90)

# =============================================================================
# Combined figure: TOP = elevation distribution (spans both columns),
# BOTTOM 2x2 = PBIAS (a), RMSE (b), KGE (c), r (d) scatter + LOESS
# =============================================================================
fig = plt.figure(figsize=(24, 12))
gs = gridspec.GridSpec(3, 2, height_ratios=[0.5, 1, 1], hspace=0.35, wspace=0.22, figure=fig)

# --- Top strip: station elevation distribution ---
ax_hist = fig.add_subplot(gs[0, :])
ax_hist.hist(gdf['alt'], bins=30, color='steelblue', edgecolor='black', alpha=0.75)
ax_hist.set_xlabel('Elevation (m)', fontsize=13)
ax_hist.set_ylabel('Number of\nstations', fontsize=13)
ax_hist.set_title('Station elevation distribution (n=70)', fontsize=15, loc='left')
ax_hist.tick_params(labelsize=11)
ax_hist.grid(axis='y', alpha=0.3)

# --- Bottom 2x2: metrics vs elevation ---
panel_specs = [('pbias', 'PBIAS', gs[1, 0], 'a', 0.0),
              ('rmse', 'RMSE', gs[1, 1], 'b', None),
              ('kge', 'KGE', gs[2, 0], 'c', 1.0),
              ('r', 'r', gs[2, 1], 'd', None)]

print("\n" + "="*90)
print("PER-PRODUCT TREND STRENGTH AND SIGNIFICANCE, BY METRIC")
print("="*90)

for metric_key, metric_label, gs_pos, panel_letter, ref_line in panel_specs:
    ax = fig.add_subplot(gs_pos)
    rho_text_lines = []
    print(f"\n--- {metric_label} vs elevation ---")
    for code in PLOT_ORDER:
        x = gdf['alt'].values
        y = gdf[f'{metric_key}_{code}'].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid, y_valid = x[mask], y[mask]

        ax.scatter(x_valid, y_valid, color=PRODUCT_COLORS[code], alpha=0.45, s=35,
                  edgecolor='none', label=DISPLAY_NAMES[code])

        smoothed_x, smoothed_y = simple_lowess(x_valid, y_valid, frac=0.6)
        ax.plot(smoothed_x, smoothed_y, color=PRODUCT_COLORS[code], linewidth=2.5)

        rho, pval = spearmanr(x_valid, y_valid)
        sig = '*' if pval < 0.05 else ''
        rho_text_lines.append(f"{DISPLAY_NAMES[code]}: \u03c1={rho:+.2f}{sig}")

        strength, sig_text, direction = interpret_rho(rho, pval)
        print(f"  {DISPLAY_NAMES[code]:12s} rho={rho:+.3f}  {strength:8s}  "
              f"metric {direction} with elevation  -  {sig_text}")

    if ref_line is not None:
        ax.axhline(ref_line, color='black', linestyle='--', linewidth=1, alpha=0.6)

    ax.set_xlabel('Elevation (m)', fontsize=14)
    ax.set_ylabel(metric_label, fontsize=16)
    ax.set_title(f'({panel_letter}) {metric_label}', fontsize=17, loc='left')
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)

    rho_box_text = '\n'.join(rho_text_lines)
    ax.text(0.98, 0.03, rho_box_text, transform=ax.transAxes, fontsize=9,
           ha='right', va='bottom', family='monospace',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))

    if panel_letter == 'a':
        ax.legend(fontsize=10, title='Product', title_fontsize=11, loc='upper right', framealpha=0.9)

fig.suptitle('Relationship between elevation and daily precipitation performance metrics\n'
             '(points = individual stations, lines = LOESS trend, boxed values = Spearman \u03c1)',
             fontsize=19, y=1.01)

out_path = os.path.join(out_dir, 'figure8_elevation_scatter.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved {out_path}")