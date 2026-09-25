# -*- coding: utf-8 -*-
"""
Detection-metric companion to figure8_elevation_scatter.py: CSI (a), FBI (b),
FAR (c), POD (d) vs. elevation, all 5 products, same treatment - station-
level scatter (no binning), LOESS trend per product, Spearman rho printed
and interpreted in the terminal, elevation distribution blended into the
same figure as a top strip.

CSI is derived from POD/FAR (same formula used throughout this project);
FBI, FAR, and POD are already stored GeoParquet columns.
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
    """Minimal LOWESS using only numpy (no statsmodels dependency)."""
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

# --- Derive CSI per product (POD/FAR already stored) ---
csi_cols = {}
for code in PLOT_ORDER:
    sr = 1 - gdf[f'far_{code}']
    pod = gdf[f'pod_{code}']
    with np.errstate(divide='ignore', invalid='ignore'):
        csi_cols[code] = 1.0 / (1.0 / pod + 1.0 / sr - 1.0)
csi_df = pd.DataFrame(csi_cols, index=gdf.index)

print("="*90)
print("REMINDER: LOESS = local smoothed trend (no shape assumed); Spearman rho")
print("= rank-based monotonic correlation, -1 to +1. |rho|<0.3 weak, 0.3-0.5")
print("moderate, >0.5 strong. '*' = statistically significant (p<0.05, n=70).")
print("(see figure8_elevation_scatter.py for the fuller explanation)")
print("="*90)

# =============================================================================
# Combined figure: TOP = elevation distribution, BOTTOM 2x2 = CSI/FBI/FAR/POD
# =============================================================================
fig = plt.figure(figsize=(17, 18))
gs = gridspec.GridSpec(3, 2, height_ratios=[0.5, 1, 1], hspace=0.35, wspace=0.22, figure=fig)

ax_hist = fig.add_subplot(gs[0, :])
ax_hist.hist(gdf['alt'], bins=20, color='steelblue', edgecolor='black', alpha=0.75)
ax_hist.set_xlabel('Elevation (m)', fontsize=13)
ax_hist.set_ylabel('Number of\nstations', fontsize=13)
ax_hist.set_title('Station elevation distribution (n=70)', fontsize=15, loc='left')
ax_hist.tick_params(labelsize=11)
ax_hist.grid(axis='y', alpha=0.3)

panel_specs = [('CSI', gs[1, 0], 'a', None),
              ('FBI', gs[1, 1], 'b', 1.0),
              ('FAR', gs[2, 0], 'c', None),
              ('POD', gs[2, 1], 'd', None)]

print("\n" + "="*90)
print("PER-PRODUCT TREND STRENGTH AND SIGNIFICANCE, BY DETECTION METRIC")
print("="*90)

for metric_label, gs_pos, panel_letter, ref_line in panel_specs:
    ax = fig.add_subplot(gs_pos)
    rho_text_lines = []
    print(f"\n--- {metric_label} vs elevation ---")
    for code in PLOT_ORDER:
        x = gdf['alt'].values
        if metric_label == 'CSI':
            y = csi_df[code].values
        else:
            y = gdf[f'{metric_label.lower()}_{code}'].values
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

fig.suptitle('Relationship between elevation and detection performance metrics\n'
             '(points = individual stations, lines = LOESS trend, boxed values = Spearman \u03c1)',
             fontsize=19, y=1.01)

out_path = os.path.join(out_dir, 'figure_detection_elevation_scatter.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved {out_path}")