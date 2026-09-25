# -*- coding: utf-8 -*-
"""
Pairs the honest, all-stations elevation scatter (product = the color/story
throughout, same as Figure 8) with an actual MAP of the same stations,
colored by which product performs best there - real geography via station
coordinates, no basin polygons or regional taxonomy introduced.

Both panels use the SAME categorical variable (product) as the thread
connecting them - geography enters through real coordinates, not through
an uncharacterized "basin" construct.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    import contextily as cx
    HAVE_CONTEXTILY = True
except ImportError:
    HAVE_CONTEXTILY = False

from configurations import PLOT_ORDER, DISPLAY_NAMES, PRODUCT_COLORS, PRODUCT_MARKERS

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
geoparquet_path = os.path.join(wkDir, 'outputs', 'station_metrics_1mm_clipped.parquet')
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

METRIC = 'rmse'   # edit to try pbias/kge/r
LOWER_IS_BETTER = {'mae', 'rmse', 'pbias'}  # for pbias, this uses abs() below


def simple_lowess(x, y, frac=0.7, n_points=100):
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


gdf = gpd.read_parquet(geoparquet_path)

# =============================================================================
# Determine each station's winning product for this metric
# =============================================================================
value_cols = {code: (gdf[f'{METRIC}_{code}'].abs() if METRIC in LOWER_IS_BETTER
                     else gdf[f'{METRIC}_{code}']) for code in PLOT_ORDER}
value_df = pd.DataFrame(value_cols, index=gdf.index)
higher_is_better = METRIC not in LOWER_IS_BETTER
score_df = value_df if higher_is_better else -value_df
gdf['station_winner'] = score_df.idxmax(axis=1)

# =============================================================================
# Figure: (a) map colored by winning product, (b) elevation scatter + LOESS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# --- (a) Real map, colored by winning product ---
ax = axes[0]
gdf_web = gdf.to_crs(epsg=3857)
for code in PLOT_ORDER:
    mask = gdf_web['station_winner'] == code
    sub = gdf_web[mask]
    if len(sub) > 0:
        ax.scatter(sub.geometry.x, sub.geometry.y, color=PRODUCT_COLORS[code], marker=PRODUCT_MARKERS[code],
                  s=90, edgecolor='black', linewidth=0.6, label=DISPLAY_NAMES[code], zorder=5)
if HAVE_CONTEXTILY:
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=8)
    except Exception:
        pass
ax.set_axis_off()
ax.set_title(f'(a) Best-performing product by {METRIC.upper()}, mapped at each station', fontsize=15)
handles = [plt.Line2D([0], [0], marker=PRODUCT_MARKERS[c], color='w', markerfacecolor=PRODUCT_COLORS[c],
          markeredgecolor='black', markersize=10, label=DISPLAY_NAMES[c]) for c in PLOT_ORDER]
ax.legend(handles=handles, loc='lower left', frameon=True, framealpha=0.9, fontsize=10, title='Best product')

# --- (b) Elevation scatter + LOESS, all products, all stations ---
ax = axes[1]
rho_lines = []
from scipy.stats import spearmanr
for code in PLOT_ORDER:
    x = gdf['alt'].values
    y = gdf[f'{METRIC}_{code}'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    ax.scatter(x[mask], y[mask], color=PRODUCT_COLORS[code], alpha=0.45, s=35, edgecolor='none')
    sx, sy = simple_lowess(x[mask], y[mask])
    ax.plot(sx, sy, color=PRODUCT_COLORS[code], linewidth=2.5, label=DISPLAY_NAMES[code])
    rho, pval = spearmanr(x[mask], y[mask])
    sig = '*' if pval < 0.05 else ''
    rho_lines.append(f"{DISPLAY_NAMES[code]}: \u03c1={rho:+.2f}{sig}")

ax.set_xlabel('Elevation (m)', fontsize=15)
ax.set_ylabel(METRIC.upper(), fontsize=15)
ax.set_title(f'(b) {METRIC.upper()} vs. elevation, all stations', fontsize=15)
ax.tick_params(labelsize=12)
ax.grid(alpha=0.3)
ax.text(0.98, 0.03, '\n'.join(rho_lines), transform=ax.transAxes, fontsize=9,
       ha='right', va='bottom', family='monospace',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))
ax.legend(fontsize=9, title='Product', title_fontsize=10, loc='upper right')

fig.suptitle(f'{METRIC.upper()}: which product wins where (a), and how that relates to elevation (b)',
            fontsize=17, y=1.03)
plt.tight_layout()

out_path = os.path.join(out_dir, f'elevation_and_map_story_{METRIC}.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved {out_path}")