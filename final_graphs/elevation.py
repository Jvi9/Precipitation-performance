# -*- coding: utf-8 -*-
"""
Tests whether the variation elevation DOESN'T explain is geographically
clustered (regional/local effects) rather than random noise.

Method: fit the LOESS elevation trend per product (same curve as
figure8_elevation_scatter.py), then for each station compute the RESIDUAL -
actual metric value minus what that station's own elevation would predict
from the smoothed curve. Map those residuals. If residuals cluster by
basin (e.g. one region consistently sits above/below its elevation-
expected value across products), that's real evidence of a regional
effect beyond topography. If residuals look geographically random, the
elevation trend already captures most of the systematic variation.

Uses RMSE first (your strongest, most universal elevation signal across all
5 products) - residualizing a real trend is more meaningful than
residualizing a metric with little elevation signal to begin with (e.g.
PISCO's r, which barely varies with elevation at all).
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
try:
    import contextily as cx
    HAVE_CONTEXTILY = True
except ImportError:
    HAVE_CONTEXTILY = False

from configurations import PLOT_ORDER, DISPLAY_NAMES, PRODUCT_COLORS

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
geoparquet_path = os.path.join(wkDir, 'outputs', 'station_metrics_1mm_clipped.parquet')
uh_shp_path = r'C:\Users\jvila\Desktop\Andean_project\gis\study_uhs\study_uhs.shp'
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

METRIC = 'pbias'  # edit to 'pbias', 'kge', 'r', etc. to test other metrics


def simple_lowess_predict(x, y, x_query, frac=0.6):
    """Fits LOWESS and returns predictions AT THE ORIGINAL x_query points
    (not a separate evenly-spaced grid) - needed to compute residuals."""
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    n = len(x_sorted)
    k = max(int(np.ceil(frac * n)), 2)
    preds = np.empty(len(x_query))
    for i, x0 in enumerate(x_query):
        dist = np.abs(x_sorted - x0)
        idx = np.argsort(dist)[:k]
        d_max = dist[idx].max()
        weights = np.ones(k) if d_max == 0 else np.clip((1 - (dist[idx] / d_max) ** 3) ** 3, 0, None)
        xi, yi, wi = x_sorted[idx], y_sorted[idx], weights
        W = np.diag(wi)
        X = np.vstack([np.ones(k), xi]).T
        try:
            beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ yi)
            preds[i] = beta[0] + beta[1] * x0
        except np.linalg.LinAlgError:
            preds[i] = np.average(yi, weights=wi)
    return preds


gdf = gpd.read_parquet(geoparquet_path)
uh = gpd.read_file(uh_shp_path)
for col in ['NOMB_UH_N5', 'NOMB_UH_N4']:
    uh[col] = uh[col].replace('', np.nan)
uh['UH_NAME'] = uh['NOMB_UH_N5'].fillna(uh['NOMB_UH_N4'])

# =============================================================================
# 1. Compute elevation-trend residual per product
# =============================================================================
residual_cols = {}
for code in PLOT_ORDER:
    x = gdf['alt'].values
    y = gdf[f'{METRIC}_{code}'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    predicted = np.full(len(x), np.nan)
    predicted[mask] = simple_lowess_predict(x[mask], y[mask], x[mask])
    residual_cols[code] = y - predicted   # positive = worse than elevation predicts, for RMSE

residual_df = pd.DataFrame(residual_cols, index=gdf.index)

# =============================================================================
# 2. Plot: small multiples, one residual map per product, shared color scale
# =============================================================================
uh_web = uh.to_crs(epsg=3857)
gdf_web = gdf.to_crs(epsg=3857)

vmax = np.nanpercentile(np.abs(residual_df.values), 95)  # robust, percentile-based scale

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
axes = axes.flatten()

for i, code in enumerate(PLOT_ORDER):
    ax = axes[i]
    uh_web.boundary.plot(ax=ax, color='gray', linewidth=0.8, zorder=1)
    sc = ax.scatter(gdf_web.geometry.x, gdf_web.geometry.y, c=residual_df[code], cmap='RdBu_r',
                    vmin=-vmax, vmax=vmax, s=90, edgecolor='black', linewidth=0.5, zorder=5)
    if HAVE_CONTEXTILY:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=8)
        except Exception:
            pass
    ax.set_axis_off()
    ax.set_title(DISPLAY_NAMES[code], fontsize=14)

# Turn off the unused 6th panel, put a shared colorbar there instead
axes[5].set_axis_off()
cbar = fig.colorbar(sc, ax=axes[5], fraction=0.6, aspect=15)
cbar.set_label(f'{METRIC.upper()} residual from elevation trend\n(+ = worse than elevation predicts)', fontsize=11)

fig.suptitle(f'{METRIC.upper()} residual from the elevation (LOESS) trend, by product\n'
            '(look for regions where residuals consistently cluster - e.g. all red or all blue\n'
            'within one basin, across multiple products - that suggests a REGIONAL effect beyond elevation)',
            fontsize=15, y=1.03)
plt.tight_layout()

out_path = os.path.join(out_dir, f'elevation_residual_map_{METRIC}.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved {out_path}")

# =============================================================================
# 3. Quantify: does residual variance differ significantly BY BASIN?
#    (a simple, honest first check - group residuals by basin, compare spread)
# =============================================================================
gdf['basin'] = gdf['station'].apply(lambda s: s.split('_')[0].strip())
print(f"\n--- Mean {METRIC.upper()} residual by basin and product ---")
print("(consistently positive or negative across MOST products for one basin =")
print("evidence of a real regional effect; mixed signs = no clear regional pattern)")
for code in PLOT_ORDER:
    print(f"\n{DISPLAY_NAMES[code]}:")
    by_basin = residual_df[code].groupby(gdf['basin']).mean()
    print(by_basin.round(3).to_string())