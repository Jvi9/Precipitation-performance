# -*- coding: utf-8 -*-
"""
ONE combined figure: 4 rows (KGE, CSI, CDD bias, CWD bias) x 2 columns
(spread magnitude, culprit product), all as subplots of a single figure -
not four separate saved images. One shared legend for the culprit column
(same 5 products every row) instead of repeating it four times.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
try:
    import contextily as cx
    HAVE_CONTEXTILY = True
except ImportError:
    HAVE_CONTEXTILY = False

from configurations import PLOT_ORDER, DISPLAY_NAMES, PRODUCT_COLORS, PRODUCT_MARKERS

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
geoparquet_path = os.path.join(wkDir, 'outputs', 'station_metrics_1mm_clipped.parquet')
boundary_path = os.path.join(wkDir, 'gis', 'study_area_shp', 'study_area.shp')  # optional
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

gdf = gpd.read_parquet(geoparquet_path)

def get_value_df(metric, mode, reference=None, derived_csi=False):
    if derived_csi:
        values = {}
        for code in PLOT_ORDER:
            sr = 1 - gdf[f'far_{code}']
            pod = gdf[f'pod_{code}']
            with np.errstate(divide='ignore', invalid='ignore'):
                values[code] = 1.0 / (1.0 / pod + 1.0 / sr - 1.0)
        value_df = pd.DataFrame(values, index=gdf.index)
        score_df = value_df.copy()
    else:
        cols = {code: f'{metric}_{code}' for code in PLOT_ORDER}
        value_df = gdf[list(cols.values())].copy()
        value_df.columns = PLOT_ORDER
        value_df.index = gdf.index
        if mode == 'max':
            score_df = value_df.copy()
        elif mode == 'closest_to_reference':
            ref = gdf[reference]
            value_df = value_df.sub(ref, axis=0)
            score_df = -value_df.abs()
    return value_df, score_df

metric_specs = {
    'KGE (accuracy)':          dict(metric='kge', mode='max'),
    'CSI (detection)':         dict(metric=None, mode='max', derived_csi=True),
    'CDD bias (dry extremes)': dict(metric='cdd', mode='closest_to_reference', reference='cdd_Observed'),
    'CWD bias (wet extremes)': dict(metric='cwd', mode='closest_to_reference', reference='cwd_Observed'),
}

boundary = gpd.read_file(boundary_path).to_crs(epsg=3857) if os.path.exists(boundary_path) else None

n_rows = len(metric_specs)
fig, axes = plt.subplots(n_rows, 2, figsize=(15, 7.5 * n_rows))

for row, (label, spec) in enumerate(metric_specs.items()):
    value_df, score_df = get_value_df(**spec)
    spread = value_df.max(axis=1) - value_df.min(axis=1)
    worst_product = score_df.idxmin(axis=1)

    plot_gdf = gdf.copy()
    plot_gdf['spread'] = spread.values
    plot_gdf['worst_code'] = worst_product.values
    plot_gdf_web = plot_gdf.to_crs(epsg=3857)
    p5, p95 = spread.quantile(0.05), spread.quantile(0.95)

    # --- Left column: spread magnitude ---
    ax = axes[row, 0]
    if boundary is not None:
        boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=3)
    plot_gdf_web.plot(ax=ax, column='spread', cmap='inferno_r', markersize=70,
                      edgecolor='black', linewidth=0.4, legend=True, vmin=p5, vmax=p95,
                      legend_kwds={'label': 'Inter-product spread', 'shrink': 0.75}, zorder=5)
    if HAVE_CONTEXTILY:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=8)
        except Exception:
            pass
    ax.set_axis_off()
    ax.set_title(f'Spread: {label}', fontsize=13)

    # --- Right column: culprit product (color + shape, no per-row legend) ---
    ax = axes[row, 1]
    if boundary is not None:
        boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=3)
    for code in PLOT_ORDER:
        mask = plot_gdf_web['worst_code'] == code
        sub = plot_gdf_web[mask]
        ax.scatter(sub.geometry.x, sub.geometry.y, color=PRODUCT_COLORS[code],
                   marker=PRODUCT_MARKERS[code], s=90, edgecolor='black', linewidth=0.5, zorder=5)
    if HAVE_CONTEXTILY:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=8)
        except Exception:
            pass
    ax.set_axis_off()
    ax.set_title(f'Furthest-from-group product: {label}', fontsize=13)

# --- One shared legend for the whole figure (culprit column), not repeated per row ---
legend_handles = [
    Line2D([0], [0], marker=PRODUCT_MARKERS[c], color='w', markerfacecolor=PRODUCT_COLORS[c],
           markeredgecolor='black', markersize=11, label=DISPLAY_NAMES[c])
    for c in PLOT_ORDER
]
fig.legend(handles=legend_handles, loc='lower center', ncol=5, fontsize=12,
           bbox_to_anchor=(0.5, -0.01), frameon=True)

fig.suptitle('Inter-product spread and furthest-from-group product,\n'
             'across four aspects of performance', fontsize=17, y=1.005)
plt.tight_layout(rect=[0, 0.02, 1, 1])

out_path = os.path.join(out_dir, 'spread_combined_grid.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved {out_path}")