# -*- coding: utf-8 -*-
"""
Two fixes to the previous combined figure:

1. ROBUST COLOR SCALING: the raw disagreement maps were being crushed by 1-2
   extreme outlier stations (e.g. Weberbauer's 214-day CDD disagreement)
   stretching the color scale so everything else looked flat. This clips to
   the 5th-95th percentile for coloring (same approach as margin_map.py),
   so the bulk of stations show visible variation. The colorbar still shows
   real units, it's only the color MAPPING that's robust to outliers.

2. PAIRED "CULPRIT" MAP: each disagreement map (magnitude - WHERE products
   disagree) is now paired with a categorical map of WHICH product is the
   outlier driving that disagreement at each station (worst-performing
   product by that metric). Together they answer both "how much" and "why."

Produces FOUR separate 2-panel figures (one per metric: KGE, CSI, CDD bias,
CWD bias) rather than one giant 8-panel figure, so you can choose which
ones actually earn a place in the paper vs. supplementary material.
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

from configurations import PLOT_ORDER, DISPLAY_NAMES, PRODUCT_COLORS

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
geoparquet_path = os.path.join(wkDir, 'outputs', 'station_metrics_1mm_clipped.parquet')
boundary_path = os.path.join(wkDir, 'gis', 'study_area_shp', 'study_area.shp')  # optional
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

gdf = gpd.read_parquet(geoparquet_path)

# =============================================================================
# 1. Compute value matrices, disagreement, and the "worst product" per metric
# =============================================================================
def get_value_df(metric, mode, reference=None, derived_csi=False):
    if derived_csi:
        values = {}
        for code in PLOT_ORDER:
            sr = 1 - gdf[f'far_{code}']
            pod = gdf[f'pod_{code}']
            with np.errstate(divide='ignore', invalid='ignore'):
                values[code] = 1.0 / (1.0 / pod + 1.0 / sr - 1.0)
        value_df = pd.DataFrame(values, index=gdf.index)
        score_df = value_df.copy()  # higher = better
    else:
        cols = {code: f'{metric}_{code}' for code in PLOT_ORDER}
        value_df = gdf[list(cols.values())].copy()
        value_df.columns = PLOT_ORDER
        value_df.index = gdf.index
        if mode == 'max':
            score_df = value_df.copy()
        elif mode == 'closest_to_reference':
            ref = gdf[reference]
            value_df = value_df.sub(ref, axis=0)  # convert to signed bias
            score_df = -value_df.abs()
    return value_df, score_df

metric_specs = {
    'KGE (accuracy)':       dict(metric='kge', mode='max'),
    'CSI (detection)':      dict(metric=None, mode='max', derived_csi=True),
    'CDD bias (dry extremes)': dict(metric='cdd', mode='closest_to_reference', reference='cdd_Observed'),
    'CWD bias (wet extremes)': dict(metric='cwd', mode='closest_to_reference', reference='cwd_Observed'),
}

# =============================================================================
# 2. Plot one 2-panel figure per metric: disagreement (robust color) + culprit
# =============================================================================
boundary = gpd.read_file(boundary_path).to_crs(epsg=3857) if os.path.exists(boundary_path) else None

for label, spec in metric_specs.items():
    value_df, score_df = get_value_df(**spec)
    disagreement = value_df.max(axis=1) - value_df.min(axis=1)
    worst_product = score_df.idxmin(axis=1)  # lowest score = worst performer = likely outlier

    plot_gdf = gdf.copy()
    plot_gdf['disagreement'] = disagreement.values
    plot_gdf['worst_product'] = worst_product.map(DISPLAY_NAMES).values
    plot_gdf_web = plot_gdf.to_crs(epsg=3857)

    # Robust color range for the disagreement panel
    p5, p95 = disagreement.quantile(0.05), disagreement.quantile(0.95)

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # --- Panel 1: disagreement magnitude, robust color scale ---
    ax = axes[0]
    if boundary is not None:
        boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, zorder=3)
    plot_gdf_web.plot(ax=ax, column='disagreement', cmap='inferno_r', markersize=100,
                      edgecolor='black', linewidth=0.4, legend=True, vmin=p5, vmax=p95,
                      legend_kwds={'label': 'Inter-product range (clipped to 5th-95th pctile for color)',
                                   'shrink': 0.7}, zorder=5)
    if HAVE_CONTEXTILY:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=8)
        except Exception:
            pass
    ax.set_axis_off()
    ax.set_title(f'Disagreement: {label}', fontsize=14)

    # --- Panel 2: which product is the culprit (categorical) ---
    ax = axes[1]
    if boundary is not None:
        boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, zorder=3)
    for code in PLOT_ORDER:
        disp = DISPLAY_NAMES[code]
        mask = plot_gdf_web['worst_product'] == disp
        plot_gdf_web[mask].plot(ax=ax, color=PRODUCT_COLORS[code], markersize=100,
                                edgecolor='black', linewidth=0.4, label=disp, zorder=5)
    if HAVE_CONTEXTILY:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=8)
        except Exception:
            pass
    ax.set_axis_off()
    ax.set_title(f'Which product is the outlier (worst performer): {label}', fontsize=14)
    handles = [mpatches.Patch(color=PRODUCT_COLORS[c], label=DISPLAY_NAMES[c]) for c in PLOT_ORDER]
    ax.legend(handles=handles, loc='lower left', frameon=True, framealpha=0.9, fontsize=10)

    plt.tight_layout()
    safe_label = label.split(' (')[0].replace(' ', '_')
    out_path = os.path.join(out_dir, f'disagreement_and_culprit_{safe_label}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved {out_path}")

    print(f"\n--- {label}: how often each product is the CULPRIT (worst performer) ---")
    print(plot_gdf['worst_product'].value_counts().to_string())