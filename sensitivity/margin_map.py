# -*- coding: utf-8 -*-
"""
Margin-of-victory map: extends the plain winner map by encoding HOW MUCH
better the winning product is than the runner-up at each station, via
marker size. A big marker = decisive win; a small marker = a close call
where the runner-up is nearly as good. Same categorical color-per-product
as the winner map, so this is a drop-in upgrade, not a new figure type to
learn.

Reads directly from the GeoParquet built by build_station_geoparquet.py.
If you've since rebuilt that file against the PISCO-date-fixed pickle,
point geoparquet_path at the new version before running this.
"""
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
try:
    import contextily as cx
    HAVE_CONTEXTILY = True
except ImportError:
    HAVE_CONTEXTILY = False

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
geoparquet_path = os.path.join(wkDir, 'outputs', 'station_metrics_1mm_clipped.parquet')
boundary_path = os.path.join(wkDir, 'gis', 'study_area_shp', 'study_area.shp')  # optional
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

products = ['rawGPM', 'gwrGPM', 'expGPM', 'rain4pe', 'PISCO']
PRODUCT_COLORS = {
    'rawGPM': '#1f77b4',
    'gwrGPM': '#2ca02c',
    'expGPM': '#d62728',
    'rain4pe': '#9467bd',
    'PISCO': '#ff7f0e',
}

# Same metric configuration as winner_map.py - 'mode' determines how "goodness"
# is scored so winner and margin can be computed generically for any metric.
METRIC_CONFIG = {
    'kge':  {'mode': 'max'},
    'pod':  {'mode': 'max'},
    'acc':  {'mode': 'max'},
    'r':    {'mode': 'max'},
    'far':  {'mode': 'min'},
    'mae':  {'mode': 'min'},
    'rmse': {'mode': 'min'},
    'fbi':  {'mode': 'closest_to_one'},
    'cdd':  {'mode': 'closest_to_reference', 'reference': 'cdd_Observed'},
    'cwd':  {'mode': 'closest_to_reference', 'reference': 'cwd_Observed'},
}


def compute_goodness(gdf, metric):
    """Returns a station x product DataFrame where HIGHER = better, regardless
    of the metric's original mode - lets winner/margin logic stay generic."""
    cfg = METRIC_CONFIG[metric]
    cols = {p: f'{metric}_{p}' for p in products}
    sub = gdf[list(cols.values())].copy()
    sub.columns = products

    if cfg['mode'] == 'max':
        goodness = sub
    elif cfg['mode'] == 'min':
        goodness = -sub
    elif cfg['mode'] == 'closest_to_one':
        goodness = -(sub - 1).abs()
    elif cfg['mode'] == 'closest_to_reference':
        ref = gdf[cfg['reference']]
        goodness = -sub.sub(ref, axis=0).abs()
    else:
        raise ValueError(f"Unknown mode for metric {metric}")
    return goodness


def compute_winner_and_margin(gdf, metric):
    goodness = compute_goodness(gdf, metric)
    sorted_vals = np.sort(goodness.values, axis=1)  # ascending
    top1 = sorted_vals[:, -1]
    top2 = sorted_vals[:, -2]
    margin = top1 - top2  # always >= 0; the actual metric's own units
    winner = goodness.idxmax(axis=1)
    return winner, pd.Series(margin, index=gdf.index)


def plot_margin_map(gdf, metric, title=None, size_range=(40, 500)):
    winner, margin = compute_winner_and_margin(gdf, metric)

    plot_gdf = gdf.copy()
    plot_gdf['winner'] = winner
    plot_gdf['margin'] = margin
    plot_gdf = plot_gdf.to_crs(epsg=3857)

    # Scale margin to marker size using percentile-based normalization
    # (robust to a few extreme outliers dominating the scale)
    m = plot_gdf['margin']
    p5, p95 = m.quantile(0.05), m.quantile(0.95)
    m_clipped = m.clip(p5, p95)
    if p95 > p5:
        m_norm = (m_clipped - p5) / (p95 - p5)
    else:
        m_norm = m_clipped * 0
    sizes = size_range[0] + m_norm * (size_range[1] - size_range[0])

    fig, ax = plt.subplots(figsize=(11, 12), dpi=200)

    if os.path.exists(boundary_path):
        boundary = gpd.read_file(boundary_path).to_crs(epsg=3857)
        boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, zorder=3)

    for prod in products:
        mask = plot_gdf['winner'] == prod
        subset = plot_gdf[mask]
        subset.plot(ax=ax, color=PRODUCT_COLORS[prod], markersize=sizes[mask],
                    edgecolor='black', linewidth=0.5, alpha=0.85, label=prod, zorder=5)

    if HAVE_CONTEXTILY:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=8)
        except Exception as e:
            print(f"Basemap failed ({e}), continuing without it.")

    ax.set_axis_off()
    ax.set_title(title or f'Winning product and margin of victory ({metric.upper()})\n'
                          f'(marker size = gap over runner-up)', fontsize=14)

    # Color legend (which product)
    color_handles = [mpatches.Patch(color=PRODUCT_COLORS[p], label=p) for p in products]
    legend1 = ax.legend(handles=color_handles, loc='lower left', frameon=True,
                         framealpha=0.9, fontsize=10, title='Winning product')
    ax.add_artist(legend1)

    # Size legend (margin magnitude) - three representative sizes
    size_vals = [p5, (p5 + p95) / 2, p95]
    size_legend_sizes = [size_range[0], (size_range[0] + size_range[1]) / 2, size_range[1]]
    size_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                           markeredgecolor='black', markersize=np.sqrt(s), label=f'{v:.3f}')
                    for s, v in zip(size_legend_sizes, size_vals)]
    ax.legend(handles=size_handles, loc='lower right', frameon=True, framealpha=0.9,
              fontsize=9, title=f'Margin ({metric.upper()} units)')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'margin_map_{metric}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved {out_path}")

    print(f"\n--- {metric.upper()}: margin summary by winning product ---")
    summary = pd.DataFrame({'winner': winner, 'margin': margin}).groupby('winner')['margin'].describe()
    print(summary.to_string())

    return winner, margin


# =============================================================================
# Run
# =============================================================================
gdf = gpd.read_parquet(geoparquet_path)

metrics_to_plot = ['kge', 'pod', 'far', 'cdd']  # edit as needed

results = {}
for metric in metrics_to_plot:
    results[metric] = plot_margin_map(gdf, metric)

# Save a combined table: station, winner, margin per metric
summary_table = gdf[['station', 'alt']].copy()
for metric, (winner, margin) in results.items():
    summary_table[f'winner_{metric}'] = winner.values
    summary_table[f'margin_{metric}'] = margin.values
summary_path = os.path.join(wkDir, 'margin_map_summary.csv')
summary_table.to_csv(summary_path, index=False)
print(f"\nSummary table saved to {summary_path}")