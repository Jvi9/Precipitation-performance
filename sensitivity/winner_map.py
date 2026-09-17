# -*- coding: utf-8 -*-
"""
Winner map: for a chosen metric, colors each station by which product performs
BEST there. Replaces "5 maps, one per product" with a single categorical map
that answers the actual question - does one product win everywhere, or does
the best choice vary by region/altitude?

Works directly off station_metrics_1mm_clipped.parquet (no need to rerun the
Generator pipeline).
"""
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# =============================================================================
# Metric configuration: how to decide a "winner" for each metric
#   mode='max'                -> higher value wins (e.g. kge, pod, acc, r)
#   mode='min'                -> lower value wins (e.g. far, mae, rmse)
#   mode='closest_to_one'     -> value closest to 1 wins (e.g. fbi)
#   mode='closest_to_reference' -> value closest to a reference column wins
#                                  (e.g. cdd/cwd vs cdd_Observed/cwd_Observed)
# =============================================================================
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


def compute_winner(gdf, metric):
    cfg = METRIC_CONFIG[metric]
    cols = {p: f'{metric}_{p}' for p in products}
    sub = gdf[list(cols.values())].copy()
    sub.columns = products  # rename for readability

    if cfg['mode'] == 'max':
        winner = sub.idxmax(axis=1)
    elif cfg['mode'] == 'min':
        winner = sub.idxmin(axis=1)
    elif cfg['mode'] == 'closest_to_one':
        winner = (sub - 1).abs().idxmin(axis=1)
    elif cfg['mode'] == 'closest_to_reference':
        ref = gdf[cfg['reference']]
        diffs = sub.sub(ref, axis=0).abs()
        winner = diffs.idxmin(axis=1)
    else:
        raise ValueError(f"Unknown mode for metric {metric}")
    return winner


def plot_winner_map(gdf, metric, title=None):
    winner = compute_winner(gdf, metric)
    plot_gdf = gdf.copy()
    plot_gdf['winner'] = winner
    plot_gdf = plot_gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(11, 12), dpi=200)

    if os.path.exists(boundary_path):
        boundary = gpd.read_file(boundary_path).to_crs(epsg=3857)
        boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, zorder=3)

    for prod in products:
        subset = plot_gdf[plot_gdf['winner'] == prod]
        subset.plot(ax=ax, color=PRODUCT_COLORS[prod], markersize=60,
                    edgecolor='black', linewidth=0.5, label=prod, zorder=5)

    if HAVE_CONTEXTILY:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=8)
        except Exception as e:
            print(f"Basemap failed ({e}), continuing without it.")

    ax.set_axis_off()
    ax.set_title(title or f'Best-performing product per station ({metric.upper()})', fontsize=15)

    handles = [mpatches.Patch(color=PRODUCT_COLORS[p], label=p) for p in products]
    ax.legend(handles=handles, loc='lower left', frameon=True, framealpha=0.9, fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'winner_map_{metric}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved {out_path}")

    print(f"\n--- Winner counts for {metric.upper()} ---")
    print(winner.value_counts().to_string())
    return winner


# =============================================================================
# Run
# =============================================================================
gdf = gpd.read_parquet(geoparquet_path)

# Edit this list to whichever metrics you want winner maps for
metrics_to_plot = ['kge', 'pod', 'far', 'cdd']

winners = {}
for metric in metrics_to_plot:
    winners[metric] = plot_winner_map(gdf, metric)

# Optional: save the winner assignments alongside station/altitude for a table in the paper
summary = gdf[['station', 'alt']].copy()
for metric, w in winners.items():
    summary[f'winner_{metric}'] = w.values
summary_path = os.path.join(wkDir, 'winner_map_summary.csv')
summary.to_csv(summary_path, index=False)
print(f"\nWinner summary table saved to {summary_path}")