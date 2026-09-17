# -*- coding: utf-8 -*-
"""
Boxplot/violin of margin-of-victory, grouped by which product won, one panel
per metric. Answers "when Rain4PE wins, is it usually by a landslide or a
hair?" without a map's spatial clutter - complements margin_map.py, which
shows WHERE the close calls and blowouts happen; this shows the overall
distribution regardless of location.

Reads directly from the GeoParquet built by build_station_geoparquet.py.
"""
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
geoparquet_path = os.path.join(wkDir, 'outputs', 'station_metrics_1mm_clipped.parquet')
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
    sorted_vals = np.sort(goodness.values, axis=1)
    top1 = sorted_vals[:, -1]
    top2 = sorted_vals[:, -2]
    margin = top1 - top2
    winner = goodness.idxmax(axis=1)
    return winner, pd.Series(margin, index=gdf.index)


# =============================================================================
# Build a tidy long-format table: station, metric, winner, margin
# =============================================================================
gdf = gpd.read_parquet(geoparquet_path)
metrics_to_plot = ['kge', 'pod', 'far', 'cdd']  # edit as needed

rows = []
for metric in metrics_to_plot:
    winner, margin = compute_winner_and_margin(gdf, metric)
    for station, w, m in zip(gdf['station'], winner, margin):
        rows.append({'station': station, 'metric': metric, 'winner': w, 'margin': m})

long_df = pd.DataFrame(rows)

# =============================================================================
# Plot: one panel per metric, x = winning product, y = margin
# =============================================================================
n_metrics = len(metrics_to_plot)
fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6), sharey=False)
if n_metrics == 1:
    axes = [axes]

for ax, metric in zip(axes, metrics_to_plot):
    sub = long_df[long_df['metric'] == metric]
    order = [p for p in products if p in sub['winner'].unique()]

    sns.boxplot(data=sub, x='winner', y='margin', order=order, ax=ax,
                palette=[PRODUCT_COLORS[p] for p in order])
    sns.stripplot(data=sub, x='winner', y='margin', order=order, ax=ax,
                  color='black', alpha=0.4, size=3, jitter=True)

    # Annotate each box with n stations won
    counts = sub['winner'].value_counts()
    for i, prod in enumerate(order):
        n = counts.get(prod, 0)
        ax.text(i, ax.get_ylim()[1] * 0.97, f'n={n}', ha='center', va='top', fontsize=9)

    ax.set_title(f'{metric.upper()}: margin over runner-up\nby winning product')
    ax.set_xlabel('')
    ax.set_ylabel(f'Margin ({metric.upper()} units)')
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
out_path = os.path.join(out_dir, 'margin_boxplot_by_winner.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved {out_path}")

# =============================================================================
# Print summary stats and save the tidy table
# =============================================================================
print("\n--- Margin summary by metric and winning product ---")
print(long_df.groupby(['metric', 'winner'])['margin'].describe().to_string())

long_out_path = os.path.join(wkDir, 'margin_by_winner_long.csv')
long_df.to_csv(long_out_path, index=False)
print(f"\nTidy long-format table saved to {long_out_path}")