# -*- coding: utf-8 -*-
"""
INTERNAL DIAGNOSTIC TOOL - not a publication figure. Dense clustered heatmap:
rows = stations, columns = each product's KGE, CSI, CDD-bias, and CWD-bias
(20 columns total: 5 products x 4 metrics), z-scored per column so
everything is on the same visual scale, with hierarchical clustering
reordering BOTH axes.

What to look for:
  - A solid vertical block of bad (red) cells in the GPM-EXP columns =
    confirms its instability is concentrated, not random.
  - A solid vertical stripe of bad cells in the Rain4PE CWD-bias column
    running through almost every row = confirms that finding is nearly
    universal, not station-specific.
  - Clusters of ROWS (stations) that group together = stations sharing a
    similar overall "failure signature" across products/metrics - worth
    checking whether they share a region, altitude band, or basin.
  - Clusters of COLUMNS = which metrics tend to fail together (should
    roughly confirm the KGE/CSI/CDD correlation vs. CWD's independence
    you already found numerically - now visible directly).
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

from configurations import PLOT_ORDER, DISPLAY_NAMES

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
geoparquet_path = os.path.join(wkDir, 'outputs', 'station_metrics_1mm_clipped.parquet')
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

gdf = gpd.read_parquet(geoparquet_path)

# =============================================================================
# 1. Build the station x (product, metric) matrix
# =============================================================================
matrix_cols = {}

# KGE - higher is better, use as-is
for code in PLOT_ORDER:
    matrix_cols[f'{DISPLAY_NAMES[code]} | KGE'] = gdf[f'kge_{code}']

# CSI (derived) - higher is better
for code in PLOT_ORDER:
    sr = 1 - gdf[f'far_{code}']
    pod = gdf[f'pod_{code}']
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix_cols[f'{DISPLAY_NAMES[code]} | CSI'] = 1.0 / (1.0 / pod + 1.0 / sr - 1.0)

# CDD bias - closer to 0 is better, so use NEGATIVE absolute bias (higher = better, consistent direction)
for code in PLOT_ORDER:
    bias = gdf[f'cdd_{code}'] - gdf['cdd_Observed']
    matrix_cols[f'{DISPLAY_NAMES[code]} | CDD bias'] = -bias.abs()

# CWD bias - same treatment
for code in PLOT_ORDER:
    bias = gdf[f'cwd_{code}'] - gdf['cwd_Observed']
    matrix_cols[f'{DISPLAY_NAMES[code]} | CWD bias'] = -bias.abs()

matrix = pd.DataFrame(matrix_cols)
matrix.index = gdf['station']

# =============================================================================
# 2. Z-score each column so every metric is on the same visual scale
#    (higher z-score = better performance on that metric, consistently,
#    after the sign flips above)
# =============================================================================
z_matrix = (matrix - matrix.mean()) / matrix.std()
z_matrix = z_matrix.dropna(how='any')  # drop any station with a missing value in any column

# =============================================================================
# 3. Clustered heatmap
# =============================================================================
g = sns.clustermap(
    z_matrix,
    cmap='RdBu',
    center=0,
    vmin=-3, vmax=3,
    figsize=(16, 22),
    dendrogram_ratio=(0.12, 0.08),
    cbar_kws={'label': 'Z-score (higher = relatively better)'},
    xticklabels=True,
    yticklabels=True,
)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=6)
g.fig.suptitle('Diagnostic clustered heatmap: stations x product-metric performance (z-scored)',
              fontsize=15, y=1.01)

out_path = os.path.join(out_dir, 'diagnostic_clustered_heatmap.png')
g.savefig(out_path, dpi=250, bbox_inches='tight')
plt.show()
print(f"Saved {out_path}")

# =============================================================================
# 4. Print the column dendrogram order - tells you which metrics cluster together
# =============================================================================
col_order = z_matrix.columns[g.dendrogram_col.reordered_ind]
print("\nColumn clustering order (metrics/products that cluster together are adjacent):")
for c in col_order:
    print(f"  {c}")

print(f"\nMatrix shape used: {z_matrix.shape[0]} stations x {z_matrix.shape[1]} product-metric columns")
print("(stations with any missing value across these 20 columns were dropped - check")
print("for that if the row count looks lower than your full 70-station network)")