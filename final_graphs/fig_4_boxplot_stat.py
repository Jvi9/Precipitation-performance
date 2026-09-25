# -*- coding: utf-8 -*-
"""
Figure 4: distribution of r (a), RMSE (b), KGE (c), PBIAS (d) across all 5
products - the continuous-metric analog of the CSI/FBI/FAR/POD boxenplot.

A boxenplot shows the median, then the IQR (25th-75th) as the first box,
then successively narrower boxes extending outward by repeatedly halving
the remaining tail (12.5th-87.5th, 6.25th-93.75th, ...) - box width reflects
the density of data in that quantile band. This is what seaborn calls
boxenplot (originally "letter-value plot", Hofmann/Wickham/Kafadar 2011).

r, RMSE, KGE, and PBIAS are all already stored GeoParquet columns - unlike
CSI in the detection version, none of these need to be derived.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from configurations import PLOT_ORDER, DISPLAY_NAMES, PRODUCT_COLORS

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
geoparquet_path = os.path.join(wkDir, 'outputs', 'station_metrics_1mm_clipped.parquet')
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

gdf = gpd.read_parquet(geoparquet_path)
display_order = [DISPLAY_NAMES[c] for c in PLOT_ORDER]
palette = {DISPLAY_NAMES[c]: PRODUCT_COLORS[c] for c in PLOT_ORDER}

# =============================================================================
# 1. Build a tidy long-format table for r, RMSE, KGE, PBIAS
# =============================================================================
rows = []
for code in PLOT_ORDER:
    for station, r_v, rmse_v, kge_v, pbias_v in zip(
        gdf['station'], gdf[f'r_{code}'], gdf[f'rmse_{code}'], gdf[f'kge_{code}'], gdf[f'pbias_{code}']
    ):
        rows.append({
            'station': station, 'product': DISPLAY_NAMES[code],
            'r': r_v, 'RMSE': rmse_v, 'KGE': kge_v, 'PBIAS': pbias_v,
        })

long_df = pd.DataFrame(rows)

# =============================================================================
# 2. 2x2 boxenplot grid: r (a), RMSE (b), KGE (c), PBIAS (d)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(24, 12))
panel_specs = [
    ('r', axes[0, 0], 'a', None),
    ('RMSE', axes[0, 1], 'b', None),
    ('KGE', axes[1, 0], 'c', 1.0),     # reference line at KGE = 1 (perfect score)
    ('PBIAS', axes[1, 1], 'd', 0.0),   # reference line at PBIAS = 0 (unbiased)
]

for metric, ax, panel_letter, ref_line in panel_specs:
    sns.boxenplot(data=long_df, x='product', y=metric, order=display_order,
                 ax=ax, palette=palette, showfliers=True)
    if ref_line is not None:
        ax.axhline(ref_line, color='black', linestyle='--', linewidth=1.3, alpha=0.7)
    ax.set_xlabel('')
    ax.set_ylabel(metric, fontsize=16)
    ax.set_title(f'({panel_letter}) {metric}', fontsize=17, loc='left')
    ax.tick_params(axis='x', rotation=20, labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Distribution of performance daily statistics across all datasets', fontsize=21, y=1.02)
plt.tight_layout()

out_path = os.path.join(out_dir, 'performance_stats_boxenplot.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved {out_path}")

# =============================================================================
# 3. Summary stats for the caption/text
# =============================================================================
for metric in ['r', 'RMSE', 'KGE', 'PBIAS']:
    print(f"\n--- {metric} distribution by product ---")
    summary = long_df.groupby('product')[metric].agg(['median', 'mean', 'std']).reindex(display_order)
    print(summary.round(3).to_string())