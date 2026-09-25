# -*- coding: utf-8 -*-
"""
Figure 6 (redone as boxenplot / letter-value plot instead of violin):
distribution of CSI (a), FBI (b), FAR (c), POD (d) across all 5 products.

A boxenplot shows the median, then the IQR (25th-75th) as the first box,
then successively narrower boxes extending outward by repeatedly halving
the remaining tail (12.5th-87.5th, 6.25th-93.75th, ...) - box width reflects
the density of data in that quantile band. This is what seaborn calls
boxenplot (originally "letter-value plot", Hofmann/Wickham/Kafadar 2011) -
built specifically for showing more distributional detail than a standard
box plot without the bandwidth-tuning ambiguity of a violin plot.

CSI is derived from POD/FAR (same formula used throughout); FBI, FAR, and
POD are already stored GeoParquet columns.
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
# 1. Build a tidy long-format table for CSI, FBI, FAR, POD
# =============================================================================
rows = []
for code in PLOT_ORDER:
    sr = 1 - gdf[f'far_{code}']
    pod = gdf[f'pod_{code}']
    with np.errstate(divide='ignore', invalid='ignore'):
        csi = 1.0 / (1.0 / pod + 1.0 / sr - 1.0)

    for station, csi_v, fbi_v, far_v, pod_v in zip(
        gdf['station'], csi, gdf[f'fbi_{code}'], gdf[f'far_{code}'], gdf[f'pod_{code}']
    ):
        rows.append({
            'station': station, 'product': DISPLAY_NAMES[code],
            'CSI': csi_v, 'FBI': fbi_v, 'FAR': far_v, 'POD': pod_v,
        })

long_df = pd.DataFrame(rows)

# =============================================================================
# 2. 2x2 boxenplot grid: CSI (a), FBI (b), FAR (c), POD (d)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(24, 12))
panel_specs = [
    ('CSI', axes[0, 0], 'a', None),
    ('FBI', axes[0, 1], 'b', 1.0),   # reference line at FBI = 1 (unbiased)
    ('FAR', axes[1, 0], 'c', None),
    ('POD', axes[1, 1], 'd', None),
]

for metric, ax, panel_letter, ref_line in panel_specs:
    sns.boxenplot(data=long_df, x='product', y=metric, order=display_order,
                 ax=ax, palette=palette, showfliers=True)
    if ref_line is not None:
        ax.axhline(ref_line, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('')
    ax.set_ylabel(metric, fontsize=16)
    ax.set_title(f'({panel_letter}) {metric}', fontsize=17, loc='left')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Distribution of forecasting indices across all datasets', fontsize=21, y=1.02)
plt.tight_layout()

out_path = os.path.join(out_dir, 'forecasting_indices_boxenplot.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved {out_path}")

# =============================================================================
# 3. Summary stats for the caption/text
# =============================================================================
for metric in ['CSI', 'FBI', 'FAR', 'POD']:
    print(f"\n--- {metric} distribution by product ---")
    summary = long_df.groupby('product')[metric].agg(['median', 'mean', 'std']).reindex(display_order)
    print(summary.round(3).to_string())