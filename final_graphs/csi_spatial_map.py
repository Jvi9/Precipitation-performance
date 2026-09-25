# -*- coding: utf-8 -*-
"""
Single spatial map showing which product achieves the best Critical Success
Index (CSI) at each station, with marker size showing the margin over the
runner-up - the spatial companion to performance_diagram.py.

CSI is not a stored GeoParquet column - it's derived algebraically from POD
and FAR (CSI = 1 / (1/POD + 1/SR - 1), where SR = 1 - FAR), the same
relationship used to build the performance diagram's background contours.
FBI (POD/SR) is reported alongside for the same winning product, since it's
the second axis of the performance-diagram story.

This is ONE map (categorical winner + magnitude), not five side-by-side
raw-value panels - same style as margin_map.py, just computed on a derived
metric instead of a stored column.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
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
# 1. Compute CSI and FBI per product, per station (derived, not stored)
# =============================================================================
csi_cols = {}
fbi_derived_cols = {}
for code in PLOT_ORDER:
    sr = 1 - gdf[f'far_{code}']
    pod = gdf[f'pod_{code}']
    with np.errstate(divide='ignore', invalid='ignore'):
        csi = 1.0 / (1.0 / pod + 1.0 / sr - 1.0)
    csi_cols[code] = csi
    fbi_derived_cols[code] = pod / sr

csi_df = pd.DataFrame(csi_cols)   # station x product
fbi_df = pd.DataFrame(fbi_derived_cols)

# =============================================================================
# 2. Winner (highest CSI) and margin over runner-up, per station
# =============================================================================
sorted_vals = np.sort(csi_df.values, axis=1)
top1 = sorted_vals[:, -1]
top2 = sorted_vals[:, -2]
margin = top1 - top2
winner = csi_df.idxmax(axis=1)

plot_gdf = gdf.copy()
plot_gdf['winner'] = winner
plot_gdf['csi_margin'] = margin
plot_gdf['winner_csi'] = top1
# FBI of the winning product at each station, for the caption/text
plot_gdf['winner_fbi'] = [fbi_df.loc[i, w] for i, w in zip(gdf.index, winner)]

# =============================================================================
# 3. Plot: single categorical map, marker size = CSI margin
# =============================================================================
plot_gdf = plot_gdf.to_crs(epsg=3857)

m = plot_gdf['csi_margin']
p5, p95 = m.quantile(0.05), m.quantile(0.95)
m_clipped = m.clip(p5, p95)
size_range = (40, 500)
m_norm = (m_clipped - p5) / (p95 - p5) if p95 > p5 else m_clipped * 0
sizes = size_range[0] + m_norm * (size_range[1] - size_range[0])

fig, ax = plt.subplots(figsize=(11, 12), dpi=200)

if os.path.exists(boundary_path):
    boundary = gpd.read_file(boundary_path).to_crs(epsg=3857)
    boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, zorder=3)

for code in PLOT_ORDER:
    mask = plot_gdf['winner'] == code
    subset = plot_gdf[mask]
    subset.plot(ax=ax, color=PRODUCT_COLORS[code], markersize=sizes[mask],
                edgecolor='black', linewidth=0.5, alpha=0.85, label=DISPLAY_NAMES[code], zorder=5)

if HAVE_CONTEXTILY:
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=8)
    except Exception as e:
        print(f"Basemap failed ({e}), continuing without it.")

ax.set_axis_off()
ax.set_title('Spatial distribution of best-performing product by CSI\n'
             '(marker size = CSI margin over runner-up)', fontsize=14)

color_handles = [mpatches.Patch(color=PRODUCT_COLORS[c], label=DISPLAY_NAMES[c]) for c in PLOT_ORDER]
legend1 = ax.legend(handles=color_handles, loc='lower left', frameon=True,
                     framealpha=0.9, fontsize=10, title='Best CSI')
ax.add_artist(legend1)

size_vals = [p5, (p5 + p95) / 2, p95]
size_legend_sizes = [size_range[0], (size_range[0] + size_range[1]) / 2, size_range[1]]
size_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markeredgecolor='black', markersize=np.sqrt(s), label=f'{v:.3f}')
                for s, v in zip(size_legend_sizes, size_vals)]
ax.legend(handles=size_handles, loc='lower right', frameon=True, framealpha=0.9,
          fontsize=9, title='CSI margin')

plt.tight_layout()
out_path = os.path.join(out_dir, 'csi_spatial_map.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved {out_path}")

# =============================================================================
# 4. Summary table for the caption/text
# =============================================================================
print("\n--- CSI winner counts across the network ---")
print(winner.value_counts().to_string())

print("\n--- Mean CSI and FBI of the winning product, by winner group ---")
summary = plot_gdf.groupby('winner')[['winner_csi', 'winner_fbi']].mean()
print(summary.round(3).to_string())

out_csv = os.path.join(wkDir, 'csi_spatial_summary.csv')
plot_gdf[['station', 'winner', 'csi_margin', 'winner_csi', 'winner_fbi']].to_csv(out_csv, index=False)
print(f"\nSummary table saved to {out_csv}")