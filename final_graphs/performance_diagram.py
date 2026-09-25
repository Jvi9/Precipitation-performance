# -*- coding: utf-8 -*-
"""
Performance diagram (Roebber, 2009) - the standard categorical-verification
plot that replaces "CSI (a) and FBI (b) spatial maps" with ONE figure showing
FOUR statistics at once, for all products simultaneously:
  - x-axis: Success Ratio (SR = 1 - FAR)
  - y-axis: POD
  - background curved contours: CSI (derived algebraically from POD and SR -
    no new data needed, since CSI = 1 / (1/POD + 1/SR - 1))
  - background dashed radiating lines: Frequency Bias (FBI = POD/SR)

Each product appears as ONE point (network mean) plus a fainter cloud of its
per-station points, so you get both the central tendency and the spread in
one panel - no map, no side-by-side panels.

Reads directly from the GeoParquet (POD and FAR are already there - CSI and
FBI-as-a-position don't need to be separately computed or stored).
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from configurations import PRODUCT_CODES, DISPLAY_NAMES, PRODUCT_COLORS, PRODUCT_MARKERS, PLOT_ORDER

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
geoparquet_path = os.path.join(wkDir, 'outputs', 'station_metrics_1mm_clipped.parquet')
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

gdf = gpd.read_parquet(geoparquet_path)

# =============================================================================
# 1. Background contours: CSI as a function of (SR, POD); bias as radiating lines
# =============================================================================
sr_grid = np.linspace(0.01, 1, 200)
pod_grid = np.linspace(0.01, 1, 200)
SR, POD = np.meshgrid(sr_grid, pod_grid)
with np.errstate(divide='ignore', invalid='ignore'):
    CSI = 1.0 / (1.0 / POD + 1.0 / SR - 1.0)
    CSI = np.where(CSI < 0, np.nan, CSI)  # mask invalid (POD/SR combinations with no valid CSI)

fig, ax = plt.subplots(figsize=(9, 9))

csi_contours = ax.contourf(SR, POD, CSI, levels=np.arange(0, 1.05, 0.1),
                           cmap='Blues', alpha=0.35, extend='neither')
csi_lines = ax.contour(SR, POD, CSI, levels=np.arange(0.1, 1.0, 0.1),
                       colors='gray', linewidths=0.6)
ax.clabel(csi_lines, inline=True, fontsize=8, fmt='%.1f')

# Bias lines (FBI = POD / SR) radiating from origin
bias_values = [0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 2.0, 3.0, 5.0]
for b in bias_values:
    sr_line = np.linspace(0.001, 1, 100)
    pod_line = b * sr_line
    valid = pod_line <= 1
    ax.plot(sr_line[valid], pod_line[valid], linestyle='--', color='black', linewidth=0.6, alpha=0.6)
    if b * 1.0 <= 1.0:
        ax.annotate(f'{b}', xy=(1.0, b), fontsize=7, color='black', alpha=0.7,
                    ha='left', va='center')
    else:
        x_edge = 1.0 / b
        ax.annotate(f'{b}', xy=(x_edge, 1.0), fontsize=7, color='black', alpha=0.7,
                    ha='center', va='bottom')

cbar = fig.colorbar(csi_contours, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Critical Success Index (CSI)')

# =============================================================================
# 2. Plot each product: faint per-station cloud + bold network-mean marker
# =============================================================================
mean_points = {}  # store for the inset and for label placement

for code in PLOT_ORDER:
    pod_col = f'pod_{code}'
    far_col = f'far_{code}'
    sr_station = 1 - gdf[far_col]
    pod_station = gdf[pod_col]

    ax.scatter(sr_station, pod_station, color=PRODUCT_COLORS[code], marker=PRODUCT_MARKERS[code],
               alpha=0.15, s=30, zorder=3)

    sr_mean = sr_station.mean()
    pod_mean = pod_station.mean()
    mean_points[code] = (sr_mean, pod_mean)
    ax.scatter(sr_mean, pod_mean, color=PRODUCT_COLORS[code], marker=PRODUCT_MARKERS[code],
               edgecolor='black', linewidth=1.2, s=220, zorder=5, label=DISPLAY_NAMES[code])

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Success Ratio (1 - FAR)', fontsize=12)
ax.set_ylabel('Probability of Detection (POD)', fontsize=12)
ax.set_title('Performance diagram: POD, Success Ratio, CSI, and Bias\n'
             '(faint dots = individual stations, bold markers = network mean)', fontsize=12)
ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
ax.set_aspect('equal')

# =============================================================================
# 2b. Zoomed inset - the three GPM-family products sit almost on top of each
#     other at full scale (POD differs by ~0.006, SR by ~0.002), so shapes
#     and colors alone can't separate them. This zooms into their cluster,
#     redrawing the same background contours/bias lines at higher resolution,
#     with a rectangle on the main plot showing what's being magnified.
# =============================================================================
gpm_codes = [c for c in PLOT_ORDER if c in ('rawGPM', 'gwrGPM', 'expGPM')]
gpm_sr = [mean_points[c][0] for c in gpm_codes]
gpm_pod = [mean_points[c][1] for c in gpm_codes]
pad = 0.02
x0, x1 = min(gpm_sr) - pad, max(gpm_sr) + pad
y0, y1 = min(gpm_pod) - pad, max(gpm_pod) + pad

axins = inset_axes(ax, width='42%', height='42%', loc='lower right', borderpad=2)

sr_grid_z = np.linspace(x0, x1, 200)
pod_grid_z = np.linspace(y0, y1, 200)
SRz, PODz = np.meshgrid(sr_grid_z, pod_grid_z)
with np.errstate(divide='ignore', invalid='ignore'):
    CSIz = 1.0 / (1.0 / PODz + 1.0 / SRz - 1.0)
axins.contourf(SRz, PODz, CSIz, levels=20, cmap='Blues', alpha=0.35)
csi_lines_z = axins.contour(SRz, PODz, CSIz, colors='gray', linewidths=0.5)
axins.clabel(csi_lines_z, inline=True, fontsize=6, fmt='%.2f')

for b in bias_values:
    sr_line = np.linspace(x0, x1, 50)
    pod_line = b * sr_line
    axins.plot(sr_line, pod_line, linestyle='--', color='black', linewidth=0.5, alpha=0.5)

for code in PLOT_ORDER:
    sr_mean, pod_mean = mean_points[code]
    if x0 <= sr_mean <= x1 and y0 <= pod_mean <= y1:
        axins.scatter(sr_mean, pod_mean, color=PRODUCT_COLORS[code], marker=PRODUCT_MARKERS[code],
                      edgecolor='black', linewidth=1.2, s=180, zorder=5)

axins.set_xlim(x0, x1)
axins.set_ylim(y0, y1)
axins.set_xticks([])
axins.set_yticks([])
for spine in axins.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

mark_inset(ax, axins, loc1=2, loc2=1, fc='none', ec='0.4', linewidth=1)

plt.tight_layout()
out_path = os.path.join(out_dir, 'performance_diagram.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved {out_path}")

# =============================================================================
# 3. Print the numeric summary (mean POD, SR, CSI, FBI per product) for the caption/text
# =============================================================================
print("\n--- Network-mean POD, Success Ratio, CSI, and Bias per product ---")
for code in PLOT_ORDER:
    sr = (1 - gdf[f'far_{code}']).mean()
    pod = gdf[f'pod_{code}'].mean()
    csi = 1.0 / (1.0 / pod + 1.0 / sr - 1.0)
    bias = pod / sr
    print(f"{DISPLAY_NAMES[code]:12s} POD={pod:.3f}  SR={sr:.3f}  CSI={csi:.3f}  Bias(FBI)={bias:.3f}")