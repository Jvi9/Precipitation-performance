# -*- coding: utf-8 -*-
"""
ONE combined figure, 4 map panels in a 2x2 grid - the SAME map type run twice:

  TOP ROW    - full comparison, all 5 products:
               (a) best-performing product by CSI, per hydrographic unit
               (b) FBI (discrete classes) of that CSI-winning product

  BOTTOM ROW - GPM family only (rawGPM/gwrGPM/expGPM - Rain4PE/PISCO excluded
               since they dominate the scale so completely that the GPM
               variants' real differences from each other are invisible
               whenever all 5 share a plot):
               (c) best-performing GPM variant by CSI, per hydrographic unit
               (d) FBI of that GPM-family CSI-winner, per hydrographic unit
               NOTE: bottom-row FBI bins are DIFFERENT from the top row -
               the GPM family's FBI values sit in a different, narrower
               range (see the printed range below) than the full
               comparison's winners, so reusing the top row's bins would
               likely make the bottom map look flat again. Check the
               printed range and retune bias_bins_gpm if needed.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point
try:
    import contextily as cx
    HAVE_CONTEXTILY = True
except ImportError:
    HAVE_CONTEXTILY = False

from configurations import PLOT_ORDER, DISPLAY_NAMES, PRODUCT_COLORS, PRODUCT_MARKERS

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
geoparquet_path = os.path.join(wkDir, 'outputs', 'station_metrics_1mm_clipped.parquet')
uh_shp_path = r'C:\Users\jvila\Desktop\Andean_project\gis\study_uhs\study_uhs.shp'
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

GPM_FAMILY = ['rawGPM', 'gwrGPM', 'expGPM']

# --- Toggle: show a minimal lat/lon coordinate box on each map? ---
# Kept OFF by default reasoning: turn on to try it, off to go back to the
# clean, label-only look. When on, only a few tick marks with degree labels
# appear on the left/bottom edges - no full grid, to avoid clutter.
SHOW_COORD_BOX = False

import math
from matplotlib.ticker import MaxNLocator, FuncFormatter

def _webmerc_to_lon(x, pos=None):
    return f"{x / 20037508.34 * 180:.2f}°"

def _webmerc_to_lat(y, pos=None):
    lat = math.degrees(2 * math.atan(math.exp(y / 20037508.34 * math.pi)) - math.pi / 2)
    return f"{lat:.2f}°"

def apply_coord_box(ax):
    """Minimal lat/lon tick labels on the left/bottom edges - no full grid,
    just enough to orient the reader without cluttering the map."""
    ax.set_axis_on()
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(_webmerc_to_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(_webmerc_to_lat))
    ax.tick_params(axis='both', labelsize=9, length=3, colors='dimgray')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('dimgray')
        ax.spines[spine].set_linewidth(0.6)


def find_label_position(polygon, clutter_points, grid_n=40):
    minx, miny, maxx, maxy = polygon.bounds
    xs = np.linspace(minx, maxx, grid_n)
    ys = np.linspace(miny, maxy, grid_n)
    best_point, best_dist = None, -1
    for x in xs:
        for y in ys:
            pt = Point(x, y)
            if polygon.contains(pt):
                d = min((pt.distance(cp) for cp in clutter_points), default=0)
                if d > best_dist:
                    best_dist, best_point = d, pt
    return best_point if best_point is not None else polygon.representative_point()


def place_all_labels(ax, uh_plot_web, clutter_points, text_fn, fontsize=11):
    placed_points = []
    for _, row in uh_plot_web.iterrows():
        avoid = clutter_points + placed_points
        label_pt = find_label_position(row.geometry, avoid)
        placed_points.append(label_pt)
        ax.annotate(text_fn(row), xy=(label_pt.x, label_pt.y), ha='center', va='center',
                   fontsize=fontsize, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor='gray'))


# =============================================================================
# 1. Load UH polygons, resolve N5-preferred/N4-fallback name; join stations
# =============================================================================
uh = gpd.read_file(uh_shp_path)
for col in ['NOMB_UH_N5', 'NOMB_UH_N4']:
    uh[col] = uh[col].replace('', np.nan)
uh['UH_NAME'] = uh['NOMB_UH_N5'].fillna(uh['NOMB_UH_N4'])

gdf = gpd.read_parquet(geoparquet_path)
gdf_uh_crs = gdf.to_crs(uh.crs)
joined = gpd.sjoin(gdf_uh_crs, uh[['UH_NAME', 'geometry']], how='left', predicate='within')

# =============================================================================
# 2. Per-UH, per-product CSI and FBI (all 5 - GPM-only subsets pulled from this)
# =============================================================================
csi_cols, fbi_cols = {}, {}
for code in PLOT_ORDER:
    sr = 1 - joined[f'far_{code}']
    pod = joined[f'pod_{code}']
    with np.errstate(divide='ignore', invalid='ignore'):
        csi_cols[code] = 1.0 / (1.0 / pod + 1.0 / sr - 1.0)
    fbi_cols[code] = joined[f'fbi_{code}']

csi_df = pd.DataFrame(csi_cols)
csi_df['UH_NAME'] = joined['UH_NAME'].values
fbi_df = pd.DataFrame(fbi_cols)
fbi_df['UH_NAME'] = joined['UH_NAME'].values

csi_by_uh = csi_df.groupby('UH_NAME').mean()
fbi_by_uh = fbi_df.groupby('UH_NAME').mean()
station_counts = joined.groupby('UH_NAME').size()

# Per-station CSI/FBI for texture on the maps (all 5 - subset per row later)
station_csi_cols = {}
for code in PLOT_ORDER:
    sr = 1 - gdf[f'far_{code}']
    pod = gdf[f'pod_{code}']
    with np.errstate(divide='ignore', invalid='ignore'):
        station_csi_cols[code] = 1.0 / (1.0 / pod + 1.0 / sr - 1.0)
station_csi_df = pd.DataFrame(station_csi_cols, index=gdf.index)

uh_all_web = uh.to_crs(epsg=3857)
stations_web = gdf.to_crs(epsg=3857)
station_points_list = list(stations_web.geometry.values)


def build_summary(product_codes):
    """Winner (by CSI) and that winner's FBI, per UH AND per station,
    restricted to `product_codes`."""
    sub_csi_by_uh = csi_by_uh[product_codes]
    winner_by_uh = sub_csi_by_uh.idxmax(axis=1)
    winner_fbi = pd.Series({u: fbi_by_uh.loc[u, winner_by_uh[u]] for u in winner_by_uh.index})
    summary = pd.DataFrame({
        'n_stations': station_counts,
        'winner': winner_by_uh.map(DISPLAY_NAMES),
        'winner_csi': sub_csi_by_uh.max(axis=1),
        'winner_fbi': winner_fbi,
    })

    sub_station_csi = station_csi_df[product_codes]
    station_winner = sub_station_csi.idxmax(axis=1)
    station_winner_fbi = pd.Series(
        [gdf.loc[i, f'fbi_{station_winner.loc[i]}'] for i in gdf.index], index=gdf.index)
    return summary, station_winner, station_winner_fbi


def draw_maps(fig, gs_csi, gs_fbi, product_codes, summary, station_winner, station_winner_fbi,
             bias_bins, bias_labels, title_csi, title_fbi):
    n_classes = len(bias_bins) - 1
    uh_plot = uh.merge(summary, left_on='UH_NAME', right_index=True, how='inner')
    uh_plot_web = uh_plot.to_crs(epsg=3857)
    uh_no_data_web = uh_all_web[~uh_all_web['UH_NAME'].isin(uh_plot_web['UH_NAME'])]
    sw = stations_web.copy()
    sw['station_winner'] = station_winner.values
    sw['station_winner_fbi'] = station_winner_fbi.values

    # --- CSI winner map ---
    ax = fig.add_subplot(gs_csi)
    uh_all_web.boundary.plot(ax=ax, color='gray', linewidth=0.8, zorder=1)
    if len(uh_no_data_web) > 0:
        uh_no_data_web.plot(ax=ax, facecolor='none', edgecolor='gray', hatch='///', linewidth=0.8, alpha=0.6, zorder=1)
    for code in product_codes:
        mask = uh_plot_web['winner'] == DISPLAY_NAMES[code]
        if mask.any():
            uh_plot_web[mask].plot(ax=ax, color=PRODUCT_COLORS[code], alpha=0.35, edgecolor='black', linewidth=0.8, zorder=2)
    for code in product_codes:
        mask = sw['station_winner'] == code
        sub = sw[mask]
        if len(sub) > 0:
            ax.scatter(sub.geometry.x, sub.geometry.y, color=PRODUCT_COLORS[code], marker=PRODUCT_MARKERS[code],
                      s=60, edgecolor='black', linewidth=0.5, zorder=5)
    place_all_labels(ax, uh_plot_web, station_points_list, lambda row: row['UH_NAME'], fontsize=11)

    # Reserve the SAME colorbar-sized strip as the FBI panel below, but leave
    # it invisible - done BEFORE add_basemap (same order as the FBI panel),
    # since contextily fetches tiles based on the axes' pixel size at call
    # time; doing this step at different points in each panel was producing
    # different effective zoom/extent between panels, not just a size illusion.
    divider_csi = make_axes_locatable(ax)
    cax_csi = divider_csi.append_axes('right', size='4%', pad=0.15)
    cax_csi.axis('off')

    if HAVE_CONTEXTILY:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=8)
        except Exception:
            pass
    if SHOW_COORD_BOX:
        apply_coord_box(ax)
    else:
        ax.set_axis_off()
    ax.set_title(title_csi, fontsize=16)
    handles = [mpatches.Patch(color=PRODUCT_COLORS[c], label=DISPLAY_NAMES[c], alpha=0.6) for c in product_codes]
    handles.append(mpatches.Patch(facecolor='none', edgecolor='gray', hatch='///', label='No station coverage'))
    ax.legend(handles=handles, loc='lower left', frameon=True, framealpha=0.9, fontsize=12, title='CSI winner', title_fontsize=12)

    # --- FBI map (discrete classes) ---
    ax = fig.add_subplot(gs_fbi)
    uh_all_web.boundary.plot(ax=ax, color='gray', linewidth=0.8, zorder=1)
    if len(uh_no_data_web) > 0:
        uh_no_data_web.plot(ax=ax, facecolor='none', edgecolor='gray', hatch='///', linewidth=0.8, alpha=0.6, zorder=1)
    cmap = plt.colormaps['RdBu_r'].resampled(n_classes + 2)
    class_norm = mcolors.BoundaryNorm(bias_bins, cmap.N, extend='both')
    uh_plot_web.plot(ax=ax, column='winner_fbi', cmap=cmap, norm=class_norm, edgecolor='black', linewidth=0.8, zorder=2)
    for code in product_codes:
        mask = sw['station_winner'] == code
        sub = sw[mask]
        if len(sub) > 0:
            ax.scatter(sub.geometry.x, sub.geometry.y, c=sub['station_winner_fbi'], cmap=cmap, norm=class_norm,
                      marker=PRODUCT_MARKERS[code], s=90, edgecolor='black', linewidth=0.8, zorder=5)
    place_all_labels(ax, uh_plot_web, station_points_list,
                    lambda row: f"{row['UH_NAME']}\nFBI={row['winner_fbi']:.2f}", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=class_norm)
    sm.set_array([])
    bin_centers = [(bias_bins[i] + bias_bins[i + 1]) / 2 for i in range(n_classes)]
    divider_fbi = make_axes_locatable(ax)
    cax_fbi = divider_fbi.append_axes('right', size='4%', pad=0.15)
    cbar = fig.colorbar(sm, cax=cax_fbi, extend='both')
    cbar.set_ticks(bin_centers)
    cbar.set_ticklabels(bias_labels, fontsize=9)
    cbar.set_label('FBI class', fontsize=11)
    if HAVE_CONTEXTILY:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=8)
        except Exception:
            pass
    if SHOW_COORD_BOX:
        apply_coord_box(ax)
    else:
        ax.set_axis_off()
    ax.set_title(title_fbi, fontsize=16)
    shape_handles = [plt.Line2D([0], [0], marker=PRODUCT_MARKERS[c], color='w', markerfacecolor='lightgray',
                    markeredgecolor='black', markersize=9, label=DISPLAY_NAMES[c]) for c in product_codes]
    shape_handles.append(mpatches.Patch(facecolor='none', edgecolor='gray', hatch='///', label='No station coverage'))
    ax.legend(handles=shape_handles, loc='lower left', frameon=True, framealpha=0.9, fontsize=12, title='Station winner (shape)', title_fontsize=12)


# =============================================================================
# 3. Build data for both rows
# =============================================================================
summary_full, station_winner_full, station_winner_fbi_full = build_summary(PLOT_ORDER)
print("--- Full comparison (5 products): per-UH winner and FBI ---")
print(summary_full.round(3).to_string())

summary_gpm, station_winner_gpm, station_winner_fbi_gpm = build_summary(GPM_FAMILY)
print("\n--- GPM-family-only: per-UH winner and FBI ---")
print(summary_gpm.round(3).to_string())
print(f"\nGPM-family winner_fbi range: {summary_gpm['winner_fbi'].min():.3f} to "
      f"{summary_gpm['winner_fbi'].max():.3f} -> check bias_bins_gpm below matches this")
print(f"GPM-family station-level winner FBI range: {station_winner_fbi_gpm.min():.3f} to "
      f"{station_winner_fbi_gpm.max():.3f}")

# =============================================================================
# 4. Bias bins - TOP row (full comparison) vs BOTTOM row (GPM family) use
#    DIFFERENT ranges, since their FBI values sit in different parts of the
#    scale. Adjust bias_bins_gpm based on the printed ranges above.
# =============================================================================
bias_bins_full = [0.5, 0.75, 0.88, 0.94, 1.00, 1.06, 1.15, 1.5]
bias_labels_full = ['0.50-0.75\nStrong under', '0.75-0.88', '0.88-0.94', '0.94-1.00\nNear balanced',
                    '1.00-1.06', '1.06-1.15', '1.15-1.50\nStrong over']

bias_bins_gpm = [0.4, 0.6, 0.72, 0.8, 0.88, 0.96, 1.1, 1.6]
bias_labels_gpm = ['0.40-0.60\nStrong under', '0.60-0.72', '0.72-0.80', '0.80-0.88\nNear balanced',
                   '0.88-0.96', '0.96-1.10', '1.10-1.60\nStrong over']

# =============================================================================
# 5. Combined figure: 1 row x 4 cols of MAPS (full comparison, then GPM family)
# =============================================================================
fig = plt.figure(figsize=(34, 11))
gs = fig.add_gridspec(1, 4, wspace=0.15)

draw_maps(fig, gs[0, 0], gs[0, 1], PLOT_ORDER, summary_full, station_winner_full, station_winner_fbi_full,
         bias_bins_full, bias_labels_full,
         '(a) Best-performing product by CSI - full comparison',
         '(b) FBI of the CSI-winning product - full comparison')

draw_maps(fig, gs[0, 2], gs[0, 3], GPM_FAMILY, summary_gpm, station_winner_gpm, station_winner_fbi_gpm,
         bias_bins_gpm, bias_labels_gpm,
         '(c) Best-performing product by CSI - GPM family only',
         '(d) FBI of the CSI-winning product - GPM family only')

fig.suptitle('CSI winner and FBI: full comparison (left) vs. GPM family only (right)', fontsize=21, y=1.03)

out_path = os.path.join(out_dir, 'combined_uh_maps_full_and_gpm.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved {out_path}")