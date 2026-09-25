# -*- coding: utf-8 -*-
"""
Same treatment as combined_uh_maps_full_and_gpm.py (CSI/FBI), applied to the
CONTINUOUS metrics r and |PBIAS| - the direct analog of "Spatial distribution
of r (a) and |PBIAS| (b)". r is the pattern/timing-correlation winner; |PBIAS|
of that SAME winning product shows whether it's also well-calibrated (low
|PBIAS|) or just good at timing while badly biased in total volume.

|PBIAS| is unsigned (a magnitude, not a ratio centered on 1 like FBI), so this
uses a SEQUENTIAL colormap with excellent->poor classes, not a diverging one.

4 map panels, 1x4: (a) r-winner full comparison, (b) |PBIAS| of that winner
full comparison, (c) r-winner GPM family only, (d) |PBIAS| of that winner
GPM family only.
"""
import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, FuncFormatter
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
SHOW_COORD_BOX = True


def _webmerc_to_lon(x, pos=None):
    return f"{x / 20037508.34 * 180:.2f}\u00b0"

def _webmerc_to_lat(y, pos=None):
    lat = math.degrees(2 * math.atan(math.exp(y / 20037508.34 * math.pi)) - math.pi / 2)
    return f"{lat:.2f}\u00b0"

def apply_coord_box(ax):
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
# 2. Per-UH, per-product r and |PBIAS|
# =============================================================================
r_cols, abs_pbias_cols = {}, {}
for code in PLOT_ORDER:
    r_cols[code] = joined[f'r_{code}']
    abs_pbias_cols[code] = joined[f'pbias_{code}'].abs()

r_df = pd.DataFrame(r_cols)
r_df['UH_NAME'] = joined['UH_NAME'].values
abs_pbias_df = pd.DataFrame(abs_pbias_cols)
abs_pbias_df['UH_NAME'] = joined['UH_NAME'].values

r_by_uh = r_df.groupby('UH_NAME').mean()
abs_pbias_by_uh = abs_pbias_df.groupby('UH_NAME').mean()
station_counts = joined.groupby('UH_NAME').size()

# Per-station r/|PBIAS| for texture on the maps
station_r_df = pd.DataFrame({code: gdf[f'r_{code}'] for code in PLOT_ORDER}, index=gdf.index)
station_abs_pbias_df = pd.DataFrame({code: gdf[f'pbias_{code}'].abs() for code in PLOT_ORDER}, index=gdf.index)

uh_all_web = uh.to_crs(epsg=3857)
stations_web = gdf.to_crs(epsg=3857)
station_points_list = list(stations_web.geometry.values)


def build_summary(product_codes):
    """Winner (by r) and that winner's |PBIAS|, per UH AND per station,
    restricted to `product_codes`."""
    sub_r_by_uh = r_by_uh[product_codes]
    winner_by_uh = sub_r_by_uh.idxmax(axis=1)
    winner_abs_pbias = pd.Series({u: abs_pbias_by_uh.loc[u, winner_by_uh[u]] for u in winner_by_uh.index})
    summary = pd.DataFrame({
        'n_stations': station_counts,
        'winner': winner_by_uh.map(DISPLAY_NAMES),
        'winner_r': sub_r_by_uh.max(axis=1),
        'winner_abs_pbias': winner_abs_pbias,
    })

    sub_station_r = station_r_df[product_codes]
    station_winner = sub_station_r.idxmax(axis=1)
    station_winner_abs_pbias = pd.Series(
        [station_abs_pbias_df.loc[i, station_winner.loc[i]] for i in gdf.index], index=gdf.index)
    return summary, station_winner, station_winner_abs_pbias


def draw_maps(fig, gs_r, gs_pbias, product_codes, summary, station_winner, station_winner_abs_pbias,
             bias_bins, bias_labels, title_r, title_pbias):
    n_classes = len(bias_bins) - 1
    uh_plot = uh.merge(summary, left_on='UH_NAME', right_index=True, how='inner')
    uh_plot_web = uh_plot.to_crs(epsg=3857)
    uh_no_data_web = uh_all_web[~uh_all_web['UH_NAME'].isin(uh_plot_web['UH_NAME'])]
    sw = stations_web.copy()
    sw['station_winner'] = station_winner.values
    sw['station_winner_abs_pbias'] = station_winner_abs_pbias.values

    # --- (r) winner map ---
    ax = fig.add_subplot(gs_r)
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
    if SHOW_COORD_BOX:
        apply_coord_box(ax)
    else:
        ax.set_axis_off()
    if HAVE_CONTEXTILY:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=8)
        except Exception:
            pass
    ax.set_title(title_r, fontsize=16)
    handles = [mpatches.Patch(color=PRODUCT_COLORS[c], label=DISPLAY_NAMES[c], alpha=0.6) for c in product_codes]
    handles.append(mpatches.Patch(facecolor='none', edgecolor='gray', hatch='///', label='No station coverage'))
    ax.legend(handles=handles, loc='lower left', frameon=True, framealpha=0.9, fontsize=12, title='r winner', title_fontsize=12)

    divider_r = make_axes_locatable(ax)
    cax_r = divider_r.append_axes('right', size='4%', pad=0.15)
    cax_r.axis('off')

    # --- |PBIAS| map (discrete classes, sequential colormap) ---
    ax = fig.add_subplot(gs_pbias)
    uh_all_web.boundary.plot(ax=ax, color='gray', linewidth=0.8, zorder=1)
    if len(uh_no_data_web) > 0:
        uh_no_data_web.plot(ax=ax, facecolor='none', edgecolor='gray', hatch='///', linewidth=0.8, alpha=0.6, zorder=1)
    cmap = plt.colormaps['YlOrRd'].resampled(n_classes + 1)  # +1: only 'max' extend needed (|PBIAS| >= 0)
    class_norm = mcolors.BoundaryNorm(bias_bins, cmap.N, extend='max')
    uh_plot_web.plot(ax=ax, column='winner_abs_pbias', cmap=cmap, norm=class_norm, edgecolor='black', linewidth=0.8, zorder=2)
    for code in product_codes:
        mask = sw['station_winner'] == code
        sub = sw[mask]
        if len(sub) > 0:
            ax.scatter(sub.geometry.x, sub.geometry.y, c=sub['station_winner_abs_pbias'], cmap=cmap, norm=class_norm,
                      marker=PRODUCT_MARKERS[code], s=90, edgecolor='black', linewidth=0.8, zorder=5)
    place_all_labels(ax, uh_plot_web, station_points_list,
                    lambda row: f"{row['UH_NAME']}\n|PBIAS|={row['winner_abs_pbias']:.1f}%", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=class_norm)
    sm.set_array([])
    bin_centers = [(bias_bins[i] + bias_bins[i + 1]) / 2 for i in range(n_classes)]
    divider_pbias = make_axes_locatable(ax)
    cax_pbias = divider_pbias.append_axes('right', size='4%', pad=0.15)
    cbar = fig.colorbar(sm, cax=cax_pbias, extend='max')
    cbar.set_ticks(bin_centers)
    cbar.set_ticklabels(bias_labels, fontsize=9)
    cbar.set_label('|PBIAS| class (%)', fontsize=11)
    if SHOW_COORD_BOX:
        apply_coord_box(ax)
    else:
        ax.set_axis_off()
    if HAVE_CONTEXTILY:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=8)
        except Exception:
            pass
    ax.set_title(title_pbias, fontsize=16)
    shape_handles = [plt.Line2D([0], [0], marker=PRODUCT_MARKERS[c], color='w', markerfacecolor='lightgray',
                    markeredgecolor='black', markersize=9, label=DISPLAY_NAMES[c]) for c in product_codes]
    shape_handles.append(mpatches.Patch(facecolor='none', edgecolor='gray', hatch='///', label='No station coverage'))
    ax.legend(handles=shape_handles, loc='lower left', frameon=True, framealpha=0.9, fontsize=12,
             title='Station winner (shape)', title_fontsize=12)


# =============================================================================
# 3. Build data for both rows
# =============================================================================
summary_full, station_winner_full, station_winner_abs_pbias_full = build_summary(PLOT_ORDER)
print("--- Full comparison (5 products): per-UH r-winner and its |PBIAS| ---")
print(summary_full.round(3).to_string())
print(f"\nFull-comparison winner |PBIAS| range: {summary_full['winner_abs_pbias'].min():.2f} to "
      f"{summary_full['winner_abs_pbias'].max():.2f}% -> check bias_bins_full below matches this")
print(f"Full-comparison station-level winner |PBIAS| range: "
      f"{station_winner_abs_pbias_full.min():.2f} to {station_winner_abs_pbias_full.max():.2f}%")

summary_gpm, station_winner_gpm, station_winner_abs_pbias_gpm = build_summary(GPM_FAMILY)
print("\n--- GPM-family-only: per-UH r-winner and its |PBIAS| ---")
print(summary_gpm.round(3).to_string())
print(f"\nGPM-family winner |PBIAS| range: {summary_gpm['winner_abs_pbias'].min():.2f} to "
      f"{summary_gpm['winner_abs_pbias'].max():.2f}% -> check bias_bins_gpm below matches this")
print(f"GPM-family station-level winner |PBIAS| range: "
      f"{station_winner_abs_pbias_gpm.min():.2f} to {station_winner_abs_pbias_gpm.max():.2f}%")

# =============================================================================
# 4. |PBIAS| bins - PLACEHOLDER, retune against the printed ranges above.
#    Sequential (0 = best), so only the TOP end needs an "extend" arrow.
# =============================================================================
bias_bins_full = [0, 5, 10, 20, 35, 50, 100]
bias_labels_full = ['0-5%\nExcellent', '5-10%\nGood', '10-20%\nFair', '20-35%\nPoor', '35-50%\nBad', '>50%\nVery bad']

bias_bins_gpm = [0, 10, 20, 35, 50, 75, 150]
bias_labels_gpm = ['0-10%\nExcellent', '10-20%\nGood', '20-35%\nFair', '35-50%\nPoor', '50-75%\nBad', '>75%\nVery bad']

# =============================================================================
# 5. Combined figure: 1 row x 4 cols of MAPS
# =============================================================================
fig = plt.figure(figsize=(34, 11))
gs = fig.add_gridspec(1, 4, wspace=0.15)

draw_maps(fig, gs[0, 0], gs[0, 1], PLOT_ORDER, summary_full, station_winner_full, station_winner_abs_pbias_full,
         bias_bins_full, bias_labels_full,
         '(a) Best-performing product by r - full comparison',
         '(b) |PBIAS| of the r-winning product - full comparison')

draw_maps(fig, gs[0, 2], gs[0, 3], GPM_FAMILY, summary_gpm, station_winner_gpm, station_winner_abs_pbias_gpm,
         bias_bins_gpm, bias_labels_gpm,
         '(c) Best-performing product by r - GPM family only',
         '(d) |PBIAS| of the r-winning product - GPM family only')

fig.suptitle('r winner and |PBIAS|: full comparison (left) vs. GPM family only (right)', fontsize=21, y=1.03)

out_path = os.path.join(out_dir, 'combined_uh_maps_r_pbias.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved {out_path}")