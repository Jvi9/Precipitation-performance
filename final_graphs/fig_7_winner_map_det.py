"""
File: fig_7_winner_map_det.py
Author: Jose P. Teran
Github: jopator
Date: 2026-10-02
Description: Winner maps with subbasin (UH) subdivision - categorical (detection) metrics
             Figure 7 of manuscript, 1x4 panels:
               (a) dataset with best CSI per station + dominant winner per subbasin
               (b) FBI of the winner (station and subbasin mean)
               (c), (d) same, restricted to GPM-based products
             CSI is derived from POD and FAR.

Modified by: Jhon (added diagnostic printing block below `stations` - no
             changes to the plotting logic itself) - prints everything
             needed to check and rebuild "Forecasting indices and detection
             trade-offs across datasets" against the corrected data. The
             OLD paragraph claimed:
               - Rain4PE AND PISCO both: high CSI, FBI close to unity
               - GPM-GWR: localized CSI improvement, moderates FBI toward
                 unity, "advantage of incorporating terrain information"
               - GPM-EXP: uneven CSI, FBI "well above unity" (overestimation)
             Given the network-mean FBI values already found this session
             (Rain4PE=1.178 overestimating, PISCO=0.938 near-unity,
             GPM-EXP=0.831 UNDERestimating, not "well above unity"), several
             of these claims are suspect before even checking basin-level detail.
"""


import os
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from configurations import DISPLAY_NAMES, PLOT_ORDER, PRODUCT_COLORS, PRODUCT_MARKERS
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

repoDir = Path(__file__).resolve().parents[1]

# Dirs
metrics_parquetFN   = repoDir / 'outputs/station_metrics_1mm_clipped.parquet'        # parquet with metrics
uhsFN               = repoDir / 'gis/study_uhs/study_uhs.shp'                        # subbasins (UHs)
plotDir             = repoDir / 'graphs'
os.makedirs(plotDir, exist_ok=True)

# --------------
# Plot settings
# --------------

plt.rcParams.update({
    'xtick.labelsize':  10,     # axis tick labels
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'axes.labelsize':   12,     # axis labels
    'axes.titlesize':   14,     # subplot titles
    'figure.titlesize': 14,     # suptitle
    'savefig.dpi':      300,
})

GPM_FAMILY = ['rawGPM', 'gwrGPM', 'expGPM']
panel_labels = ['(a)', '(b)', '(c)', '(d)']
map_fontsize = plt.rcParams['legend.fontsize']

# FBI classes, same for both product sets: 1 is unbiased; <1 underestimates rain-event
# frequency (brown), >1 overestimates (green). Open classes below 0.5 and above 1.5
FBI_BOUNDS = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
_n_colors = len(FBI_BOUNDS) + 1     # 5 classes + 'min' and 'max' extensions
FBI_CMAP = mcolors.ListedColormap(plt.get_cmap('BrBG')(np.linspace(0.05, 0.95, _n_colors)))
FBI_NORM = mcolors.BoundaryNorm(FBI_BOUNDS, _n_colors, extend='both')

ERROR_FILL_ALPHA = 0.8     # subbasin fill transparency on the FBI panels

# Subbasin label positions (axes fraction), placed outside the subbasins with a leader line
LABEL_POS = {
    'Alto Marañón IV': (0.56, 0.96),
    'Crisnejas':       (0.06, 0.67),
    'Alto Marañón V':  (0.10, 0.55),
    'Alto Huallaga':   (0.20, 0.43),
    'Pachitea':        (0.68, 0.68),
    'Perené':          (0.28, 0.34),
    'Mantaro':         (1.00, 0.12),     # right of the basin, clear of the legend and below the colorbar
}

no_coverage_patch = mpatches.Patch(facecolor='white', edgecolor='grey', hatch='////', label='No station coverage')

# --------------
# Load data
# --------------

metrics_df = gpd.read_parquet(metrics_parquetFN)

# CSI is not in the parquet, derived from POD and FAR: CSI = 1 / (1/POD + 1/(1-FAR) - 1)
for code in PLOT_ORDER:
    pod, far = metrics_df[f'pod_{code}'], metrics_df[f'far_{code}']
    with np.errstate(divide='ignore', invalid='ignore'):
        metrics_df[f'csi_{code}'] = 1 / (1 / pod + 1 / (1 - far) - 1)

# Subbasins: level-5 name, level-4 where level 5 is missing (Mantaro, Pachitea)
# Same lon/lat as the stations; set_crs instead of to_crs avoids a PROJ database error in pytj_313
uhs = gpd.read_file(uhsFN).set_crs(metrics_df.crs, allow_override=True)
uhs['basin'] = uhs['NOMB_UH_N5'].fillna(uhs['NOMB_UH_N4'])
stations = metrics_df.sjoin(uhs[['basin', 'geometry']], how='left', predicate='within')

# =============================================================================
# DIAGNOSTIC PRINTING - everything needed to check/rebuild "Forecasting
# indices and detection trade-offs across datasets" against the corrected
# data. Checks, specifically:
#   - Rain4PE AND PISCO: "consistently high CSI" + "balanced FBI close to
#     unity" for BOTH - printed per-basin FBI for each shows whether this
#     holds for both or just one
#   - GPM-GWR: "localized improvement in CSI" + "moderating FBI towards
#     unity" relative to GPM-IMERGF/EXP
#   - GPM-EXP: "FBI well above unity" (overestimation) - network-mean FBI
#     was already found to be 0.831 (UNDERestimation) earlier this session,
#     so this claim is checked directly, basin by basin
# =============================================================================

def full_basin_table(codes, metric_prefix):
    """Per-basin mean of f'{metric_prefix}_{code}' for every code."""
    covered = stations['basin'].dropna()
    df = stations[[f'{metric_prefix}_{c}' for c in codes]].set_axis(codes, axis=1)
    by_basin = df.loc[covered.index].groupby(covered).mean()
    return by_basin.rename(columns=DISPLAY_NAMES)


print("="*95)
print("DIAGNOSTIC 1: per-basin mean CSI, ALL 5 PRODUCTS")
print("="*95)
csi_by_basin_full = full_basin_table(PLOT_ORDER, 'csi')
print(csi_by_basin_full.round(3).to_string())

winner_full = csi_by_basin_full.idxmax(axis=1)
margin_full = csi_by_basin_full.apply(lambda row: sorted(row.values)[-1] - sorted(row.values)[-2], axis=1)
runner_up_full = pd.Series(
    [csi_by_basin_full.columns[np.argsort(csi_by_basin_full.loc[b].values)[-2]] for b in csi_by_basin_full.index],
    index=csi_by_basin_full.index)
print("\n--- Winner (by CSI) and margin over runner-up, per basin ---")
for b in csi_by_basin_full.index:
    print(f"  {b:20s} winner={winner_full[b]:10s} CSI={csi_by_basin_full.loc[b, winner_full[b]]:.3f}  "
          f"runner_up={runner_up_full[b]:10s}  margin={margin_full[b]:.3f}")

print("\n" + "="*95)
print("DIAGNOSTIC 2: per-basin mean FBI, ALL 5 PRODUCTS")
print("(check 'Rain4PE AND PISCO both balanced FBI close to unity' claim -")
print("does this hold for BOTH, or just one of them, in every basin?)")
print("="*95)
fbi_by_basin_full = full_basin_table(PLOT_ORDER, 'fbi')
print(fbi_by_basin_full.round(3).to_string())

print("\n" + "="*95)
print("DIAGNOSTIC 3: station-level CSI-winner counts, ALL 5 PRODUCTS")
print("="*95)
csi_df_full = stations[[f'csi_{c}' for c in PLOT_ORDER]].set_axis(PLOT_ORDER, axis=1)
station_winner_full = csi_df_full.idxmax(axis=1).map(DISPLAY_NAMES)
print(station_winner_full.value_counts().to_string())

# --- Same diagnostics, restricted to the GPM family ---
print("\n" + "="*95)
print("DIAGNOSTIC 4: per-basin mean CSI, GPM FAMILY ONLY")
print("(check 'GPM-GWR showed localized improvement in CSI' claim - does GWR")
print("actually win any basin among the GPM family, and by how much?)")
print("="*95)
csi_by_basin_gpm = full_basin_table(GPM_FAMILY, 'csi')
print(csi_by_basin_gpm.round(3).to_string())
winner_gpm = csi_by_basin_gpm.idxmax(axis=1)
margin_gpm = csi_by_basin_gpm.apply(lambda row: sorted(row.values)[-1] - sorted(row.values)[-2], axis=1)
print("\n--- Winner (by CSI) and margin, per basin, GPM family only ---")
for b in csi_by_basin_gpm.index:
    print(f"  {b:20s} winner={winner_gpm[b]:12s} CSI={csi_by_basin_gpm.loc[b, winner_gpm[b]]:.3f}  "
          f"margin_over_runner_up={margin_gpm[b]:.3f}")

print("\n" + "="*95)
print("DIAGNOSTIC 5: per-basin mean FBI, GPM FAMILY ONLY")
print("(direct check of 'GPM-GWR moderates FBI towards unity' AND 'GPM-EXP FBI")
print("well above unity' claims - both tested basin by basin here)")
print("="*95)
fbi_by_basin_gpm = full_basin_table(GPM_FAMILY, 'fbi')
print(fbi_by_basin_gpm.round(3).to_string())

print("\n--- GPM-EXP's FBI specifically, sorted descending (checking 'well above unity') ---")
print(fbi_by_basin_gpm['GPM-EXP'].sort_values(ascending=False).round(3).to_string())

print("\n" + "="*95)
print("DIAGNOSTIC 6: station-level CSI-winner counts, GPM FAMILY ONLY")
print("="*95)
csi_df_gpm = stations[[f'csi_{c}' for c in GPM_FAMILY]].set_axis(GPM_FAMILY, axis=1)
station_winner_gpm = csi_df_gpm.idxmax(axis=1).map(DISPLAY_NAMES)
print(station_winner_gpm.value_counts().to_string())

print("\n" + "="*95)
print("DIAGNOSTIC 7: |FBI - 1| (distance from unity) per basin, ALL 5 PRODUCTS")
print("(the most direct test of 'balanced/close to unity' - ranks who is")
print("ACTUALLY closest to unbiased detection frequency, basin by basin)")
print("="*95)
dist_from_unity = (fbi_by_basin_full - 1).abs()
print(dist_from_unity.round(3).to_string())
closest_to_unity = dist_from_unity.idxmin(axis=1)
print("\n--- Closest-to-unity-FBI product per basin ---")
print(closest_to_unity.to_string())


# --------------
# Helpers
# --------------

def basin_summary(codes):
    """Winner by CSI (highest) and the FBI of that winner, per station and per subbasin.
    Per subbasin (as in Jhon's original): the product with the highest mean CSI over the subbasin's
    stations, and that product's mean FBI over the same stations."""
    st = stations[['basin', 'geometry']].copy()
    csi_df = stations[[f'csi_{c}' for c in codes]].set_axis(codes, axis=1)
    st['winner'] = csi_df.idxmax(axis=1)
    fbi_df = stations[[f'fbi_{c}' for c in codes]].set_axis(codes, axis=1)
    st['err'] = fbi_df.to_numpy()[np.arange(len(st)), [codes.index(w) for w in st['winner']]]

    covered = st['basin'].dropna()
    basin_winner = csi_df.loc[covered.index].groupby(covered).mean().idxmax(axis=1)
    fbi_by_basin = fbi_df.loc[covered.index].groupby(covered).mean()
    basin_err = pd.Series({b: fbi_by_basin.loc[b, code] for b, code in basin_winner.items()})
    return st, basin_winner, basin_err


def draw_basins(ax, facecolors):
    """Fill subbasins from {basin: color}; subbasins without stations are hatched."""
    has = uhs['basin'].isin(facecolors.keys())
    uhs[has].plot(ax=ax, color=[facecolors[b] for b in uhs.loc[has, 'basin']], edgecolor='black', linewidth=0.8)
    uhs[~has].plot(ax=ax, facecolor='white', edgecolor='grey', hatch='////', linewidth=0.8)
    uhs.boundary.plot(ax=ax, color='black', linewidth=0.8)


def draw_stations(ax, st, codes, by_error=False):
    """Marker shape = winning product; color = product color, or FBI class if `by_error`."""
    for code in codes:
        sub = st[st['winner'] == code]
        if sub.empty:
            continue
        color = FBI_CMAP(FBI_NORM(sub['err'])) if by_error else PRODUCT_COLORS[code]
        ax.scatter(sub.geometry.x, sub.geometry.y, marker=PRODUCT_MARKERS[code], c=color,
                   s=45, edgecolors='black', linewidths=0.6, zorder=5)


def label_basins(ax, texts):
    for _, row in uhs[uhs['basin'].isin(texts.keys())].iterrows():
        pt = row.geometry.representative_point()
        ax.annotate(texts[row['basin']], (pt.x, pt.y), xytext=LABEL_POS[row['basin']],
                    textcoords='axes fraction', ha='center', va='center',
                    fontsize=map_fontsize, fontweight='bold', zorder=6, annotation_clip=False,
                    bbox={'boxstyle': 'round,pad=0.2', 'facecolor': 'white', 'edgecolor': 'grey', 'alpha': 0.85},
                    arrowprops={'arrowstyle': '-', 'color': 'black', 'linewidth': 0.7, 'shrinkA': 0, 'shrinkB': 0})
        ax.plot(pt.x, pt.y, marker='o', markersize=3, color='black', zorder=6)


def winner_handles(codes, filled, coverage):
    """Legend handles: one marker per product, plus the hatched no-coverage patch if `coverage`."""
    return [Line2D([], [], marker=PRODUCT_MARKERS[code], linestyle='none', markersize=8,
                   markerfacecolor=PRODUCT_COLORS[code] if filled else 'white',
                   markeredgecolor='black', label=DISPLAY_NAMES[code])
            for code in codes] + ([no_coverage_patch] if coverage else [])


def add_north_arrow_and_scale(ax, x=0.75, y=0.80, length_km=100):
    """QGIS-style north arrow (half black / half white) above a scale bar, both centered on
    (x, y) in axes fraction."""
    # Arrow drawn in inches from its base point, so its shape is independent of the map aspect
    trans = ax.figure.dpi_scale_trans + mtransforms.ScaledTranslation(x, y + 0.01, ax.transAxes)
    h, w, notch = 0.5, 0.15, 0.12
    tip, left, mid, right = (0, h), (-w, 0), (0, notch), (w, 0)
    ax.add_patch(mpatches.Polygon([tip, left, mid], closed=True, facecolor='black', edgecolor='black',
                                  linewidth=1, transform=trans, zorder=7))
    ax.add_patch(mpatches.Polygon([tip, mid, right], closed=True, facecolor='white', edgecolor='black',
                                  linewidth=1, transform=trans, zorder=7))
    ax.text(0, h + 0.05, 'N', transform=trans, ha='center', va='bottom',
            fontsize=plt.rcParams['axes.labelsize'], fontweight='bold', zorder=7)

    # Data is in lon/lat: convert km to degrees of longitude at the map's mid-latitude
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = length_km / (111.32 * np.cos(np.radians((y0 + y1) / 2)))
    xc = x0 + x * (x1 - x0)
    ys = y0 + (y - 0.05) * (y1 - y0)
    ax.plot([xc - dx / 2, xc + dx / 2], [ys, ys], color='black', linewidth=3, solid_capstyle='butt')
    ax.text(xc, ys + 0.01 * (y1 - y0), f'{length_km} km', ha='center', va='bottom',
            fontsize=map_fontsize)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


# --------------
# Figure 7 -> 1x4 winner maps: (a, b) all products, (c, d) GPM-based products only
# --------------

fig, axs = plt.subplots(1, 4, figsize=(22, 8))

for pair, codes in enumerate([PLOT_ORDER, GPM_FAMILY]):
    st, basin_winner, basin_err = basin_summary(codes)
    basin_err = basin_err.dropna()
    ax_w, ax_e = axs[2 * pair], axs[2 * pair + 1]

    # Winner panel: subbasin tinted by its winning product, stations by winning product
    draw_basins(ax_w, {b: mcolors.to_rgba(PRODUCT_COLORS[code], 0.35) for b, code in basin_winner.items()})
    draw_stations(ax_w, st, codes)
    label_basins(ax_w, {b: b for b in basin_winner.index})
    ax_w.legend(handles=winner_handles(codes, filled=True, coverage=True), title='Dataset with best CSI',
                loc='lower left', frameon=True, framealpha=0.9)

    # FBI panel: subbasin and stations colored by FBI class of the winner
    draw_basins(ax_e, {b: mcolors.to_rgba(FBI_CMAP(FBI_NORM(v)), ERROR_FILL_ALPHA) for b, v in basin_err.items()})
    draw_stations(ax_e, st, codes, by_error=True)
    label_basins(ax_e, {b: f'{b}\nFBI={v:.2f}' for b, v in basin_err.items()})
    ax_e.legend(handles=winner_handles(codes, filled=False, coverage=False), title='Best dataset on station (shape)',
                loc='lower left', frameon=True, framealpha=0.9)

    cax = ax_e.inset_axes([1.0, 0.3, 0.035, 0.45])
    cbar = fig.colorbar(ScalarMappable(norm=FBI_NORM, cmap=FBI_CMAP), cax=cax, extend='both')
    cbar.set_ticks(FBI_BOUNDS)
    cbar.set_label('FBI')

for i, ax in enumerate(axs):
    ax.set_axis_off()
    add_north_arrow_and_scale(ax)
    ax.text(0.0, 1.0, panel_labels[i], transform=ax.transAxes, va='top',
            fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')

fig.tight_layout(w_pad=4)     # extra horizontal space between panels
fig.savefig(plotDir / 'fig7_winner_csi_fbi.png', bbox_inches='tight')
plt.close(fig)