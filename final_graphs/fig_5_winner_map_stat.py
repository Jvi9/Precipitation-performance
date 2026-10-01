"""
File: fig_5_winner_map_stat.py
Author: Jose P. Teran
Github: jopator
Date: 2026-10-02
Description: Winner maps with subbasin (UH) subdivision - continuous metrics
             Figure 5 of manuscript, 1x4 panels:
               (a) dataset with best r per station + dominant winner per subbasin
               (b) |PBIAS| of the winner (station and subbasin mean)
               (c), (d) same, restricted to GPM-based products
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

# |PBIAS| (%) classes, same for both product sets; values above 50 fall in the open '>50' class
PBIAS_BOUNDS = [0, 5, 10, 20, 35, 50]
_n_colors = len(PBIAS_BOUNDS)       # 5 classes + 1 'max' extension
PBIAS_CMAP = mcolors.ListedColormap(plt.get_cmap('YlOrRd')(np.linspace(0.05, 0.95, _n_colors)))
PBIAS_NORM = mcolors.BoundaryNorm(PBIAS_BOUNDS, _n_colors, extend='max')

ERROR_FILL_ALPHA = 0.8     # subbasin fill transparency on the |PBIAS| panels

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

# Subbasins: level-5 name, level-4 where level 5 is missing (Mantaro, Pachitea)
# Same lon/lat as the stations; set_crs instead of to_crs avoids a PROJ database error in pytj_313
uhs = gpd.read_file(uhsFN).set_crs(metrics_df.crs, allow_override=True)
uhs['basin'] = uhs['NOMB_UH_N5'].fillna(uhs['NOMB_UH_N4'])
stations = metrics_df.sjoin(uhs[['basin', 'geometry']], how='left', predicate='within')


# --------------
# Helpers
# --------------

def basin_summary(codes):
    """Winner by r (highest) and the |PBIAS| of that winner, per station and per subbasin.
    Per subbasin (as in Jhon's original): the product with the highest mean r over the subbasin's
    stations, and that product's mean |PBIAS| over the same stations."""
    st = stations[['basin', 'geometry']].copy()
    r_df = stations[[f'r_{c}' for c in codes]].set_axis(codes, axis=1)
    st['winner'] = r_df.idxmax(axis=1)
    pbias_df = stations[[f'pbias_{c}' for c in codes]].abs().set_axis(codes, axis=1)
    st['err'] = pbias_df.to_numpy()[np.arange(len(st)), [codes.index(w) for w in st['winner']]]

    covered = st['basin'].dropna()
    basin_winner = r_df.loc[covered.index].groupby(covered).mean().idxmax(axis=1)
    pbias_by_basin = pbias_df.loc[covered.index].groupby(covered).mean()
    basin_err = pd.Series({b: pbias_by_basin.loc[b, code] for b, code in basin_winner.items()})
    return st, basin_winner, basin_err


def draw_basins(ax, facecolors):
    """Fill subbasins from {basin: color}; subbasins without stations are hatched."""
    has = uhs['basin'].isin(facecolors.keys())
    uhs[has].plot(ax=ax, color=[facecolors[b] for b in uhs.loc[has, 'basin']], edgecolor='black', linewidth=0.8)
    uhs[~has].plot(ax=ax, facecolor='white', edgecolor='grey', hatch='////', linewidth=0.8)
    uhs.boundary.plot(ax=ax, color='black', linewidth=0.8)


def draw_stations(ax, st, codes, by_error=False):
    """Marker shape = winning product; color = product color, or |PBIAS| class if `by_error`."""
    for code in codes:
        sub = st[st['winner'] == code]
        if sub.empty:
            continue
        color = PBIAS_CMAP(PBIAS_NORM(sub['err'])) if by_error else PRODUCT_COLORS[code]
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
# Figure 5 -> 1x4 winner maps: (a, b) all products, (c, d) GPM-based products only
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
    ax_w.legend(handles=winner_handles(codes, filled=True, coverage=True), title='Dataset with best r',
                loc='lower left', frameon=True, framealpha=0.9)

    # |PBIAS| panel: subbasin and stations colored by |PBIAS| class of the winner
    draw_basins(ax_e, {b: mcolors.to_rgba(PBIAS_CMAP(PBIAS_NORM(v)), ERROR_FILL_ALPHA) for b, v in basin_err.items()})
    draw_stations(ax_e, st, codes, by_error=True)
    label_basins(ax_e, {b: f'{b}\n|PBIAS|={v:.1f}%' for b, v in basin_err.items()})
    ax_e.legend(handles=winner_handles(codes, filled=False, coverage=False), title='Best dataset on station (shape)',
                loc='lower left', frameon=True, framealpha=0.9)

    cax = ax_e.inset_axes([1.0, 0.3, 0.035, 0.45])
    cbar = fig.colorbar(ScalarMappable(norm=PBIAS_NORM, cmap=PBIAS_CMAP), cax=cax, extend='max')
    cbar.set_ticks(PBIAS_BOUNDS)
    cbar.set_label('|PBIAS| (%)')

for i, ax in enumerate(axs):
    ax.set_axis_off()
    add_north_arrow_and_scale(ax)
    ax.text(0.0, 1.0, panel_labels[i], transform=ax.transAxes, va='top',
            fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')

fig.tight_layout(w_pad=4)     # extra horizontal space between panels
fig.savefig(plotDir / 'fig5_winner_r_pbias.png', bbox_inches='tight')
plt.close(fig)
