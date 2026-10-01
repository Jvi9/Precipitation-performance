"""
File: fig_3_taylor.py
Author: Jose P. Teran
Github: jopator
Date: 2026-10-02
Description: Taylor diagram - GPM - NDVI ds datasets - PISCO - Rain4pe
             Figure 3 of manuscript.
             Daily series 2005-2018, standard deviation normalized by the observed one.
             Faint points = single stations, bold points = mean over stations.
"""


import os
import pickle
import sys
from pathlib import Path

repoDir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoDir))        # the pickle needs precipitation_core

import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as FA
import mpl_toolkits.axisartist.grid_finder as GF
import numpy as np
import pandas as pd
from configurations import DISPLAY_NAMES, PLOT_ORDER, PRODUCT_COLORS, PRODUCT_MARKERS
from matplotlib.projections.polar import PolarAxes

# Dirs
data_pklFN  = repoDir / 'datasets/data_locked_loaded_clipped_fixed.pkl'      # corrected data by Jhon
plotDir     = repoDir / 'graphs'
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

min_range = '2005-01-01'
max_range = '2018-12-31'


# --------------
# Helpers
# --------------

def prep_series(df, col, start, end):
    df = df.copy()
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    return df.loc[start:end, col]


class TaylorDiagram:
    """Quarter-circle Taylor diagram.
    Angle = correlation (arccos), radius = normalized standard deviation.
    """
    def __init__(self, fig, ref_std=1.0, srange=(0, 1.6), rect=111):
        self.ref_std = ref_std
        self.smin, self.smax = srange[0] * ref_std, srange[1] * ref_std
        tr = PolarAxes.PolarTransform()

        # Correlation ticks (nonlinear angular grid)
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        tlocs = np.arccos(rlocs)
        gl1 = GF.FixedLocator(tlocs)
        tf1 = GF.DictFormatter(dict(zip(tlocs, [f"{x:.2f}" for x in rlocs])))

        ghelper = FA.GridHelperCurveLinear(
            tr, extremes=(0, np.pi / 2, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        ax = fig.add_subplot(rect, axes_class=FA.FloatingAxes, grid_helper=ghelper)
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Normalized standard deviation")

        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")

        ax.axis["bottom"].set_visible(False)

        self._ax = ax
        self.ax = ax.get_aux_axes(tr)

        # Reference std arc and observed point
        t = np.linspace(0, np.pi / 2)
        self.ax.plot(t, np.full_like(t, self.ref_std), 'k--', linewidth=1, label='_nolegend_')
        self.ax.plot([0], [self.ref_std], 'k*', markersize=14, label='Observed')

    def add_sample(self, std_ratio, r, *args, **kwargs):
        theta = np.arccos(np.clip(r, -1, 1))
        return self.ax.plot(theta, std_ratio, *args, **kwargs)

    def add_rmse_contours(self, levels=5):
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax, 100),
                             np.linspace(0, np.pi / 2, 100))
        rms = np.sqrt(self.ref_std ** 2 + rs ** 2 - 2 * self.ref_std * rs * np.cos(ts))
        contours = self.ax.contour(ts, rs, rms, levels, colors='gray',
                                   linestyles='dotted', linewidths=0.8)
        self.ax.clabel(contours, inline=1, fontsize=8, fmt='%.1f')


# --------------
# Load data
# --------------

with open(data_pklFN, "rb") as f:
    data_pkl = pickle.load(f)       # This has all the time series per stations per dataset

# --------------
# Per-station correlation and std ratio (daily), for every product
# --------------

rows = []
for station_name, station in data_pkl.items():
    obs = prep_series(station.data, 'Precipitation', min_range, max_range)
    for prod in PLOT_ORDER:
        sim = prep_series(getattr(station, prod), 'precipitationCal', min_range, max_range)
        mask = obs.notna() & sim.notna()
        o, s = obs[mask], sim[mask]
        if len(o) < 30 or o.std() == 0:
            continue
        rows.append({'station': station_name, 'product': prod,
                     'r': s.corr(o), 'std_ratio': s.std() / o.std()})

taylor_stats_df = pd.DataFrame(rows)

# One point per product getting mean across stations
product_summary = (taylor_stats_df.groupby('product').agg(r=('r', 'mean'), std_ratio=('std_ratio', 'mean')).reindex(PLOT_ORDER))

# --------------
# Figure 3 -> Instead of violin plot, taylor diagram.
# --------------

fig = plt.figure(figsize=(9, 9))
taylor = TaylorDiagram(fig, ref_std=1.0, srange=(0, 1.6))
taylor.add_rmse_contours(levels=5)

# Faint points - station level
for prod in PLOT_ORDER:
    sub = taylor_stats_df[taylor_stats_df['product'] == prod]
    theta = np.arccos(np.clip(sub['r'], -1, 1))
    taylor.ax.scatter(theta, sub['std_ratio'], marker=PRODUCT_MARKERS[prod], color=PRODUCT_COLORS[prod],
                      alpha=0.15, s=25, zorder=2)

# Bold points - mean
for prod in PLOT_ORDER:
    row = product_summary.loc[prod]
    taylor.add_sample(row['std_ratio'], row['r'], marker=PRODUCT_MARKERS[prod], markersize=10,
                      color=PRODUCT_COLORS[prod], markeredgecolor='black',
                      markeredgewidth=1, label=DISPLAY_NAMES[prod], linestyle='none', zorder=5)

fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.85))

fig.savefig(plotDir / 'fig3_taylor_diagram.png', bbox_inches='tight')
plt.close(fig)
