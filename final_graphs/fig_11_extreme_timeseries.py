"""
File: fig_11_extreme_timeseries.py
Author: Jose P. Teran
Github: jopator
Date: 2026-10-02
Description: Annual extreme indices at two stations of interest, 2005-2018
             Figure 11 of manuscript:
               (a) R20 at Intercuenca Alto Huallaga - Tananta (high R20)
               (b) CDD at Crisnejas - Sondor-Matara (high CDD)
             Legend values = Pearson r between each dataset's and the observed annual series.
             Annual indices are computed with Station._extreme_indices (precipitation_core) from the
             corrected station pickle.
"""


import os
import pickle
import sys
from pathlib import Path

repoDir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoDir))        # the pickle needs precipitation_core

import matplotlib.pyplot as plt
import pandas as pd
from configurations import (
    DISPLAY_NAMES,
    OBSERVED_COLOR,
    OBSERVED_LABEL,
    PLOT_ORDER,
    PRODUCT_COLORS,
)

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
wet_threshold = 1.0         # mm/day, same as the detection metrics

# (station, index, y-axis label)
panel_specs = [
    ('Intercuenca Alto Huallaga_ Tananta', 'r20', 'R20 (days)'),
    ('Crisnejas_ Sondor-Matara',           'cdd', 'CDD (days)'),
]
panel_labels = ['(a)', '(b)']
LEGEND_HEADROOM = 1.35      # y-axis top = max bar * this, leaves room for the legend

# Source attribute in the station objects -> (label, color); observed first
SOURCES = {'data': (OBSERVED_LABEL, OBSERVED_COLOR),
           **{code: (DISPLAY_NAMES[code], PRODUCT_COLORS[code]) for code in PLOT_ORDER}}

# --------------
# Load data
# --------------

with open(data_pklFN, "rb") as f:
    data_pkl = pickle.load(f)       # This has all the time series per stations per dataset

# --------------
# Figure 11 -> Yearly bars per dataset at two stations
# --------------

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for i, (ax, (station, idx, ylabel)) in enumerate(zip(axs, panel_specs)):
    df = pd.DataFrame({src: data_pkl[station]._extreme_indices(src, min_range, max_range, wet_threshold)[idx]
                       for src in SOURCES})

    df.plot(kind='bar', ax=ax, color=[color for _, color in SOURCES.values()], width=0.8, legend=False)

    labels = [OBSERVED_LABEL] + [f'{SOURCES[src][0]} (r={df[src].corr(df["data"]):.2f})' for src in PLOT_ORDER]
    ax.legend(ax.get_legend_handles_labels()[0], labels, loc='upper right', ncol=2, framealpha=0.9)

    ax.set_ylim(0, df.max().max() * LEGEND_HEADROOM)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Year' if i == len(axs) - 1 else '')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.annotate(panel_labels[i], xy=(0, 1), xycoords='axes fraction', xytext=(-40, 8), textcoords='offset points',
                fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')

fig.tight_layout()
fig.savefig(plotDir / 'fig11_extreme_timeseries.png', bbox_inches='tight')
plt.close(fig)
