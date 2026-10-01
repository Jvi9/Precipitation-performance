"""
File: fig_10_extreme_indices.py
Author: Jose P. Teran
Github: jopator
Date: 2026-10-02
Description: ETCCDI extreme indices summary (CDD, CWD, R10, R20, R95p, R99p), 2005-2018
             Figure 10 of manuscript, 1x3 heatmaps (rows = datasets, columns = indices):
               (a) mean of the annual index, averaged over stations
               (b) standard deviation of the annual index, averaged over stations
               (c) Pearson R between the dataset's and the observed annual series, averaged over stations
             Annual indices are computed with Station._extreme_indices (precipitation_core) from the
             corrected station pickle. Colors in (a) and (b) are scaled separately for the
             day-count indices (CDD, CWD, R10, R20) and the mm indices (R95p, R99p).
"""


import os
import pickle
import sys
from pathlib import Path

repoDir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoDir))        # the pickle needs precipitation_core

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from configurations import DISPLAY_NAMES, OBSERVED_LABEL, PLOT_ORDER

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

INDICES = ['cdd', 'cwd', 'r10', 'r20', 'r95p', 'r99p']
INDEX_LABELS = {'cdd': 'CDD', 'cwd': 'CWD', 'r10': 'R10', 'r20': 'R20', 'r95p': 'R95p', 'r99p': 'R99p'}
DAYS_IDX = ['cdd', 'cwd', 'r10', 'r20']     # days
MM_IDX = ['r95p', 'r99p']                   # mm

# Source attribute in the station objects -> row label
SOURCES = {'data': OBSERVED_LABEL, **{code: DISPLAY_NAMES[code] for code in PLOT_ORDER}}

# (statistic, colormap, number format)
panel_specs = [
    ('mean', 'Blues',   '.0f'),
    ('std',  'Purples', '.0f'),
    ('R',    'RdYlGn',  '.2f'),
]
panel_labels = ['(a)', '(b)', '(c)']
R_VMIN, R_VMAX = 0.0, 0.7

# --------------
# Load data
# --------------

with open(data_pklFN, "rb") as f:
    data_pkl = pickle.load(f)       # This has all the time series per stations per dataset

# --------------
# Annual extreme indices per source and station: {source: {station: {index: Series(year)}}}
# --------------

extreme = {src: {name: station._extreme_indices(src, min_range, max_range, wet_threshold)
                 for name, station in data_pkl.items()}
           for src in SOURCES}

# Per station and index: mean and std over years, and R against the observed annual series
rows = []
for src, label in SOURCES.items():
    for name, indices in extreme[src].items():
        for idx in INDICES:
            series = indices[idx]
            R = float('nan')
            if src != 'data':
                both = pd.concat([series, extreme['data'][name][idx]], axis=1).dropna()
                if len(both) > 1:
                    R = both.iloc[:, 0].corr(both.iloc[:, 1])
            rows.append({'dataset': label, 'station': name, 'index': idx,
                         'mean': series.mean(), 'std': series.std(), 'R': R})

agg_df = pd.DataFrame(rows).groupby(['dataset', 'index'])[['mean', 'std', 'R']].mean().reset_index()

# Rows sorted by mean R over the indices (observed has no R and goes last)
order = agg_df.groupby('dataset')['R'].mean().sort_values(ascending=False).index


def group_normalized(h):
    """Scale to 0-1 separately for the day-count and the mm indices, so both get the full color range."""
    h_norm = pd.DataFrame(index=h.index, columns=h.columns, dtype=float)
    for cols in (DAYS_IDX, MM_IDX):
        vmin, vmax = h[cols].min().min(), h[cols].max().max()
        h_norm[cols] = (h[cols] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    return h_norm


# --------------
# Figure 10 -> Heatmaps of mean, std and R of the extreme indices
# --------------

fig, axs = plt.subplots(1, 3, figsize=(16, 5))
annot_kws = {'fontsize': plt.rcParams['legend.fontsize'], 'fontweight': 'bold'}

for i, (ax, (stat, cmap_name, fmt)) in enumerate(zip(axs, panel_specs)):
    h = agg_df.pivot(index='dataset', columns='index', values=stat).loc[order, INDICES]
    cmap = matplotlib.colormaps[cmap_name].copy()

    if stat == 'R':
        cmap.set_bad('lightgray')       # observed row (no R) in gray
        sns.heatmap(h, annot=True, fmt=fmt, cmap=cmap, vmin=R_VMIN, vmax=R_VMAX, cbar=False,
                    annot_kws=annot_kws, ax=ax)
    else:
        sns.heatmap(group_normalized(h), annot=h, fmt=fmt, cmap=cmap, vmin=0, vmax=1, cbar=False,
                    annot_kws=annot_kws, ax=ax)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([INDEX_LABELS[c] for c in h.columns], rotation=0)
    ax.set_yticklabels(h.index, rotation=0)
    ax.text(-0.1, 1.05, panel_labels[i], transform=ax.transAxes,
            fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')

fig.tight_layout(w_pad=3)
fig.savefig(plotDir / 'fig10_extreme_indices.png', bbox_inches='tight')
plt.close(fig)
