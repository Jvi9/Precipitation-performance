"""
File: fig_4_boxplot_stat.py
Author: Jose P. Teran
Github: jopator
Date: 2026-10-02
Description: Boxen plots of daily performance statistics across all datasets
             Figure 4 of manuscript: r (a), RMSE (b), KGE (c), PBIAS (d).
             A boxenplot shows the median, then the IQR (25th-75th) as the first box,
             then successively narrower boxes halving the remaining tail
             (12.5th-87.5th, 6.25th-93.75th, ...) ("letter-value plot", Hofmann/Wickham/Kafadar 2011).
"""


import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from configurations import DISPLAY_NAMES, PLOT_ORDER, PRODUCT_COLORS

repoDir = Path(__file__).resolve().parents[1]

# Dirs
metrics_parquetFN   = repoDir / 'outputs/station_metrics_1mm_clipped.parquet'        # parquet with metrics
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

display_order = [DISPLAY_NAMES[c] for c in PLOT_ORDER]
palette = {DISPLAY_NAMES[c]: PRODUCT_COLORS[c] for c in PLOT_ORDER}

stat_labels = {
    'r':     'r',
    'rmse':  'RMSE (mm day$^{-1}$)',
    'kge':   'KGE',
    'pbias': 'PBIAS (%)',
}
panel_labels = ['(a)', '(b)', '(c)', '(d)']

# --------------
# Load data
# --------------

metrics_df = gpd.read_parquet(metrics_parquetFN)


def metric_long(stat):
    """Wide parquet columns ({stat}_{code}) -> long df with Dataset / Statistic columns."""
    cols = {f'{stat}_{code}': DISPLAY_NAMES[code] for code in PLOT_ORDER}
    return (metrics_df[list(cols)].rename(columns=cols)
            .melt(var_name='Dataset', value_name='Statistic'))


# --------------
# Figure 4 -> Boxen plots. No significant change. Just clearer fonts, color scheme and consistent higher dpi.
# --------------

fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for i, (ax, stat_name) in enumerate(zip(axs.ravel(), stat_labels)):
    sns.boxenplot(x='Dataset', y='Statistic', hue='Dataset',
                  data=metric_long(stat_name), palette=palette,
                  order=display_order, hue_order=display_order, legend=False, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(stat_labels[stat_name])
    ax.grid(True, alpha=0.3)
    ax.text(-0.1, 1.05, panel_labels[i], transform=ax.transAxes,
            fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')

fig.tight_layout()
fig.savefig(plotDir / 'fig4_continuous_metrics.png', bbox_inches='tight')
plt.close(fig)
