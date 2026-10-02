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

Modified by: Jhon (added diagnostic printing block below `metrics_df` - no
             changes to the plotting logic itself) - prints median, mean,
             std, and IQR per product for each statistic, ranked by the
             metric's own "better" direction, so the paragraph can be
             rebuilt against actual values rather than the pre-correction
             ranking currently in the text (which has Rain4PE leading r/KGE
             and PISCO lowest - already shown elsewhere this session to be
             reversed: PISCO leads on every continuous metric).
"""


import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
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
# Direction for ranking: True = higher is better, False = lower is better.
# PBIAS is ranked by |PBIAS| (closest to zero), handled separately below.
HIGHER_IS_BETTER = {'r': True, 'rmse': False, 'kge': True}
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


# =============================================================================
# DIAGNOSTIC PRINTING - median/mean/std/IQR per product per statistic, ranked
# by each metric's own "better" direction. Checks every specific claim in
# the old paragraph: "Rain4PE highest r", "PISCO lowest r", "GWR outperforms
# EXP (r)", "Rain4PE and PISCO lowest RMSE", "Rain4PE and PISCO near-zero
# PBIAS, IMERGF underestimates, EXP overestimates", "KGE highest Rain4PE,
# substantial variability; GWR moderate; PISCO intermediate; EXP/IMERGF
# low and unstable".
# =============================================================================

print("="*95)
print("DIAGNOSTIC: median, mean, std, IQR per product, for every statistic")
print("Ranked by each metric's own 'better' direction (r, KGE: higher better;")
print("RMSE: lower better; PBIAS: ranked separately by |PBIAS|, closest to 0 = best)")
print("="*95)

summary_tables = {}
for stat, label in stat_labels.items():
    df_wide = metrics_df[[f'{stat}_{code}' for code in PLOT_ORDER]].rename(
        columns={f'{stat}_{code}': DISPLAY_NAMES[code] for code in PLOT_ORDER})

    summary = pd.DataFrame({
        'median': df_wide.median(),
        'mean': df_wide.mean(),
        'std': df_wide.std(),
        'Q1': df_wide.quantile(0.25),
        'Q3': df_wide.quantile(0.75),
    })
    summary['IQR'] = summary['Q3'] - summary['Q1']

    if stat == 'pbias':
        summary['abs_median'] = df_wide.abs().median()
        summary = summary.sort_values('abs_median')
    else:
        summary = summary.sort_values('median', ascending=not HIGHER_IS_BETTER[stat])

    summary_tables[stat] = summary
    print(f"\n--- {label} (ranked best to worst) ---")
    print(summary.round(3).to_string())

# =============================================================================
# Explicit ranking + "substantial variability / unstable" check (std-based)
# =============================================================================
print("\n" + "="*95)
print("RANKING SUMMARY (1st = best) and STD-based stability check")
print("(old text claims: 'Rain4PE KGE highest but substantial variability',")
print("'GPM-EXP/IMERGF low AND unstable KGE' - check std values directly)")
print("="*95)
for stat, label in stat_labels.items():
    summary = summary_tables[stat]
    print(f"\n{label}:")
    for rank, (product, row) in enumerate(summary.iterrows(), start=1):
        print(f"  {rank}. {product:12s} median={row['median']:+.3f}  std={row['std']:.3f}  "
              f"IQR={row['IQR']:.3f}")

# Direction of PBIAS specifically (over vs underestimation), since the old
# text makes directional claims ("IMERGF underestimated", "EXP overestimated")
print("\n" + "="*95)
print("PBIAS DIRECTION (sign), per product - checks 'IMERGF underestimates,")
print("EXP overestimates, Rain4PE/PISCO near-zero' claim directly")
print("="*95)
pbias_summary = summary_tables['pbias']
for product, row in pbias_summary.iterrows():
    direction = 'underestimates' if row['median'] < 0 else 'overestimates' if row['median'] > 0 else 'unbiased'
    print(f"  {product:12s} median PBIAS={row['median']:+.3f}%  -> {direction}")


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