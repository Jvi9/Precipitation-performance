"""
File: fig_6_boxplot.py
Author: Jose P. Teran
Github: jopator
Date: 2026-10-02
Description: Boxen plots of categorical (detection) statistics across all datasets
             Figure 6 of manuscript: CSI (a), FBI (b), FAR (c), POD (d).
             A boxenplot shows the median, then the IQR (25th-75th) as the first box,
             then successively narrower boxes halving the remaining tail
             (12.5th-87.5th, 6.25th-93.75th, ...) ("letter-value plot", Hofmann/Wickham/Kafadar 2011).
             CSI is derived from POD and FAR; FBI, FAR and POD are stored GeoParquet columns.

Modified by: Jhon (added diagnostic printing block below CSI derivation -
             no changes to the plotting logic itself) - prints median,
             mean, std, IQR per product for CSI/FBI/FAR/POD, ranked by each
             metric's own "better" direction (FBI ranked by |FBI-1|,
             closest to unity = best, same treatment as PBIAS in Figure 4).
             Checks every specific claim in the old paragraph: "Rain4PE
             highest CSI, minimal variability", "PISCO lowest CSI",
             "Rain4PE lowest FAR, PISCO highest FAR", "Rain4PE highest POD",
             "PISCO and GPM-GWR concentrate near FBI=1".
"""


import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
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
    'csi': 'CSI',
    'fbi': 'FBI',
    'far': 'FAR',
    'pod': 'POD',
}
# Direction for ranking: True = higher is better, False = lower is better.
# FBI is ranked by |FBI-1| (closest to unity), handled separately below.
HIGHER_IS_BETTER = {'csi': True, 'far': False, 'pod': True}
panel_labels = ['(a)', '(b)', '(c)', '(d)']

# --------------
# Load data
# --------------

metrics_df = gpd.read_parquet(metrics_parquetFN)

# CSI is not in the parquet, derived from POD and FAR: CSI = 1 / (1/POD + 1/(1-FAR) - 1)
for code in PLOT_ORDER:
    pod, far = metrics_df[f'pod_{code}'], metrics_df[f'far_{code}']
    with np.errstate(divide='ignore', invalid='ignore'):
        metrics_df[f'csi_{code}'] = 1 / (1 / pod + 1 / (1 - far) - 1)


def metric_long(stat):
    """Wide parquet columns ({stat}_{code}) -> long df with Dataset / Statistic columns."""
    cols = {f'{stat}_{code}': DISPLAY_NAMES[code] for code in PLOT_ORDER}
    return (metrics_df[list(cols)].rename(columns=cols)
            .melt(var_name='Dataset', value_name='Statistic'))


# =============================================================================
# DIAGNOSTIC PRINTING - median/mean/std/IQR per product per statistic, ranked
# by each metric's own "better" direction.
# =============================================================================

print("="*95)
print("DIAGNOSTIC: median, mean, std, IQR per product, for every detection statistic")
print("Ranked by each metric's own 'better' direction (CSI, POD: higher better;")
print("FAR: lower better; FBI: ranked separately by |FBI-1|, closest to unity = best)")
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

    if stat == 'fbi':
        summary['abs_dev_from_unity'] = (df_wide - 1).abs().median()
        summary = summary.sort_values('abs_dev_from_unity')
    else:
        summary = summary.sort_values('median', ascending=not HIGHER_IS_BETTER[stat])

    summary_tables[stat] = summary
    print(f"\n--- {label} (ranked best to worst) ---")
    print(summary.round(3).to_string())

# =============================================================================
# Explicit ranking + variability check
# =============================================================================
print("\n" + "="*95)
print("RANKING SUMMARY (1st = best) and STD-based variability check")
print("(old text claims: 'Rain4PE highest CSI with minimal variability',")
print("'PISCO/GPM-GWR concentrate near FBI=1' - check std and |FBI-1| directly)")
print("="*95)
for stat, label in stat_labels.items():
    summary = summary_tables[stat]
    print(f"\n{label}:")
    for rank, (product, row) in enumerate(summary.iterrows(), start=1):
        print(f"  {rank}. {product:12s} median={row['median']:+.3f}  std={row['std']:.3f}  "
              f"IQR={row['IQR']:.3f}")

# FBI direction specifically (over- vs under-prediction), since the old
# text claims Rain4PE is "elevated" (overpredicts) and PISCO/GWR "near unity"
print("\n" + "="*95)
print("FBI DIRECTION (over- vs under-prediction of event frequency), per product")
print("="*95)
fbi_summary = summary_tables['fbi']
for product, row in fbi_summary.iterrows():
    direction = 'overpredicts frequency' if row['median'] > 1 else 'underpredicts frequency' if row['median'] < 1 else 'unbiased'
    print(f"  {product:12s} median FBI={row['median']:.3f}  -> {direction}")


# --------------
# Figure 6 -> Boxen plots. No significant change. Just clearer fonts, color scheme and consistent higher dpi.
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
fig.savefig(plotDir / 'fig6_categorical_metrics.png', bbox_inches='tight')
plt.close(fig)