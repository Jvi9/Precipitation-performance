# -*- coding: utf-8 -*-
"""
ETCCDI extreme-value index accuracy check (CDD, CWD, R10, R20, R95p, R99p),
per product, against observed - the Figure 10c-equivalent spatial-pattern
correlation, PLUS mean bias (magnitude and direction) per product per index.

ELEVATION DEPENDENCE WAS REMOVED from this script deliberately, not
overlooked. The Discussion's "Uncertainty in validation" section already
establishes that elevation correlations across this network rest on an
uneven station distribution (40/70 stations between 2500-3750m, none
between 1250-1750m, fewer than 10 above 3750m). Computing a further
elevation correlation on an already-thin per-index, per-product subset
(n=70 split six ways) pushes that same thin sample even further - a
"significant" rho under those conditions (e.g. Rain4PE's R10 rho=-0.60 from
the earlier version of this script) is not solid enough to support a
mechanism-level claim in the text. Only genuinely cross-checkable claims -
overall accuracy, and bias magnitude/direction - remain here.

Two things are computed, per product per index:
  1. Overall accuracy: Pearson correlation, ACROSS STATIONS, between the
     product's index value and the observed index value - does the product
     correctly identify WHICH stations have more/less of this extreme,
     relative to other stations. Matches the original Figure 10c
     definition - NOT the same as the daily-scale r in Figures 3/4/8, which
     measures day-to-day timing agreement, not spatial pattern agreement.
  2. Mean bias (product - observed): magnitude and direction only, no
     elevation attached - supports claims like "Rain4PE overestimates
     R10/R20" without implying anything about where that overestimation is
     worse or better.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import pearsonr

from final_graphs.configurations import PLOT_ORDER, DISPLAY_NAMES

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
geoparquet_path = os.path.join(wkDir, 'outputs', 'station_metrics_1mm_clipped.parquet')

INDICES = ['cdd', 'cwd', 'r10', 'r20', 'r95p', 'r99p']
INDEX_LABELS = {'cdd': 'CDD', 'cwd': 'CWD', 'r10': 'R10', 'r20': 'R20', 'r95p': 'R95p', 'r99p': 'R99p'}

gdf = gpd.read_parquet(geoparquet_path)

print("="*95)
print("WHAT THESE NUMBERS MEAN")
print("="*95)
print("Overall accuracy (r): Pearson correlation, across all 70 stations, between")
print("a product's index value and the OBSERVED index value. High r = the product")
print("correctly identifies WHICH stations have more/less of this extreme,")
print("relative to other stations - a spatial-pattern-agreement measure, matching")
print("the original Figure 10c definition - not the same as the daily-scale r")
print("used in Figures 3/4/8, which measures DAY-TO-DAY timing agreement.")
print()
print("Mean bias (product - observed): average over-/under-estimation, in the")
print("index's own units (days for CDD/CWD, mm for R10/R20/R95p/R99p). No")
print("elevation attached - see module docstring for why that was dropped.")
print("="*95)

# =============================================================================
# 1. Overall accuracy per product per index (Figure 10c equivalent), WITH
#    explicit margin over the runner-up, so the text can state precisely
#    where PISCO's lead is wide versus where other datasets converge
# =============================================================================
print("\n" + "="*95)
print("OVERALL ACCURACY: correlation (r) between product and observed index value,")
print("across all stations - ranked best to worst per index, with margin over runner-up")
print("="*95)

accuracy_rows = []
for index in INDICES:
    obs_col = f'{index}_Observed'
    for code in PLOT_ORDER:
        prod_col = f'{index}_{code}'
        mask = gdf[obs_col].notna() & gdf[prod_col].notna()
        r, pval = pearsonr(gdf.loc[mask, obs_col], gdf.loc[mask, prod_col])
        bias = (gdf.loc[mask, prod_col] - gdf.loc[mask, obs_col])
        accuracy_rows.append({'index': INDEX_LABELS[index], 'product': DISPLAY_NAMES[code],
                              'r': r, 'pval': pval, 'n': mask.sum(),
                              'mean_bias': bias.mean(), 'std_bias': bias.std()})

accuracy_df = pd.DataFrame(accuracy_rows)
for index in INDEX_LABELS.values():
    sub = accuracy_df[accuracy_df['index'] == index].sort_values('r', ascending=False).reset_index(drop=True)
    print(f"\n--- {index} ---")
    for i, row in sub.iterrows():
        sig = '*' if row['pval'] < 0.05 else ''
        margin_text = ''
        if i == 0 and len(sub) > 1:
            margin_text = f"  (margin over 2nd place: {row['r'] - sub.loc[1, 'r']:+.3f})"
        print(f"  {row['product']:12s} r={row['r']:+.3f}{sig}  (n={row['n']}){margin_text}")

# Spread across products, per index - tells you at a glance whether this
# index is one where datasets diverge sharply or converge closely
print("\n--- Spread of accuracy (r) across the 5 products, per index ---")
print("(wide spread = datasets diverge sharply; narrow spread = they converge)")
for index in INDEX_LABELS.values():
    sub = accuracy_df[accuracy_df['index'] == index]
    print(f"  {index:6s}: max={sub['r'].max():.3f}  min={sub['r'].min():.3f}  "
          f"spread={sub['r'].max() - sub['r'].min():.3f}")

# =============================================================================
# 2. Mean bias (magnitude + direction), per product per index - NO elevation
# =============================================================================
print("\n" + "="*95)
print("MEAN BIAS (product - observed), per product per index")
print("(direction and magnitude only - supports claims like 'Rain4PE overestimates")
print("R10/R20' without any elevation attached)")
print("="*95)
for index in INDEX_LABELS.values():
    sub = accuracy_df[accuracy_df['index'] == index]
    print(f"\n--- {index} ---")
    for _, row in sub.sort_values('mean_bias', key=abs).iterrows():
        direction = 'overestimates' if row['mean_bias'] > 0 else 'underestimates' if row['mean_bias'] < 0 else 'unbiased'
        print(f"  {row['product']:12s} mean_bias={row['mean_bias']:+.2f}  std={row['std_bias']:.2f}  -> {direction}")