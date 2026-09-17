# -*- coding: utf-8 -*-
"""
1. Scans EVERY station and EVERY simulated product (rawGPM, gwrGPM, expGPM,
   rain4pe, PISCO) for negative precipitation values - not just the 3 flagged
   stations - so you know how widespread this actually is before deciding how
   big a fix this is.
2. Produces a cleaned copy of final_data with all negative values clipped to
   zero (standard practice for downscaling residual correction, e.g.
   Immerzeel et al. 2009), saved as a new pickle so your original data file
   is untouched.

NOTE: this clips the sign/magnitude issue (affects MAE/RMSE/PBIAS/R/KGE).
It will NOT by itself fix the multi-year expGPM "dry run" at the 3 outlier
stations found earlier - a day with precip < wet_threshold is already
classified as dry whether the value is -15 or 0. That's a separate,
substantive downscaling-failure question, not a sign-convention bug.
"""
import os
import pickle
import copy
import pandas as pd
import numpy as np

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
load_data_path = f'{wkDir}/datasets/data_locked_loaded.pkl'
clipped_save_path = f'{wkDir}/datasets/data_locked_loaded_clipped.pkl'

sim_products = ['rawGPM', 'gwrGPM', 'expGPM', 'rain4pe', 'PISCO']

with open(load_data_path, 'rb') as f:
    final_data = pickle.load(f)

# ---------------------------------------------------------------------------
# 1. Scope check: how many negative values per product, across ALL stations
# ---------------------------------------------------------------------------
scope_rows = []

for station_name, station in final_data.items():
    for product in sim_products:
        df = getattr(station, product, None)
        if df is None:
            continue
        series = df['precipitationCal'] if 'precipitationCal' in df.columns else None
        if series is None:
            continue
        n = len(series)
        n_neg = int((series < 0).sum())
        if n_neg > 0:
            scope_rows.append({
                'station': station_name,
                'product': product,
                'n_total': n,
                'n_negative': n_neg,
                'pct_negative': round(100 * n_neg / n, 2),
                'min_value': float(series.min()),
            })

scope_df = pd.DataFrame(scope_rows)

print("="*90)
print("SCOPE: negative-value counts by product, across the whole network")
print("="*90)
if scope_df.empty:
    print("No negative values found in any product.")
else:
    by_product = scope_df.groupby('product').agg(
        stations_affected=('station', 'nunique'),
        total_negative_values=('n_negative', 'sum'),
        max_pct_negative_at_a_station=('pct_negative', 'max'),
        most_negative_value=('min_value', 'min'),
    )
    print(by_product.to_string())

    print("\n--- Per-station detail (sorted by pct_negative, descending) ---")
    print(scope_df.sort_values('pct_negative', ascending=False).to_string(index=False))

    scope_out_path = os.path.join(wkDir, 'negative_precip_scope.csv')
    scope_df.to_csv(scope_out_path, index=False)
    print(f"\nFull scope table saved to {scope_out_path}")

# ---------------------------------------------------------------------------
# 2. Produce a cleaned copy: negatives clipped to zero
# ---------------------------------------------------------------------------
final_data_clipped = copy.deepcopy(final_data)

clip_summary = []
for station_name, station in final_data_clipped.items():
    for product in sim_products:
        df = getattr(station, product, None)
        if df is None or 'precipitationCal' not in df.columns:
            continue
        n_clipped = int((df['precipitationCal'] < 0).sum())
        if n_clipped > 0:
            df['precipitationCal'] = df['precipitationCal'].clip(lower=0)
            clip_summary.append({'station': station_name, 'product': product, 'n_clipped': n_clipped})

with open(clipped_save_path, 'wb') as f:
    pickle.dump(final_data_clipped, f)

total_clipped = sum(row['n_clipped'] for row in clip_summary)
print(f"\nClipped {total_clipped} negative values to zero across "
      f"{len(clip_summary)} station/product combinations.")
print(f"Cleaned dataset saved to {clipped_save_path}")
print("\nTo use it: point load_data_path in sensitivity_test.py / "
      "station_level_full_analysis.py at this new file instead of the "
      "original, and rerun. This changes MAE/RMSE/PBIAS/R/KGE - it will NOT "
      "change CDD/CWD/detection stats, since those only look at whether a "
      "value is above or below a threshold, and negative values already "
      "fell below any sensible threshold before clipping.")