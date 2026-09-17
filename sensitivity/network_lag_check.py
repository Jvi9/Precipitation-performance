# -*- coding: utf-8 -*-
"""
Network-wide check: does shifting PISCO by -1 day improve correlation with
observed at EVERY station, or only the 3 already checked? rawGPM is included
as a control at every station too, to confirm the shift is PISCO-specific
and not a general pipeline date-handling issue.

If PISCO improves at lag -1 almost everywhere, that's strong confirmation of
a systematic one-day offset in how PISCO's time index was built/extracted -
worth fixing at the source before anything downstream (Taylor diagram,
winner maps, GeoParquet, manuscript tables) is finalized.
"""
import os
import pickle
import numpy as np
import pandas as pd

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
clipped_data_path = f'{wkDir}/datasets/data_locked_loaded_clipped.pkl'
min_range = '2005-01-01'
max_range = '2018-12-31'

products_to_check = ['PISCO', 'rawGPM', 'gwrGPM', 'expGPM', 'rain4pe']

with open(clipped_data_path, 'rb') as f:
    final_data = pickle.load(f)

def prep_series(df, col, start, end):
    df = df.copy()
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    return df.loc[start:end, col]

def lagged_corr(obs, sim, max_lag=3):
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        shifted = sim.shift(lag)
        mask = obs.notna() & shifted.notna()
        if mask.sum() < 30:
            results[lag] = np.nan
            continue
        results[lag] = obs[mask].corr(shifted[mask])
    return pd.Series(results)

rows = []
for station_name, station in final_data.items():
    obs = prep_series(station.data, 'Precipitation', min_range, max_range)
    for prod in products_to_check:
        sim = prep_series(getattr(station, prod), 'precipitationCal', min_range, max_range)
        lag_corrs = lagged_corr(obs, sim, max_lag=3)
        if lag_corrs.isna().all():
            continue
        best_lag = lag_corrs.idxmax()
        best_r = lag_corrs.max()
        r_lag0 = lag_corrs[0]
        r_lagm1 = lag_corrs[-1]
        rows.append({
            'station': station_name,
            'product': prod,
            'r_lag0': r_lag0,
            'r_lag_minus1': r_lagm1,
            'improvement_at_minus1': r_lagm1 - r_lag0,
            'best_lag': best_lag,
            'r_best_lag': best_r,
        })

df = pd.DataFrame(rows)

# =============================================================================
# Summary: how many stations show a meaningful lag-(-1) improvement, per product
# =============================================================================
THRESH = 0.05  # "meaningful improvement" cutoff, same as used in the single-station check

print("="*90)
print("Network-wide lag check summary (threshold for 'meaningful improvement' = 0.05)")
print("="*90)

for prod in products_to_check:
    sub = df[df['product'] == prod]
    n_total = len(sub)
    n_best_minus1 = (sub['best_lag'] == -1).sum()
    n_improved_minus1 = (sub['improvement_at_minus1'] > THRESH).sum()
    mean_improvement = sub['improvement_at_minus1'].mean()
    mean_r_lag0 = sub['r_lag0'].mean()
    mean_r_lagm1 = sub['r_lag_minus1'].mean()
    print(f"\n{prod}:")
    print(f"  Stations where lag -1 is the BEST lag: {n_best_minus1} / {n_total}")
    print(f"  Stations with meaningful improvement (>{THRESH}) at lag -1: {n_improved_minus1} / {n_total}")
    print(f"  Mean r at lag 0:  {mean_r_lag0:.3f}")
    print(f"  Mean r at lag -1: {mean_r_lagm1:.3f}")
    print(f"  Mean improvement at lag -1: {mean_improvement:.3f}")

# =============================================================================
# Full per-station table for PISCO specifically (the one under suspicion)
# =============================================================================
print("\n" + "="*90)
print("Full per-station detail for PISCO (sorted by improvement, descending)")
print("="*90)
pisco_df = df[df['product'] == 'PISCO'].sort_values('improvement_at_minus1', ascending=False)
print(pisco_df.to_string(index=False))

# =============================================================================
# Save everything
# =============================================================================
out_path = os.path.join(wkDir, 'network_lag_check.csv')
df.to_csv(out_path, index=False)
print(f"\nFull results (all products, all stations) saved to {out_path}")