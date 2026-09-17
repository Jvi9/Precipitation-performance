# -*- coding: utf-8 -*-
"""
Direct test of "does GPM capture peaks better because it isn't softened by
gauge correction?" - conditions on days OBSERVED precipitation was extreme
(exceeding the gauge's own 95th/99th percentile), and checks what each
product reported on THOSE SPECIFIC DAYS. This is different from R95p/R99p
in the GeoParquet, which uses each product's own percentile threshold and
can't tell you whether a product actually captured the real events versus
just being noisy overall.

For each station:
  - threshold = observed 95th (and 99th) percentile of daily precip
  - select days where OBSERVED exceeds that threshold (real extreme events)
  - for each product, on those exact days:
      - mean value (is it capturing the magnitude, or smoothing it down?)
      - ratio to observed mean on those days (>1 = overshoots peaks,
        <1 = softens/underestimates peaks - this is the key test)
      - POD restricted to these extreme days only (did it detect rain
        AT ALL on the day of a real extreme event?)
      - peak ratio: product's own max / observed max, network-wide

If rain4pe/PISCO show ratio < 1 (softened) and GPM-based products show
ratio > 1 or closer to 1, that's direct, specific evidence for the
softening hypothesis - not just an artifact of each product's own
percentile definition.
"""
import os
import pickle
import numpy as np
import pandas as pd

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
clipped_fixed_pickle = f'{wkDir}/datasets/data_locked_loaded_clipped_fixed.pkl'
min_range = '2005-01-01'
max_range = '2018-12-31'

products = ['rawGPM', 'gwrGPM', 'expGPM', 'rain4pe', 'PISCO']

with open(clipped_fixed_pickle, 'rb') as f:
    final_data = pickle.load(f)

def prep_series(df, col, start, end):
    df = df.copy()
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    return df.loc[start:end, col]

rows = []
peak_rows = []

for station_name, station in final_data.items():
    obs = prep_series(station.data, 'Precipitation', min_range, max_range).dropna()
    if len(obs) < 30:
        continue

    obs_max = obs.max()

    for pctile in [0.95, 0.99]:
        threshold = obs.quantile(pctile)
        extreme_days = obs[obs > threshold].index

        for prod in products:
            sim = prep_series(getattr(station, prod), 'precipitationCal', min_range, max_range)
            sim_on_extreme_days = sim.reindex(extreme_days)
            obs_on_extreme_days = obs.reindex(extreme_days)

            mask = sim_on_extreme_days.notna() & obs_on_extreme_days.notna()
            if mask.sum() < 5:
                continue

            sim_vals = sim_on_extreme_days[mask]
            obs_vals = obs_on_extreme_days[mask]

            mean_ratio = sim_vals.mean() / obs_vals.mean() if obs_vals.mean() != 0 else np.nan
            mean_bias = (sim_vals - obs_vals).mean()
            pod_on_extreme_days = (sim_vals >= 1.0).mean()  # did it register ANY rain that day

            rows.append({
                'station': station_name,
                'percentile': pctile,
                'product': prod,
                'n_extreme_days': mask.sum(),
                'obs_mean_on_extreme_days': obs_vals.mean(),
                'sim_mean_on_extreme_days': sim_vals.mean(),
                'ratio_sim_to_obs': mean_ratio,
                'mean_bias': mean_bias,
                'pod_on_extreme_days': pod_on_extreme_days,
            })

    # Network-wide peak ratio (station max, not just percentile-conditioned)
    for prod in products:
        sim = prep_series(getattr(station, prod), 'precipitationCal', min_range, max_range).dropna()
        if len(sim) == 0:
            continue
        peak_rows.append({
            'station': station_name,
            'product': prod,
            'obs_max': obs_max,
            'sim_max': sim.max(),
            'peak_ratio': sim.max() / obs_max if obs_max != 0 else np.nan,
        })

results_df = pd.DataFrame(rows)
peak_df = pd.DataFrame(peak_rows)

print("="*90)
print("CONDITIONAL PEAK-CAPTURE TEST: ratio of simulated to observed value,")
print("computed ONLY on days observed precipitation was extreme")
print("(ratio < 1 = softened/underestimated peaks; ratio > 1 = overshoots)")
print("="*90)

for pctile in [0.95, 0.99]:
    sub = results_df[results_df['percentile'] == pctile]
    summary = sub.groupby('product').agg(
        mean_ratio=('ratio_sim_to_obs', 'mean'),
        median_ratio=('ratio_sim_to_obs', 'median'),
        mean_bias=('mean_bias', 'mean'),
        mean_pod_on_extreme_days=('pod_on_extreme_days', 'mean'),
    ).reindex(products)
    print(f"\n--- Days observed > {int(pctile*100)}th percentile ---")
    print(summary.round(3).to_string())

print("\n" + "="*90)
print("PEAK RATIO: each product's own maximum value / observed maximum, per station")
print("="*90)
peak_summary = peak_df.groupby('product')['peak_ratio'].agg(['mean', 'median', 'std']).reindex(products)
print(peak_summary.round(3).to_string())

print("\nInterpretation guide:")
print("- ratio_sim_to_obs < 1 on extreme days = that product SOFTENS real peaks")
print("- ratio_sim_to_obs > 1 = that product OVERSHOOTS real peaks")
print("- peak_ratio < 1 = that product never reaches the same maximum intensity as the gauge")
print("- pod_on_extreme_days = fraction of real extreme-rain days where the product")
print("  registered at least 1mm - low value means it MISSED the event entirely,")
print("  not just underestimated its magnitude")

out_path = os.path.join(wkDir, 'peak_capture_test.csv')
results_df.to_csv(out_path, index=False)
peak_df.to_csv(os.path.join(wkDir, 'peak_ratio_test.csv'), index=False)
print(f"\nSaved detailed results to {out_path} and peak_ratio_test.csv")