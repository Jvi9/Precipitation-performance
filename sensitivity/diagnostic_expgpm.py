# -*- coding: utf-8 -*-
"""
Diagnostic for the three extreme expGPM CDD-bias outlier stations found in the
threshold sensitivity analysis:
  - Crisnejas_ Augusto Weberbauer   (bias +198.6 / +205.2 days)
  - Intercuenca Alto Huallaga_ Campanilla (bias +131.7 / +152.9 days)
  - Pachitea_ Tournavista           (bias +127.3 / +152.0 days)

These are far beyond anything else in the 70-station network (next-largest
outlier elsewhere in the table is under +35 days) and appear at BOTH
thresholds, meaning this isn't a threshold artifact - something specific to
expGPM at these three (low-elevation, 149-586m) stations is producing runs of
near-zero values far longer than observed or any other product shows.

This script pulls the raw daily series directly from the pickle (not through
Generator) to inspect: NaN/zero/negative rates, the longest zero-run in
expGPM vs observed, and a zoomed plot around that run so you can see by eye
whether it's a real dry spell, a data gap, or a stuck/fill-value artifact.
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0. Setup - EDIT THESE TO MATCH YOUR ENVIRONMENT
# ---------------------------------------------------------------------------
wkDir = r'C:\Users\jvila\Desktop\Andean_project'
load_data_path = f'{wkDir}/datasets/data_locked_loaded.pkl'
min_range = '2005-01-01'
max_range = '2018-12-31'
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

with open(load_data_path, 'rb') as f:
    final_data = pickle.load(f)

outlier_stations = [
    'Crisnejas_ Augusto Weberbauer',
    'Intercuenca Alto Huallaga_ Campanilla',
    'Pachitea_ Tournavista',
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def prep_series(df, col, start, end):
    df = df.copy()
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    return df.loc[start:end, col]

def longest_run(bool_series):
    """Longest run of consecutive True values, and the date the run ends."""
    grp = (bool_series != bool_series.shift()).cumsum()
    run_lengths = bool_series.astype(int).groupby(grp).cumsum()
    end_date = run_lengths.idxmax()
    return int(run_lengths.max()), end_date

# ---------------------------------------------------------------------------
# Main diagnostic loop
# ---------------------------------------------------------------------------
summary_rows = []

for station_name in outlier_stations:
    print(f"\n{'='*90}\n{station_name}\n{'='*90}")
    station = final_data[station_name]

    obs = prep_series(station.data, 'Precipitation', min_range, max_range)
    expgpm = prep_series(station.expGPM, 'precipitationCal', min_range, max_range)
    rawgpm = prep_series(station.rawGPM, 'precipitationCal', min_range, max_range)
    gwrgpm = prep_series(station.gwrGPM, 'precipitationCal', min_range, max_range)

    # --- Basic value diagnostics for expGPM ---
    n = len(expgpm)
    n_nan = int(expgpm.isna().sum())
    n_zero = int((expgpm == 0).sum())
    n_neg = int((expgpm < 0).sum())
    print(f"expGPM series length: {n}")
    print(f"  NaNs: {n_nan} ({100*n_nan/n:.1f}%)  |  zeros: {n_zero} ({100*n_zero/n:.1f}%)  |  negatives: {n_neg}")
    print(f"  min/max: {expgpm.min()} / {expgpm.max()}")
    print(f"Observed min/max: {obs.min()} / {obs.max()}  |  NaNs: {int(obs.isna().sum())}")

    # --- Longest run of near-zero (<1mm) values in expGPM vs observed ---
    expgpm_low = expgpm.fillna(0) < 1.0
    obs_low = obs.fillna(0) < 1.0
    exp_run_len, exp_run_end = longest_run(expgpm_low)
    obs_run_len, obs_run_end = longest_run(obs_low)
    print(f"Longest run of days <1mm in expGPM: {exp_run_len} days, ending {exp_run_end.date()}")
    print(f"Longest run of days <1mm in OBSERVED: {obs_run_len} days, ending {obs_run_end.date()}")

    # --- What do the OTHER products show during expGPM's longest dry run? ---
    run_start = exp_run_end - pd.Timedelta(days=exp_run_len - 1)
    print(f"During expGPM's dry run ({run_start.date()} to {exp_run_end.date()}):")
    print(f"  Observed total: {obs.loc[run_start:exp_run_end].sum():.1f} mm, "
          f"days with obs > 0: {int((obs.loc[run_start:exp_run_end] > 0).sum())}")
    print(f"  rawGPM total:   {rawgpm.loc[run_start:exp_run_end].sum():.1f} mm")
    print(f"  gwrGPM total:   {gwrgpm.loc[run_start:exp_run_end].sum():.1f} mm")

    summary_rows.append({
        'station': station_name,
        'expGPM_nan_pct': round(100 * n_nan / n, 2),
        'expGPM_zero_pct': round(100 * n_zero / n, 2),
        'expGPM_longest_dry_run_days': exp_run_len,
        'obs_longest_dry_run_days': obs_run_len,
        'obs_total_mm_during_expGPM_dry_run': round(float(obs.loc[run_start:exp_run_end].sum()), 1),
        'rawGPM_total_mm_during_expGPM_dry_run': round(float(rawgpm.loc[run_start:exp_run_end].sum()), 1),
        'gwrGPM_total_mm_during_expGPM_dry_run': round(float(gwrgpm.loc[run_start:exp_run_end].sum()), 1),
        'expGPM_max': float(expgpm.max()),
        'obs_max': float(obs.max()),
    })

    # --- Plot: full series + zoom around the longest expGPM dry run ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))

    axes[0].plot(obs.index, obs.values, label='Observed', color='black', linewidth=0.7)
    axes[0].plot(expgpm.index, expgpm.values, label='expGPM', color='red', alpha=0.7)
    axes[0].plot(rawgpm.index, rawgpm.values, label='rawGPM', color='blue', alpha=0.4)
    axes[0].plot(gwrgpm.index, gwrgpm.values, label='gwrGPM', color='green', alpha=0.4)
    axes[0].axvspan(run_start, exp_run_end, color='orange', alpha=0.15, label='expGPM dry run')
    axes[0].set_title(f'{station_name} - full daily series')
    axes[0].set_ylabel('Precipitation (mm)')
    axes[0].legend(loc='upper right')

    zoom_start = run_start - pd.Timedelta(days=30)
    zoom_end = exp_run_end + pd.Timedelta(days=30)
    for series, label, color in [(obs, 'Observed', 'black'), (expgpm, 'expGPM', 'red'),
                                   (rawgpm, 'rawGPM', 'blue'), (gwrgpm, 'gwrGPM', 'green')]:
        sub = series.loc[zoom_start:zoom_end]
        axes[1].plot(sub.index, sub.values, label=label, color=color, marker='o', markersize=2)
    axes[1].axvspan(run_start, exp_run_end, color='orange', alpha=0.15)
    axes[1].set_title(f'Zoom on expGPM dry run: {run_start.date()} to {exp_run_end.date()} ({exp_run_len} days)')
    axes[1].set_ylabel('Precipitation (mm)')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    safe_name = station_name.replace(' ', '_').replace('/', '-')
    out_path = os.path.join(out_dir, f'diagnostic_{safe_name}.png')
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved diagnostic plot to {out_path}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
summary_df = pd.DataFrame(summary_rows).set_index('station')
print("\n" + "="*90)
print("SUMMARY - expGPM outlier diagnostics")
print("="*90)
print(summary_df.to_string())

summary_path = os.path.join(wkDir, 'expgpm_outlier_diagnostics.csv')
summary_df.to_csv(summary_path)
print(f"\nSummary saved to {summary_path}")