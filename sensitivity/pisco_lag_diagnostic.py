# -*- coding: utf-8 -*-
"""
Diagnostic for PISCO's oddly low correlation despite a near-1 std ratio
(Taylor diagram finding). Checks whether this is a genuine skill issue or a
date-alignment artifact (timezone/UTC-vs-local day boundary, off-by-one
resampling) by:
  1. Plotting PISCO vs observed over a short window at a low-r station, so
     you can eyeball whether wet days line up or are shifted.
  2. Computing correlation at lags -3 to +3 days - if PISCO's real
     correlation peaks at a non-zero lag, that's a strong signal of a
     systematic day-shift rather than a genuine skill problem.
Repeats for a couple of stations and for rawGPM as a control (rawGPM's
correlation should NOT improve much at any lag, since it wasn't flagged as
having this issue).
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
clipped_data_path = f'{wkDir}/datasets/data_locked_loaded_clipped.pkl'
min_range = '2005-01-01'
max_range = '2018-12-31'
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

# Stations with notably low PISCO r from the Taylor diagram data
check_stations = ['Perene_ Satipo', 'Pachitea_ Tournavista', 'Perene_ Puerto Ocopa']
control_products = ['PISCO', 'rawGPM']  # rawGPM as a control - should NOT show a lag effect

with open(clipped_data_path, 'rb') as f:
    final_data = pickle.load(f)

def prep_series(df, col, start, end):
    df = df.copy()
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    return df.loc[start:end, col]

def lagged_corr(obs, sim, max_lag=3):
    """Correlation of obs[t] with sim[t+lag] for lag in -max_lag..+max_lag.
    Positive lag means sim is shifted FORWARD relative to obs (sim lags obs)."""
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        shifted = sim.shift(lag)
        mask = obs.notna() & shifted.notna()
        if mask.sum() < 30:
            results[lag] = np.nan
            continue
        results[lag] = obs[mask].corr(shifted[mask])
    return pd.Series(results)

# =============================================================================
# Run diagnostic per station
# =============================================================================
summary_rows = []

for station_name in check_stations:
    if station_name not in final_data:
        print(f"Station '{station_name}' not found - check exact name/spacing.")
        continue
    station = final_data[station_name]
    obs = prep_series(station.data, 'Precipitation', min_range, max_range)

    print(f"\n{'='*90}\n{station_name}\n{'='*90}")

    fig, axes = plt.subplots(len(control_products), 2, figsize=(16, 4 * len(control_products)))
    if len(control_products) == 1:
        axes = axes.reshape(1, -1)

    for i, prod in enumerate(control_products):
        sim = prep_series(getattr(station, prod), 'precipitationCal', min_range, max_range)

        # --- Lagged correlation ---
        lag_corrs = lagged_corr(obs, sim, max_lag=3)
        best_lag = lag_corrs.idxmax()
        best_r = lag_corrs.max()
        zero_lag_r = lag_corrs[0]
        print(f"\n{prod}: correlation at lag 0 = {zero_lag_r:.3f}")
        print(f"{prod}: correlation by lag ->")
        print(lag_corrs.round(3).to_string())
        print(f"{prod}: BEST lag = {best_lag} days (r = {best_r:.3f})")
        if best_lag != 0 and (best_r - zero_lag_r) > 0.05:
            print(f"  *** Possible day-shift: correlation improves by "
                  f"{best_r - zero_lag_r:.3f} at lag {best_lag} ***")
        else:
            print(f"  No meaningful improvement away from lag 0 - not a shift artifact here.")

        summary_rows.append({
            'station': station_name, 'product': prod, 'r_lag0': zero_lag_r,
            'best_lag': best_lag, 'r_best_lag': best_r,
            'improvement': best_r - zero_lag_r,
        })

        # --- Lag-correlation bar plot ---
        axes[i, 0].bar(lag_corrs.index, lag_corrs.values, color='steelblue')
        axes[i, 0].axvline(0, color='k', linestyle='--', linewidth=0.8)
        axes[i, 0].set_title(f'{prod}: correlation by lag (days)')
        axes[i, 0].set_xlabel('Lag (days, +ve = sim shifted later)')
        axes[i, 0].set_ylabel('Correlation')

        # --- Short window overlay (first available full year, 60-day window) ---
        window_start = obs.first_valid_index() + pd.Timedelta(days=200)
        window_end = window_start + pd.Timedelta(days=60)
        obs_w = obs.loc[window_start:window_end]
        sim_w = sim.loc[window_start:window_end]
        sim_w_shifted = sim.shift(best_lag).loc[window_start:window_end]

        axes[i, 1].plot(obs_w.index, obs_w.values, label='Observed', color='black', marker='o', markersize=3)
        axes[i, 1].plot(sim_w.index, sim_w.values, label=f'{prod} (lag 0)', color='red', alpha=0.7, marker='o', markersize=3)
        if best_lag != 0:
            axes[i, 1].plot(sim_w_shifted.index, sim_w_shifted.values, label=f'{prod} (shifted {best_lag}d)',
                             color='green', alpha=0.7, linestyle='--', marker='o', markersize=3)
        axes[i, 1].set_title(f'{prod} vs Observed - {window_start.date()} to {window_end.date()}')
        axes[i, 1].set_ylabel('Precipitation (mm)')
        axes[i, 1].legend(fontsize=8)

    plt.tight_layout()
    safe_name = station_name.replace(' ', '_').replace('/', '-')
    out_path = os.path.join(out_dir, f'lag_diagnostic_{safe_name}.png')
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"\nSaved {out_path}")

# =============================================================================
# Summary table
# =============================================================================
summary_df = pd.DataFrame(summary_rows)
print("\n" + "="*90)
print("SUMMARY - lag diagnostic")
print("="*90)
print(summary_df.to_string(index=False))

summary_path = os.path.join(wkDir, 'lag_diagnostic_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary saved to {summary_path}")