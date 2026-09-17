# -*- coding: utf-8 -*-
"""
Full threshold sensitivity sweep: instead of just 1.0 vs 2.5 mm/day, this
sweeps a whole range of thresholds (default 1.0 to 10.0 mm/day, step 0.5)
and tracks how every detection stat (FBI/FAR/POD/ACC) and extreme index
(CDD/CWD) responds, for all 5 products at once, in one nested figure - one
subplot per metric, all products as lines within each subplot.

PERFORMANCE NOTE: the per-station daily series (obs + all 5 products) are
parsed and aligned ONCE up front, then reused across every threshold in the
sweep - this avoids re-parsing datetime indices ~19x, which would otherwise
make this considerably slower. Detection stats use the symmetric-threshold
definition (obs and sim both compared against the same cutoff), matching
the fix already applied to station_class.py.

Two extra things this makes possible, since you now have the FULL curve
rather than 2 points:
  1. WINNER-PER-THRESHOLD TRACKING: does "which product wins" stay stable
     across the whole 1-10mm range, or flip at some point? (printed table)
  2. INFLECTION/PLATEAU CHECK: metrics that keep changing steeply all the
     way to 10mm suggest the network's precipitation distribution has a lot
     of mass in that range (a genuine climatological signal); metrics that
     flatten out early suggest most of the sensitivity lives below ~2-3mm,
     which would support choosing a threshold in that lower range with more
     confidence.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
clipped_fixed_pickle = f'{wkDir}/datasets/data_locked_loaded_clipped_fixed.pkl'
min_range = '2005-01-01'
max_range = '2018-12-31'
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

products = ['rawGPM', 'gwrGPM', 'expGPM', 'rain4pe', 'PISCO']
PRODUCT_COLORS = {
    'rawGPM': '#1f77b4', 'gwrGPM': '#2ca02c', 'expGPM': '#d62728',
    'rain4pe': '#9467bd', 'PISCO': '#ff7f0e',
}

# --- EDIT the threshold range/step here ---
THRESHOLDS = np.round(np.arange(0.1, 10.5, 0.5), 2)  # 1.0, 1.5, ..., 10.0 (19 values)

# =============================================================================
# 0. Load and precompute all per-station series ONCE
# =============================================================================
with open(clipped_fixed_pickle, 'rb') as f:
    final_data = pickle.load(f)

def prep_series(df, col, start, end):
    df = df.copy()
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    return df.loc[start:end, col]

print("Pre-parsing all station series (done once, reused across all thresholds)...")
station_series = {}
for station_name, station in final_data.items():
    obs = prep_series(station.data, 'Precipitation', min_range, max_range)
    sims = {prod: prep_series(getattr(station, prod), 'precipitationCal', min_range, max_range)
            for prod in products}
    station_series[station_name] = {'obs': obs, 'sims': sims}
print(f"Done. {len(station_series)} stations loaded.")

# =============================================================================
# 1. Metric functions (vectorized per threshold, reused precomputed series)
# =============================================================================
def detection_stats(obs, sim, threshold):
    mask = obs.notna() & sim.notna()
    o, s = obs[mask], sim[mask]
    hits = ((o >= threshold) & (s >= threshold)).sum()
    false_alarms = ((o < threshold) & (s >= threshold)).sum()
    misses = ((o >= threshold) & (s < threshold)).sum()
    correct_negatives = ((o < threshold) & (s < threshold)).sum()
    denom_fbi = hits + misses
    denom_far = hits + false_alarms
    denom_acc = hits + false_alarms + misses + correct_negatives
    fbi = (hits + false_alarms) / denom_fbi if denom_fbi > 0 else np.nan
    far = false_alarms / denom_far if denom_far > 0 else np.nan
    pod = hits / denom_fbi if denom_fbi > 0 else np.nan
    acc = (hits + correct_negatives) / denom_acc if denom_acc > 0 else np.nan
    return fbi, far, pod, acc

def extreme_indices(series, threshold):
    s = series.dropna()
    if len(s) == 0:
        return np.nan, np.nan
    cdd_list, cwd_list = [], []
    for year, yearly in s.groupby(s.index.year):
        wet = (yearly > threshold).astype(int)
        dry = (yearly < threshold).astype(int)
        cwd = wet.groupby((yearly <= threshold).cumsum()).sum().max()
        cdd = dry.groupby((yearly >= threshold).cumsum()).sum().max()
        cwd_list.append(cwd)
        cdd_list.append(cdd)
    return float(np.nanmean(cdd_list)), float(np.nanmean(cwd_list))

# =============================================================================
# 2. Sweep all thresholds
# =============================================================================
rows = []
for t in THRESHOLDS:
    for station_name, data in station_series.items():
        obs = data['obs']

        obs_cdd, obs_cwd = extreme_indices(obs, t)
        rows.append({'threshold': t, 'station': station_name, 'product': 'Observed',
                      'fbi': np.nan, 'far': np.nan, 'pod': np.nan, 'acc': np.nan,
                      'cdd': obs_cdd, 'cwd': obs_cwd})

        for prod in products:
            sim = data['sims'][prod]
            fbi, far, pod, acc = detection_stats(obs, sim, t)
            cdd, cwd = extreme_indices(sim, t)
            rows.append({'threshold': t, 'station': station_name, 'product': prod,
                          'fbi': fbi, 'far': far, 'pod': pod, 'acc': acc,
                          'cdd': cdd, 'cwd': cwd})
    print(f"  threshold {t} mm/day done")

results_df = pd.DataFrame(rows)
full_out_path = os.path.join(wkDir, 'full_threshold_sweep.csv')
results_df.to_csv(full_out_path, index=False)
print(f"\nFull per-station sweep saved to {full_out_path} ({len(results_df)} rows)")

# Network-mean aggregation (this is what gets plotted)
agg = results_df.groupby(['threshold', 'product'])[['fbi', 'far', 'pod', 'acc', 'cdd', 'cwd']].mean().reset_index()
agg_out_path = os.path.join(wkDir, 'full_threshold_sweep_network_mean.csv')
agg.to_csv(agg_out_path, index=False)
print(f"Network-mean summary saved to {agg_out_path}")

# =============================================================================
# 3. Big nested figure: one subplot per metric, all products as lines
# =============================================================================
metrics_to_plot = ['fbi', 'far', 'pod', 'acc', 'cdd', 'cwd']
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
axes = axes.flatten()

for ax, metric in zip(axes, metrics_to_plot):
    for prod in products:
        sub = agg[agg['product'] == prod].sort_values('threshold')
        ax.plot(sub['threshold'], sub[metric], marker='o', markersize=3,
                label=prod, color=PRODUCT_COLORS[prod])
    if metric in ('cdd', 'cwd'):
        obs_sub = agg[agg['product'] == 'Observed'].sort_values('threshold')
        ax.plot(obs_sub['threshold'], obs_sub[metric], 'k--', linewidth=2, label='Observed')
    ax.set_title(metric.upper(), fontsize=13)
    ax.set_xlabel('Threshold (mm/day)')
    ax.set_ylabel(metric.upper())
    ax.grid(alpha=0.3)

axes[0].legend(loc='best', fontsize=8)
fig.suptitle('Threshold sensitivity sweep (1-10 mm/day): all products, all metrics', fontsize=15, y=1.01)
plt.tight_layout()
out_path = os.path.join(out_dir, 'threshold_sweep_full.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved {out_path}")

# =============================================================================
# 4. Bonus: does the "winner" per metric stay stable across the whole sweep?
# =============================================================================
print("\n" + "="*90)
print("WINNER STABILITY ACROSS THE THRESHOLD SWEEP")
print("="*90)

winner_configs = {
    'pod': ('max', None),
    'far': ('min', None),
    'acc': ('max', None),
    'cdd': ('closest_to_reference', 'cdd'),
    'cwd': ('closest_to_reference', 'cwd'),
}

for metric, (mode, ref_metric) in winner_configs.items():
    winners_by_threshold = {}
    for t in THRESHOLDS:
        sub = agg[(agg['threshold'] == t) & (agg['product'] != 'Observed')].set_index('product')
        if mode == 'max':
            winner = sub[metric].idxmax()
        elif mode == 'min':
            winner = sub[metric].idxmin()
        elif mode == 'closest_to_reference':
            ref_val = agg[(agg['threshold'] == t) & (agg['product'] == 'Observed')][ref_metric].values[0]
            winner = (sub[metric] - ref_val).abs().idxmin()
        winners_by_threshold[t] = winner

    unique_winners = set(winners_by_threshold.values())
    stability = "STABLE" if len(unique_winners) == 1 else f"CHANGES ({len(unique_winners)} different winners)"
    print(f"\n{metric.upper()} ({stability}):")
    print(pd.Series(winners_by_threshold).to_string())

print("\nDone. If a metric shows 'STABLE' across the whole 1-10mm range, your "
      "conclusion about which product is best on that metric does not depend "
      "on the specific threshold chosen - a strong robustness statement for "
      "the response letter. If it 'CHANGES', report at which threshold the "
      "flip happens and consider whether that threshold is physically "
      "meaningful (e.g. does it correspond to the noise floor discussed in "
      "your methods) or arbitrary.")