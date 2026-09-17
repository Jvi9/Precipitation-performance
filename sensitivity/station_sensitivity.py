# -*- coding: utf-8 -*-
"""
Full station-level breakdown of the 1.0 vs 2.5 mm/day sensitivity test:
  (a) CDD / CWD - as before, but now with untruncated terminal output
  (b) FBI / FAR / POD / ACC - NEW: per-station delta between thresholds,
      outlier flags, and per-station "which product wins" flip detection
  (c) altitude correlation for both

Assumes `generators` (dict with keys '1.0mm' and '2.5mm') already exists in
your session from sensitivity_test.py. If starting a fresh session, uncomment
the rebuild block below.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Force full, untruncated output everywhere in this session ---
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# --- Uncomment if `generators` doesn't already exist in this session ---
from analysis_class import *
wkDir = r'C:\Users\jvila\Desktop\Andean_project'
os.chdir(wkDir)
load_data_path = f'{os.getcwd()}/datasets/data_locked_loaded_clipped.pkl'
min_range = '2005-01-01'
max_range = '2018-12-31'
thresholds = {'1.0mm': 1.0, '2.5mm': 2.5}
generators = {label: Generator(load_data_path, min_range, max_range, thr)
              for label, thr in thresholds.items()}

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)
thresholds = {'1.0mm': 1.0, '2.5mm': 2.5}

gen_1mm = generators['1.0mm']
gen_25mm = generators['2.5mm']

products = {
    'Observed': 'extremeIndicesObserved',
    'rawGPM': 'extremeIndicesrawGPM',
    'gwrGPM': 'extremeIndicesgwrGPM',
    'expGPM': 'extremeIndicesexpGPM',
    'PISCO': 'extremeIndicesPISCO',
    'rain4pe': 'extremeIndicesrain4pe',
}
sim_products = [p for p in products if p != 'Observed']

geo = gen_1mm.geo_summary().set_index('station')
geo['alt'] = pd.to_numeric(geo['alt'], errors='coerce')

# =============================================================================
# PART A - CDD / CWD station-level bias (same as before, full printing)
# =============================================================================
def extract_cdd_cwd_station(generator, attr_name):
    ext_dict = getattr(generator, attr_name)
    rows = []
    for station, ei in ext_dict.items():
        rows.append({'station': station, 'cdd_mean': ei.cdd.mean(), 'cwd_mean': ei.cwd.mean()})
    return pd.DataFrame(rows).set_index('station')

station_level = {}
for label, gen in generators.items():
    cdd_df = pd.DataFrame({p: extract_cdd_cwd_station(gen, attr)['cdd_mean'] for p, attr in products.items()})
    cwd_df = pd.DataFrame({p: extract_cdd_cwd_station(gen, attr)['cwd_mean'] for p, attr in products.items()})
    station_level[label] = {'cdd': cdd_df, 'cwd': cwd_df}

def bias_table(index_name):
    rows = []
    for label in thresholds:
        df = station_level[label][index_name]
        obs = df['Observed']
        for prod in sim_products:
            bias = df[prod] - obs
            for station, b in bias.items():
                rows.append({
                    'station': station,
                    'altitude': geo.loc[station, 'alt'] if station in geo.index else np.nan,
                    'product': prod,
                    'threshold': label,
                    'bias': b,
                })
    return pd.DataFrame(rows)

cdd_bias = bias_table('cdd')
cwd_bias = bias_table('cwd')

def sign_flip_stations(bias_df):
    pivot = bias_df.pivot_table(index=['station', 'product'], columns='threshold', values='bias')
    pivot['sign_flip'] = np.sign(pivot['1.0mm']) != np.sign(pivot['2.5mm'])
    return pivot

cdd_flips = sign_flip_stations(cdd_bias)
cwd_flips = sign_flip_stations(cwd_bias)

def flag_outliers(bias_df, value_col='bias', z_thresh=2.0, group_cols=('product', 'threshold')):
    out = []
    for keys, group in bias_df.groupby(list(group_cols)):
        mu, sigma = group[value_col].mean(), group[value_col].std()
        if sigma == 0 or np.isnan(sigma):
            continue
        z = (group[value_col] - mu) / sigma
        mask = np.abs(z) > z_thresh
        outliers = group[mask].copy()
        outliers['z_score'] = z[mask]
        out.append(outliers)
    return pd.concat(out) if out else pd.DataFrame()

cdd_outliers = flag_outliers(cdd_bias)
cwd_outliers = flag_outliers(cwd_bias)

print("\n" + "="*90)
print("PART A - CDD / CWD")
print("="*90)

print("\n--- [A1] Stations/products where CDD bias sign flips between thresholds ---")
print(cdd_flips[cdd_flips['sign_flip']].sort_index().to_string())

print("\n--- [A2] Stations/products where CWD bias sign flips between thresholds ---")
print(cwd_flips[cwd_flips['sign_flip']].sort_index().to_string())

print("\n--- [A3] Outlier stations for CDD bias (|z| > 2, within product+threshold) ---")
print(cdd_outliers.sort_values(['product', 'threshold', 'z_score']).to_string() if not cdd_outliers.empty else "None found")

print("\n--- [A4] Outlier stations for CWD bias (|z| > 2, within product+threshold) ---")
print(cwd_outliers.sort_values(['product', 'threshold', 'z_score']).to_string() if not cwd_outliers.empty else "None found")

print("\n--- [A5] Correlation between altitude and CDD bias (per product, per threshold) ---")
for (prod, thr), group in cdd_bias.groupby(['product', 'threshold']):
    valid = group.dropna(subset=['altitude', 'bias'])
    if len(valid) > 2:
        r = valid['altitude'].corr(valid['bias'])
        print(f"  {prod} | {thr}: r = {r:.3f} (n={len(valid)})")

print("\n--- [A6] Correlation between altitude and CWD bias (per product, per threshold) ---")
for (prod, thr), group in cwd_bias.groupby(['product', 'threshold']):
    valid = group.dropna(subset=['altitude', 'bias'])
    if len(valid) > 2:
        r = valid['altitude'].corr(valid['bias'])
        print(f"  {prod} | {thr}: r = {r:.3f} (n={len(valid)})")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.boxplot(data=cdd_bias, x='product', y='bias', hue='threshold', ax=axes[0])
axes[0].axhline(0, color='k', linestyle='--', linewidth=0.8)
axes[0].set_title('Station-level CDD bias distribution')
sns.boxplot(data=cwd_bias, x='product', y='bias', hue='threshold', ax=axes[1])
axes[1].axhline(0, color='k', linestyle='--', linewidth=0.8)
axes[1].set_title('Station-level CWD bias distribution')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'station_level_bias_boxplot.png'), dpi=300)
plt.show()

# =============================================================================
# PART B - Detection stats (FBI / FAR / POD / ACC), station level - NEW
# =============================================================================
print("\n" + "="*90)
print("PART B - Detection stats (FBI / FAR / POD / ACC)")
print("="*90)

detection_metrics = ['fbi', 'far', 'pod', 'acc']
detection_by_threshold = {
    label: gen.join_stats(metrics=detection_metrics)
    for label, gen in generators.items()
}

def extract_metric_wide(df, metric):
    """From a join_stats() output (columns like 'fbi_rawGPM'), pull one metric
    as a station x product wide table."""
    cols = [c for c in df.columns if c.startswith(metric + '_')]
    sub = df[cols].copy()
    sub.columns = [c.replace(metric + '_', '') for c in cols]
    return sub

# --- B1. Long-format table: station, altitude, metric, product, threshold, value, delta ---
detection_rows = []
for metric in detection_metrics:
    wide_1 = extract_metric_wide(detection_by_threshold['1.0mm'], metric)
    wide_25 = extract_metric_wide(detection_by_threshold['2.5mm'], metric)
    for prod in wide_1.columns:
        for station in wide_1.index:
            v1 = wide_1.loc[station, prod]
            v25 = wide_25.loc[station, prod] if station in wide_25.index else np.nan
            detection_rows.append({
                'station': station,
                'altitude': geo.loc[station, 'alt'] if station in geo.index else np.nan,
                'metric': metric,
                'product': prod,
                'value_1.0mm': v1,
                'value_2.5mm': v25,
                'delta': v25 - v1,
            })
detection_long = pd.DataFrame(detection_rows)

print("\n--- [B1] Full station-level detection stats table (both thresholds + delta) ---")
print(detection_long.sort_values(['metric', 'product', 'station']).to_string())

# --- B2. Outlier stations by DELTA (largest shift between thresholds) ---
detection_outliers = flag_outliers(detection_long, value_col='delta', group_cols=('metric', 'product'))
print("\n--- [B2] Outlier stations by delta (|z| > 2, within metric+product) ---")
print(detection_outliers.sort_values(['metric', 'product', 'z_score']).to_string() if not detection_outliers.empty else "None found")

# --- B3. Per-station "which product wins" flip detection ---
# pod/acc: higher is better. far: lower is better. fbi: closest to 1 is better.
print("\n--- [B3] Stations where the BEST product changes between thresholds ---")
winner_flip_rows = []
for metric in detection_metrics:
    wide_1 = extract_metric_wide(detection_by_threshold['1.0mm'], metric)
    wide_25 = extract_metric_wide(detection_by_threshold['2.5mm'], metric)

    if metric == 'far':
        winner_1 = wide_1.idxmin(axis=1)
        winner_25 = wide_25.idxmin(axis=1)
    elif metric == 'fbi':
        winner_1 = (wide_1 - 1).abs().idxmin(axis=1)
        winner_25 = (wide_25 - 1).abs().idxmin(axis=1)
    else:  # pod, acc
        winner_1 = wide_1.idxmax(axis=1)
        winner_25 = wide_25.idxmax(axis=1)

    for station in winner_1.index:
        w1 = winner_1.get(station, np.nan)
        w25 = winner_25.get(station, np.nan)
        flip = w1 != w25
        winner_flip_rows.append({
            'station': station,
            'altitude': geo.loc[station, 'alt'] if station in geo.index else np.nan,
            'metric': metric,
            'winner_1.0mm': w1,
            'winner_2.5mm': w25,
            'flip': flip,
        })
winner_flips = pd.DataFrame(winner_flip_rows)
print(winner_flips[winner_flips['flip']].sort_values(['metric', 'station']).to_string())

print(f"\nSummary - stations with a winner flip, by metric:")
print(winner_flips.groupby('metric')['flip'].sum().to_string())

# --- B4. Altitude correlation with delta (per metric, per product) ---
print("\n--- [B4] Correlation between altitude and delta (2.5mm - 1.0mm), per metric/product ---")
for (metric, prod), group in detection_long.groupby(['metric', 'product']):
    valid = group.dropna(subset=['altitude', 'delta'])
    if len(valid) > 2:
        r = valid['altitude'].corr(valid['delta'])
        print(f"  {metric} | {prod}: r = {r:.3f} (n={len(valid)})")

# --- B5. Distribution plots of delta per metric/product ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, metric in zip(axes.flat, detection_metrics):
    sub = detection_long[detection_long['metric'] == metric]
    sns.boxplot(data=sub, x='product', y='delta', ax=ax)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_title(f'Station-level delta (2.5mm - 1.0mm) for {metric.upper()}')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'station_level_detection_delta_boxplot.png'), dpi=300)
plt.show()

# =============================================================================
# PART C - Save everything to one combined workbook
# =============================================================================
station_level_path = os.path.join(wkDir, 'station_level_threshold_sensitivity.xlsx')
with pd.ExcelWriter(station_level_path) as writer:
    # CDD/CWD
    cdd_bias.to_excel(writer, sheet_name='cdd_bias_long', index=False)
    cwd_bias.to_excel(writer, sheet_name='cwd_bias_long', index=False)
    cdd_flips.to_excel(writer, sheet_name='cdd_sign_flips')
    cwd_flips.to_excel(writer, sheet_name='cwd_sign_flips')
    cdd_outliers.to_excel(writer, sheet_name='cdd_outliers', index=False)
    cwd_outliers.to_excel(writer, sheet_name='cwd_outliers', index=False)
    # Detection stats
    detection_long.to_excel(writer, sheet_name='detection_long', index=False)
    detection_outliers.to_excel(writer, sheet_name='detection_outliers', index=False)
    winner_flips.to_excel(writer, sheet_name='detection_winner_flips', index=False)

print(f"\nAll station-level results saved to {station_level_path}")