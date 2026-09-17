# -*- coding: utf-8 -*-
"""
Created for the 1.0 vs 2.5 mm/day threshold sensitivity test requested by
the HESS editor / R1 / R2.

Requires the patched analysis_class.py and station_class.py (raise_extremeIndices,
Generator._run_extremeIndices, Station._extreme_indices all now accept/propagate
a wet_threshold so CDD/CWD use the SAME cutoff as FBI/FAR/POD/ACC).

Mirrors the structure of your existing driver script (Generator, join_stats,
sort_altitude, etc.) so it should drop in next to it with minimal changes.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analysis_class import *  # same import style as your driver script

# =============================================================================
# 0. Setup - matches your existing driver script
# =============================================================================
wkDir = r'C:\Users\jvila\Desktop\Andean_project'
os.chdir(wkDir)

cwd = os.getcwd()
load_data_path = f'{cwd}/datasets/data_locked_loaded.pkl'
min_range = '2005-01-01'
max_range = '2018-12-31'

out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

thresholds = {'1.0mm': 1.0, '2.5mm': 2.5}
generators = {}

for label, thr in thresholds.items():
    print(f"\n=== Running Generator with threshold = {thr} mm/day ===")
    generators[label] = Generator(load_data_path, min_range, max_range, thr)

gen_1mm = generators['1.0mm']
gen_25mm = generators['2.5mm']

# =============================================================================
# 1. Sanity check - continuous accuracy metrics should be IDENTICAL across
#    thresholds, since _performance() never touches min_obs_threshold at all.
#    If this comes back False, something else changed between runs - stop and
#    investigate before trusting anything below.
# =============================================================================
continuous_metrics = ['mae', 'pbias', 'rmse', 'r', 'kge']
cont_1mm = gen_1mm.join_stats(metrics=continuous_metrics)
cont_25mm = gen_25mm.join_stats(metrics=continuous_metrics)

identical = np.allclose(
    cont_1mm.values.astype(float), cont_25mm.values.astype(float), equal_nan=True
)
print(f"\nContinuous metrics (MAE/PBIAS/RMSE/R/KGE) identical across thresholds: {identical}")
if not identical:
    print("WARNING: continuous metrics differ between the two runs. They should not, "
          "since _performance() does not use min_obs_threshold. Check for stale state "
          "or a modified pickle between runs before trusting the rest of this script.")

# =============================================================================
# 2. Detection-capability stats: FBI / FAR / POD / ACC
# =============================================================================
detection_metrics = ['fbi', 'far', 'pod', 'acc']
detection_by_threshold = {
    label: gen.join_stats(metrics=detection_metrics)
    for label, gen in generators.items()
}

detection_means = pd.DataFrame({
    label: df.mean()
    for label, df in detection_by_threshold.items()
})
print("\n--- Mean detection stats across stations, by threshold ---")
print(detection_means.round(3))

# Does the product ranking change between thresholds?
print("\n--- Product rankings by threshold ---")
for metric in ['pod', 'far', 'acc']:
    cols = [c for c in detection_means.index if c.startswith(metric + '_')]
    for label in thresholds:
        ranked = detection_means.loc[cols, label].sort_values(ascending=(metric == 'far'))
        products = [c.replace(metric + '_', '') for c in ranked.index]
        print(f"  {metric.upper()} | {label}: {products}")

# =============================================================================
# 3. CDD / CWD (ETCCDI indices) - the part that changes with this patch
# =============================================================================
products = {
    'Observed': 'extremeIndicesObserved',
    'rawGPM': 'extremeIndicesrawGPM',
    'gwrGPM': 'extremeIndicesgwrGPM',
    'expGPM': 'extremeIndicesexpGPM',
    'PISCO': 'extremeIndicesPISCO',
    'rain4pe': 'extremeIndicesrain4pe',
}

def extract_cdd_cwd(generator, attr_name):
    ext_dict = getattr(generator, attr_name)
    rows = []
    for station, ei in ext_dict.items():
        rows.append({'station': station, 'cdd_mean': ei.cdd.mean(), 'cwd_mean': ei.cwd.mean()})
    return pd.DataFrame(rows).set_index('station')

summary_rows = []
for label, gen in generators.items():
    for prod_label, attr in products.items():
        df = extract_cdd_cwd(gen, attr)
        summary_rows.append({
            'threshold': label,
            'product': prod_label,
            'cdd_mean': df['cdd_mean'].mean(),
            'cwd_mean': df['cwd_mean'].mean(),
        })

cdd_cwd_summary = pd.DataFrame(summary_rows)
pivot_cdd = cdd_cwd_summary.pivot(index='product', columns='threshold', values='cdd_mean')
pivot_cwd = cdd_cwd_summary.pivot(index='product', columns='threshold', values='cwd_mean')
pivot_cdd['pct_change'] = (pivot_cdd['2.5mm'] - pivot_cdd['1.0mm']) / pivot_cdd['1.0mm'] * 100
pivot_cwd['pct_change'] = (pivot_cwd['2.5mm'] - pivot_cwd['1.0mm']) / pivot_cwd['1.0mm'] * 100

print("\n--- Mean CDD by product and threshold (with % change) ---")
print(pivot_cdd.round(2))
print("\n--- Mean CWD by product and threshold (with % change) ---")
print(pivot_cwd.round(2))

# =============================================================================
# 4. Plots
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
pivot_cdd[['1.0mm', '2.5mm']].plot(kind='bar', ax=axes[0])
axes[0].set_title('Mean CDD by product and threshold')
axes[0].set_ylabel('Days')
pivot_cwd[['1.0mm', '2.5mm']].plot(kind='bar', ax=axes[1])
axes[1].set_title('Mean CWD by product and threshold')
axes[1].set_ylabel('Days')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'threshold_sensitivity_cdd_cwd.png'), dpi=300)
plt.show()

fig2, ax2 = plt.subplots(figsize=(10, 6))
detection_means.T.plot(kind='bar', ax=ax2)
ax2.set_title('Detection-capability stats by threshold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'threshold_sensitivity_detection.png'), dpi=300)
plt.show()

# =============================================================================
# 5. Save everything for the manuscript / response letter
# =============================================================================
results_path = os.path.join(wkDir, 'threshold_sensitivity_results.xlsx')
with pd.ExcelWriter(results_path) as writer:
    detection_means.to_excel(writer, sheet_name='detection_stats_mean')
    cdd_cwd_summary.to_excel(writer, sheet_name='cdd_cwd_raw', index=False)
    pivot_cdd.to_excel(writer, sheet_name='cdd_pct_change')
    pivot_cwd.to_excel(writer, sheet_name='cwd_pct_change')

print(f"\nDone. Results saved to {results_path}")