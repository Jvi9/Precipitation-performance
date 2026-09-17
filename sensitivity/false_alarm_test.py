# -*- coding: utf-8 -*-
"""
Dose-response version of the representativeness test: instead of lumping all
false alarms together, bins them by HOW MUCH rain the product reported on
that day. Tests the specific idea: "if GPM says 10mm and the gauge says 0,
that's a more credible signal of real missed rain than GPM saying 1.2mm and
the gauge saying 0" - i.e., does neighbor gauges' wetness (independent,
real evidence of regional rain) increase as the false-alarm magnitude
increases?

Same caveat as before: neighbor spacing in this network (see
mean_dist_km) is on the order of tens of km, so this tests regional rain
co-occurrence, not fine-scale (sub-km) representativeness specifically -
report the distance alongside any conclusion.
"""
import os
import pickle
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
clipped_fixed_pickle = f'{wkDir}/datasets/data_locked_loaded_clipped_fixed.pkl'
min_range = '2005-01-01'
max_range = '2018-12-31'

products = ['rawGPM', 'gwrGPM', 'expGPM', 'rain4pe', 'PISCO']
threshold = 1.0
K_NEIGHBORS = 3

# Magnitude bins for false-alarm days (based on the SIMULATED/product value
# reported that day) - edit as needed
MAGNITUDE_BINS = [1.0, 2.5, 5.0, 10.0, 20.0, np.inf]
MAGNITUDE_LABELS = ['1-2.5mm', '2.5-5mm', '5-10mm', '10-20mm', '>20mm']

with open(clipped_fixed_pickle, 'rb') as f:
    final_data = pickle.load(f)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def prep_series(df, col, start, end):
    df = df.copy()
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    return df.loc[start:end, col]

# =============================================================================
# 1. Neighbors + preloaded observed series (same as before)
# =============================================================================
station_names = list(final_data.keys())
coords = {s: (float(final_data[s].lat), float(final_data[s].lon)) for s in station_names}

neighbors = {}
for s in station_names:
    dists = [(o, haversine_km(*coords[s], *coords[o])) for o in station_names if o != s]
    dists.sort(key=lambda kv: kv[1])
    neighbors[s] = dists[:K_NEIGHBORS]

obs_series = {s: prep_series(final_data[s].data, 'Precipitation', min_range, max_range)
              for s in station_names}

# =============================================================================
# 2. For each station/product, collect (magnitude, neighbor_wet) per FA day
# =============================================================================
fa_rows = []

for s in station_names:
    obs_s = obs_series[s]
    neigh_list = neighbors[s]
    neigh_names = [n for n, _ in neigh_list]
    avg_neigh_dist = np.mean([d for _, d in neigh_list])

    neigh_obs = pd.concat([obs_series[n] for n in neigh_names], axis=1)
    neigh_obs.columns = neigh_names
    neigh_wet_frac = (neigh_obs >= threshold).mean(axis=1, skipna=True)

    for prod in products:
        sim = prep_series(getattr(final_data[s], prod), 'precipitationCal', min_range, max_range)
        mask = obs_s.notna() & sim.notna()
        o, si = obs_s[mask], sim[mask]

        fa_mask = (o < threshold) & (si >= threshold)
        fa_days = o[fa_mask].index
        fa_magnitudes = si.reindex(fa_days)
        fa_neigh_wet = neigh_wet_frac.reindex(fa_days)

        valid = fa_magnitudes.notna() & fa_neigh_wet.notna()
        for day in fa_days[valid]:
            fa_rows.append({
                'station': s,
                'product': prod,
                'avg_neighbor_dist_km': avg_neigh_dist,
                'magnitude': fa_magnitudes[day],
                'neighbor_wet': fa_neigh_wet[day],  # fraction (0 to 1) of K neighbors wet that day
            })

fa_df = pd.DataFrame(fa_rows)
fa_df['magnitude_bin'] = pd.cut(fa_df['magnitude'], bins=MAGNITUDE_BINS, labels=MAGNITUDE_LABELS, right=False)

# =============================================================================
# 3. Dose-response summary: neighbor wetness by magnitude bin, per product
# =============================================================================
print("="*90)
print("DOSE-RESPONSE: neighbor gauges' wetness on false-alarm days, BINNED BY")
print("HOW MUCH RAIN THE PRODUCT REPORTED that day")
print("="*90)

pivot = fa_df.groupby(['product', 'magnitude_bin'], observed=True)['neighbor_wet'].agg(['mean', 'count'])
for prod in products:
    print(f"\n--- {prod} ---")
    sub = pivot.loc[prod].reindex(MAGNITUDE_LABELS)
    print(sub.round(4).to_string())

# =============================================================================
# 4. Quantitative trend test: correlation between magnitude and neighbor wetness
# =============================================================================
print("\n" + "="*90)
print("TREND TEST: Spearman correlation between false-alarm magnitude and")
print("neighbor-gauge wetness (positive + significant = supports dose-response,")
print("i.e. bigger false alarms ARE more often corroborated by real regional rain)")
print("="*90)

from scipy.stats import spearmanr
for prod in products:
    sub = fa_df[fa_df['product'] == prod]
    if len(sub) > 30:
        rho, pval = spearmanr(sub['magnitude'], sub['neighbor_wet'])
        print(f"{prod}: n={len(sub)}, spearman r={rho:.4f}, p-value={pval:.2e}")
    else:
        print(f"{prod}: not enough false-alarm days for a reliable test (n={len(sub)})")

print(f"\nAverage neighbor distance across network: {fa_df['avg_neighbor_dist_km'].mean():.1f} km")
print("(Interpret the trend as regional rain co-occurrence at this spacing - see caveat)")

# =============================================================================
# 5. Save
# =============================================================================
out_path = os.path.join(wkDir, 'dose_response_false_alarms.csv')
fa_df.to_csv(out_path, index=False)
print(f"\nFull per-event false-alarm data saved to {out_path}")