# -*- coding: utf-8 -*-
"""
Tests TOTAL RAINFALL AMOUNT properly - your PBIAS numbers so far are the
overall bias summed across the whole 14-year record, which can hide a lot:
  - a product could be right on average but wildly inconsistent year to year
  - the network-mean PBIAS could hide large station-to-station cancellation
    (some stations badly over-, others under-, estimated)
  - the bias could be concentrated in one season (e.g. wet-season totals
    badly biased while dry-season totals are fine, or vice versa)

This script computes:
  1. OVERALL total volume bias per station (recomputed fresh here, with
     the full distribution across 70 stations - not just the network mean)
  2. ANNUAL total bias: for each of the 14 years, is the bias consistent,
     or does it swing between over- and under-estimating year to year?
  3. WET (DJF) vs DRY (JJA) season total volume bias - extends your earlier
     correlation-based wet/dry split to actual rainfall AMOUNTS
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
WET_MONTHS = [12, 1, 2]
DRY_MONTHS = [6, 7, 8]

with open(clipped_fixed_pickle, 'rb') as f:
    final_data = pickle.load(f)

def prep_series(df, col, start, end):
    df = df.copy()
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    return df.loc[start:end, col]

def pct_bias(sim, obs):
    """PBIAS on the overlapping valid-data period only."""
    mask = sim.notna() & obs.notna()
    s, o = sim[mask], obs[mask]
    if o.sum() == 0:
        return np.nan
    return (s.sum() - o.sum()) / o.sum() * 100

# =============================================================================
# 1. OVERALL total volume bias - full distribution, not just the mean
# =============================================================================
overall_rows = []
for station_name, station in final_data.items():
    obs = prep_series(station.data, 'Precipitation', min_range, max_range)
    for prod in products:
        sim = prep_series(getattr(station, prod), 'precipitationCal', min_range, max_range)
        overall_rows.append({
            'station': station_name,
            'product': prod,
            'pbias_overall': pct_bias(sim, obs),
        })
overall_df = pd.DataFrame(overall_rows)

print("="*90)
print("1. OVERALL TOTAL VOLUME BIAS - full distribution across 70 stations")
print("="*90)
summary1 = overall_df.groupby('product')['pbias_overall'].agg(
    ['mean', 'median', 'std', 'min', 'max']
).reindex(products)
print(summary1.round(2).to_string())
print("\n(Large 'std' relative to 'mean' = network mean is hiding a lot of")
print("station-to-station cancellation - some badly over-, others under-estimated)")

# =============================================================================
# 2. ANNUAL total bias - is it consistent year to year, or does it swing?
# =============================================================================
annual_rows = []
for station_name, station in final_data.items():
    obs = prep_series(station.data, 'Precipitation', min_range, max_range)
    for prod in products:
        sim = prep_series(getattr(station, prod), 'precipitationCal', min_range, max_range)
        for year, obs_year in obs.groupby(obs.index.year):
            sim_year = sim[sim.index.year == year]
            b = pct_bias(sim_year, obs_year)
            annual_rows.append({
                'station': station_name, 'product': prod, 'year': year, 'pbias_annual': b,
            })
annual_df = pd.DataFrame(annual_rows)

print("\n" + "="*90)
print("2. ANNUAL TOTAL BIAS - consistency across years (network mean per product)")
print("="*90)
summary2 = annual_df.groupby('product')['pbias_annual'].agg(
    ['mean', 'std']
).reindex(products)
summary2.columns = ['mean_annual_pbias', 'std_across_station_years']
print(summary2.round(2).to_string())
print("\n('std_across_station_years' = how much the annual bias swings around")
print("its own mean, across all station-years - high value = inconsistent,")
print("sometimes over- sometimes under-estimating, even if the long-run mean looks fine)")

print("\n--- Fraction of station-years where annual bias has OPPOSITE sign from")
print("    that product's own overall mean bias direction ---")
for prod in products:
    overall_mean_sign = np.sign(overall_df[overall_df['product'] == prod]['pbias_overall'].mean())
    sub = annual_df[annual_df['product'] == prod]['pbias_annual'].dropna()
    opposite_frac = (np.sign(sub) != overall_mean_sign).mean()
    print(f"{prod}: {opposite_frac:.1%} of station-years flip sign vs overall direction")

# =============================================================================
# 3. WET (DJF) vs DRY (JJA) season total volume bias
# =============================================================================
season_rows = []
for station_name, station in final_data.items():
    obs = prep_series(station.data, 'Precipitation', min_range, max_range)
    for prod in products:
        sim = prep_series(getattr(station, prod), 'precipitationCal', min_range, max_range)
        for season_name, months in [('Wet (DJF)', WET_MONTHS), ('Dry (JJA)', DRY_MONTHS)]:
            obs_s = obs[obs.index.month.isin(months)]
            sim_s = sim[sim.index.month.isin(months)]
            b = pct_bias(sim_s, obs_s)
            season_rows.append({
                'station': station_name, 'product': prod, 'season': season_name, 'pbias_season': b,
            })
season_df = pd.DataFrame(season_rows)

print("\n" + "="*90)
print("3. WET-SEASON vs DRY-SEASON TOTAL VOLUME BIAS")
print("="*90)
summary3 = season_df.groupby(['product', 'season'])['pbias_season'].mean().unstack()
summary3 = summary3.reindex(products)
print(summary3.round(2).to_string())
print("\n(If wet-season bias is very different from dry-season bias, the overall")
print("PBIAS is an average of two quite different error regimes, which matters a")
print("lot for hydrological applications where wet-season totals dominate runoff)")

# =============================================================================
# Save everything
# =============================================================================
overall_df.to_csv(os.path.join(wkDir, 'total_bias_overall.csv'), index=False)
annual_df.to_csv(os.path.join(wkDir, 'total_bias_annual.csv'), index=False)
season_df.to_csv(os.path.join(wkDir, 'total_bias_season.csv'), index=False)
print(f"\nAll three tables saved to {wkDir}")



# -*- coding: utf-8 -*-
"""
Visualizes the three total-rainfall-amount analyses in one 2x2 figure:
  A. Overall PBIAS distribution per product (boxplot across 70 stations) -
     shows PISCO's tight spread vs. Rain4PE/expGPM's wide spread
  B. Wet (DJF) vs dry (JJA) season bias per product (grouped bars) - shows
     the universal seasonal reversal
  C. Annual PBIAS trajectory per product, year by year (line plot) - shows
     how much bias swings year to year, visually
  D. Sign-flip fraction per product (bar chart) - how often a station-year's
     bias direction disagrees with that product's own long-run average

Reads the three CSVs saved by total_amount_test.py - run that first.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

products = ['rawGPM', 'gwrGPM', 'expGPM', 'rain4pe', 'PISCO']
PRODUCT_COLORS = {
    'rawGPM': '#1f77b4', 'gwrGPM': '#2ca02c', 'expGPM': '#d62728',
    'rain4pe': '#9467bd', 'PISCO': '#ff7f0e',
}

overall_df = pd.read_csv(os.path.join(wkDir, 'total_bias_overall.csv'))
annual_df = pd.read_csv(os.path.join(wkDir, 'total_bias_annual.csv'))
season_df = pd.read_csv(os.path.join(wkDir, 'total_bias_season.csv'))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# =============================================================================
# A. Overall PBIAS distribution (boxplot across stations)
# =============================================================================
ax = axes[0, 0]
data_for_box = [overall_df[overall_df['product'] == p]['pbias_overall'].dropna() for p in products]
bp = ax.boxplot(data_for_box, labels=products, patch_artist=True, showfliers=True)
for patch, p in zip(bp['boxes'], products):
    patch.set_facecolor(PRODUCT_COLORS[p])
    patch.set_alpha(0.6)
ax.axhline(0, color='k', linestyle='--', linewidth=1)
ax.set_title('A. Overall PBIAS distribution across 70 stations', fontsize=13)
ax.set_ylabel('PBIAS (%)')
ax.grid(axis='y', alpha=0.3)

# =============================================================================
# B. Wet vs dry season bias (grouped bars)
# =============================================================================
ax = axes[0, 1]
season_summary = season_df.groupby(['product', 'season'])['pbias_season'].mean().unstack().reindex(products)
x = np.arange(len(products))
width = 0.35
ax.bar(x - width/2, season_summary['Dry (JJA)'], width, label='Dry (JJA)', color='#d95f02', alpha=0.85)
ax.bar(x + width/2, season_summary['Wet (DJF)'], width, label='Wet (DJF)', color='#1b9e77', alpha=0.85)
ax.axhline(0, color='k', linestyle='--', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(products)
ax.set_title('B. Wet-season vs dry-season total volume bias', fontsize=13)
ax.set_ylabel('PBIAS (%)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# =============================================================================
# C. Annual PBIAS trajectory per product (network mean per year)
# =============================================================================
ax = axes[1, 0]
annual_summary = annual_df.groupby(['product', 'year'])['pbias_annual'].mean().unstack(level=0).reindex(columns=products)
for p in products:
    ax.plot(annual_summary.index, annual_summary[p], marker='o', markersize=4,
            label=p, color=PRODUCT_COLORS[p])
ax.axhline(0, color='k', linestyle='--', linewidth=1)
ax.set_title('C. Annual PBIAS by year (network mean)', fontsize=13)
ax.set_xlabel('Year')
ax.set_ylabel('PBIAS (%)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# =============================================================================
# D. Sign-flip fraction per product
# =============================================================================
ax = axes[1, 1]
flip_fracs = []
for p in products:
    overall_mean_sign = np.sign(overall_df[overall_df['product'] == p]['pbias_overall'].mean())
    sub = annual_df[annual_df['product'] == p]['pbias_annual'].dropna()
    flip_frac = (np.sign(sub) != overall_mean_sign).mean()
    flip_fracs.append(flip_frac * 100)
bars = ax.bar(products, flip_fracs, color=[PRODUCT_COLORS[p] for p in products], alpha=0.85)
ax.axhline(50, color='gray', linestyle=':', linewidth=1, label='50% (coin-flip)')
ax.set_title('D. Station-years with bias sign opposite to\nproduct\'s long-run average direction', fontsize=13)
ax.set_ylabel('% of station-years')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, flip_fracs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.0f}%', ha='center', fontsize=10)

plt.tight_layout()
out_path = os.path.join(out_dir, 'total_rainfall_amount_summary.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved {out_path}")