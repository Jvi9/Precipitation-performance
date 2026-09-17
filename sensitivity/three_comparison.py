# -*- coding: utf-8 -*-
"""
Three comparison analyses, in one script, using the PISCO-date-fixed data:

  1. DOWNSCALING VALUE-ADDED SCATTER
     rawGPM vs. gwrGPM/expGPM, one point per station, with a 1:1 reference
     line. Points above the line = downscaling improved on the parent
     product at that station; below = it made things worse. Directly
     answers "did downscaling add value" as a per-station count, not just
     a qualitative claim.

  2. SKILL VS. ALTITUDE
     Stations binned by elevation; each product's median skill per bin,
     plotted as boxplots. Tests whether performance degrades at high
     altitude for every product (a generalizable Andes finding) or only
     for some.

  3. WET-SEASON VS. DRY-SEASON SPLIT
     Correlation with observed, computed separately for DJF (wet) and JJA
     (dry) months, per station per product. Uses the raw clipped+fixed
     pickle directly (season splits aren't in the GeoParquet, which only
     holds whole-period aggregates).

Sections 1 and 2 read from the GeoParquet (annual aggregate stats).
Section 3 reads from the raw pickle (needs daily data to split by month).
"""
import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
geoparquet_path = os.path.join(wkDir, 'outputs', 'station_metrics_1mm_clipped.parquet')  # rebuilt after PISCO fix
clipped_fixed_pickle = f'{wkDir}/datasets/data_locked_loaded_clipped_fixed.pkl'
min_range = '2005-01-01'
max_range = '2018-12-31'
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

products = ['rawGPM', 'gwrGPM', 'expGPM', 'rain4pe', 'PISCO']
downscaled_products = ['gwrGPM', 'expGPM']  # compared against rawGPM as their parent
PRODUCT_COLORS = {
    'rawGPM': '#1f77b4', 'gwrGPM': '#2ca02c', 'expGPM': '#d62728',
    'rain4pe': '#9467bd', 'PISCO': '#ff7f0e',
}

gdf = gpd.read_parquet(geoparquet_path)

# =============================================================================
# 1. DOWNSCALING VALUE-ADDED SCATTER
# =============================================================================
print("="*90)
print("1. DOWNSCALING VALUE-ADDED SCATTER")
print("="*90)

scatter_metric = 'kge'  # edit to 'pod', 'r', etc. - must exist as f'{metric}_{product}' in the GeoParquet

fig, axes = plt.subplots(1, len(downscaled_products), figsize=(6 * len(downscaled_products), 6))
if len(downscaled_products) == 1:
    axes = [axes]

for ax, prod in zip(axes, downscaled_products):
    x = gdf[f'{scatter_metric}_rawGPM']
    y = gdf[f'{scatter_metric}_{prod}']
    improved = y > x

    ax.scatter(x[improved], y[improved], color=PRODUCT_COLORS[prod], alpha=0.7,
               label=f'Improved (n={improved.sum()})', edgecolor='black', linewidth=0.3)
    ax.scatter(x[~improved], y[~improved], color='gray', alpha=0.5,
               label=f'Degraded/unchanged (n={(~improved).sum()})', edgecolor='black', linewidth=0.3)

    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', linewidth=1, label='1:1 line')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f'rawGPM {scatter_metric.upper()}')
    ax.set_ylabel(f'{prod} {scatter_metric.upper()}')
    ax.set_title(f'rawGPM vs {prod}')
    ax.legend(fontsize=9)
    ax.set_aspect('equal')

    print(f"\n{prod} vs rawGPM parent ({scatter_metric.upper()}):")
    print(f"  Improved at {improved.sum()} / {len(gdf)} stations "
          f"({100*improved.sum()/len(gdf):.1f}%)")
    print(f"  Mean change: {(y - x).mean():+.4f}")

plt.tight_layout()
out_path = os.path.join(out_dir, f'downscaling_value_added_{scatter_metric}.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved {out_path}")

# =============================================================================
# 2. SKILL VS. ALTITUDE
# =============================================================================
print("\n" + "="*90)
print("2. SKILL VS. ALTITUDE")
print("="*90)

altitude_metric = 'kge'  # edit as needed
alt_bins = [0, 2000, 3000, 4000, 6000]
alt_labels = ['<2000m', '2000-3000m', '3000-4000m', '>4000m']

alt_df = gdf[['alt']].copy()
alt_df['alt_bin'] = pd.cut(gdf['alt'], bins=alt_bins, labels=alt_labels)
for prod in products:
    alt_df[prod] = gdf[f'{altitude_metric}_{prod}']

alt_long = alt_df.melt(id_vars=['alt', 'alt_bin'], value_vars=products,
                        var_name='product', value_name=altitude_metric)

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=alt_long, x='alt_bin', y=altitude_metric, hue='product', ax=ax,
            palette=PRODUCT_COLORS, order=alt_labels)
ax.set_title(f'{altitude_metric.upper()} by elevation band and product')
ax.set_xlabel('Elevation band')
ax.set_ylabel(altitude_metric.upper())
ax.legend(title='Product', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
out_path = os.path.join(out_dir, f'skill_vs_altitude_{altitude_metric}.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved {out_path}")

print(f"\n--- Median {altitude_metric.upper()} by elevation band and product ---")
print(alt_long.groupby(['alt_bin', 'product'], observed=True)[altitude_metric].median().unstack().to_string())

n_per_bin = gdf.groupby(pd.cut(gdf['alt'], bins=alt_bins, labels=alt_labels), observed=True).size()
print(f"\nStations per elevation band:\n{n_per_bin.to_string()}")

# =============================================================================
# 3. WET-SEASON VS. DRY-SEASON SPLIT (needs raw daily data, not the GeoParquet)
# =============================================================================
print("\n" + "="*90)
print("3. WET-SEASON VS. DRY-SEASON CORRELATION SPLIT")
print("="*90)

WET_MONTHS = [12, 1, 2]   # DJF
DRY_MONTHS = [6, 7, 8]    # JJA

with open(clipped_fixed_pickle, 'rb') as f:
    final_data = pickle.load(f)

def prep_series(df, col, start, end):
    df = df.copy()
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    return df.loc[start:end, col]

season_rows = []
for station_name, station in final_data.items():
    obs = prep_series(station.data, 'Precipitation', min_range, max_range)
    for prod in products:
        sim = prep_series(getattr(station, prod), 'precipitationCal', min_range, max_range)
        for season_name, months in [('Wet (DJF)', WET_MONTHS), ('Dry (JJA)', DRY_MONTHS)]:
            obs_s = obs[obs.index.month.isin(months)]
            sim_s = sim[sim.index.month.isin(months)]
            mask = obs_s.notna() & sim_s.notna()
            if mask.sum() < 30:
                r = np.nan
            else:
                r = obs_s[mask].corr(sim_s[mask])
            season_rows.append({
                'station': station_name, 'product': prod, 'season': season_name, 'r': r,
            })

season_df = pd.DataFrame(season_rows)

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=season_df, x='product', y='r', hue='season', ax=ax, order=products)
ax.set_title('Correlation with observed: wet season (DJF) vs dry season (JJA)')
ax.set_ylabel('Correlation (r)')
ax.set_xlabel('')
plt.tight_layout()
out_path = os.path.join(out_dir, 'wet_vs_dry_season_correlation.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved {out_path}")

print("\n--- Mean correlation by product and season ---")
print(season_df.groupby(['product', 'season'])['r'].mean().unstack().to_string())

# =============================================================================
# Save everything
# =============================================================================
alt_long.to_csv(os.path.join(wkDir, 'skill_vs_altitude.csv'), index=False)
season_df.to_csv(os.path.join(wkDir, 'wet_vs_dry_season.csv'), index=False)
print(f"\nAll underlying tables saved to {wkDir}")