# -*- coding: utf-8 -*-
"""
Final recomputation pipeline for the manuscript, using:
  - the negative-value-clipped dataset (data_locked_loaded_clipped.pkl)
  - the symmetric-threshold Station class (station_class_symmetric_threshold.py
    renamed to station_class.py, or imported directly - see note below)
  - a single unified threshold of 1.0 mm/day (ETCCDI convention) for BOTH
    detection stats (FBI/FAR/POD/ACC) and CDD/CWD

Produces ONE row per station with every metric for every product, plus
lat/lon geometry, saved as a single GeoParquet file for downstream plotting.

REQUIRES: geopandas, shapely, pyarrow
    pip install geopandas shapely pyarrow --break-system-packages   (if needed)

IMPORTANT: this script assumes analysis_class.py and station_class.py in your
working directory already have the patches from earlier in this conversation:
  - station_class.py: _extreme_indices(wet_threshold=...) and the symmetric
    threshold fix in _detection_capability
  - analysis_class.py: raise_extremeIndices(wet_threshold=...) and
    Generator._run_extremeIndices() passing wet_threshold=self.min_threshold
If station_class_symmetric_threshold.py is a separate file, copy its content
into station_class.py (overwriting) before running this script.
"""
import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from analysis_class import *  # Generator, productStats, extremeIndices, etc.

# =============================================================================
# 0. Setup
# =============================================================================
wkDir = r'C:\Users\jvila\Desktop\Andean_project'
os.chdir(wkDir)

clipped_data_path = f'{wkDir}/datasets/data_locked_loaded_clipped_fixed.pkl'
min_range = '2005-01-01'
max_range = '2018-12-31'
min_threshold = 1.0  # unified threshold, ETCCDI convention

out_dir = os.path.join(wkDir, 'outputs')
os.makedirs(out_dir, exist_ok=True)

print(f"Running Generator on clipped dataset with threshold = {min_threshold} mm/day ...")
gen = Generator(clipped_data_path, min_range, max_range, min_threshold)

products = {
    'rawGPM': ('statsrawGPM_dict', 'extremeIndicesrawGPM'),
    'gwrGPM': ('statsgwrGPM_dict', 'extremeIndicesgwrGPM'),
    'expGPM': ('statsexpGPM_dict', 'extremeIndicesexpGPM'),
    'rain4pe': ('statsrain4pe_dict', 'extremeIndicesrain4pe'),
    'PISCO': ('statsPISCO_dict', 'extremeIndicesPISCO'),
}
observed_extreme_attr = 'extremeIndicesObserved'

continuous_metrics = ['mae', 'pbias', 'rmse', 'r', 'kge']
detection_metrics = ['fbi', 'far', 'pod', 'acc']
extreme_indices = ['cdd', 'cwd', 'r10', 'r20', 'r95p', 'r99p']

# =============================================================================
# 1. Geo summary (lat/lon/alt) - the base of every row
# =============================================================================
geo = gen.geo_summary().set_index('station')
geo['lat'] = pd.to_numeric(geo['lat'], errors='coerce')
geo['lon'] = pd.to_numeric(geo['lon'], errors='coerce')
geo['alt'] = pd.to_numeric(geo['alt'], errors='coerce')

station_names = list(geo.index)
wide = pd.DataFrame(index=station_names)
wide['lat'] = geo['lat']
wide['lon'] = geo['lon']
wide['alt'] = geo['alt']

# =============================================================================
# 2. Continuous + detection stats per product (from join_stats / parameters_report)
# =============================================================================
for prod, (stats_attr, _) in products.items():
    stats_dict = getattr(gen, stats_attr)
    report = parameters_report(stats_dict)  # station x metric
    for metric in continuous_metrics + detection_metrics:
        col_name = f"{metric}_{prod}"
        wide[col_name] = report[metric].reindex(station_names)

# =============================================================================
# 3. Extreme indices - mean across years per station per product
# =============================================================================
def mean_extreme_indices(ext_dict, index_names):
    """ext_dict: {station: extremeIndices instance}. Returns station -> {index: mean}."""
    rows = {}
    for station, ei in ext_dict.items():
        rows[station] = {idx: getattr(ei, idx).mean() for idx in index_names}
    return pd.DataFrame(rows).T  # station x index

for prod, (_, ext_attr) in products.items():
    ext_dict = getattr(gen, ext_attr)
    ext_df = mean_extreme_indices(ext_dict, extreme_indices)
    for idx in extreme_indices:
        col_name = f"{idx}_{prod}"
        wide[col_name] = ext_df[idx].reindex(station_names)

# Observed extreme indices too (useful for bias calcs downstream, e.g. in plotting)
obs_ext_dict = getattr(gen, observed_extreme_attr)
obs_ext_df = mean_extreme_indices(obs_ext_dict, extreme_indices)
for idx in extreme_indices:
    wide[f"{idx}_Observed"] = obs_ext_df[idx].reindex(station_names)

# =============================================================================
# 4. Build GeoDataFrame and save
# =============================================================================
wide = wide.reset_index().rename(columns={'index': 'station'})
geometry = [Point(lon, lat) for lon, lat in zip(wide['lon'], wide['lat'])]
gdf = gpd.GeoDataFrame(wide, geometry=geometry, crs="EPSG:4326")

print(f"\nFinal table shape: {gdf.shape[0]} stations x {gdf.shape[1]} columns")
print(f"Columns: {list(gdf.columns)}")

geoparquet_path = os.path.join(out_dir, 'station_metrics_1mm_clipped.parquet')
gdf.to_parquet(geoparquet_path)
print(f"\nSaved GeoParquet to {geoparquet_path}")

# Also save a plain CSV (no geometry) for quick inspection outside GIS tools
csv_path = os.path.join(out_dir, 'station_metrics_1mm_clipped.csv')
wide.to_csv(csv_path, index=False)
print(f"Saved CSV (non-spatial) to {csv_path}")

# =============================================================================
# 5. Quick sanity print
# =============================================================================
print("\n--- Sanity check: first 5 stations, key columns ---")
sanity_cols = ['station', 'lat', 'lon', 'alt', 'kge_rain4pe', 'pod_rain4pe',
               'cdd_rain4pe', 'cdd_Observed']
print(gdf[sanity_cols].head().to_string())