# -*- coding: utf-8 -*-
"""
Rebuilds station.PISCO for every station DIRECTLY from PISCOp_daily.nc using
the confirmed-correct date index (z=1 -> 1981-01-01, daily, no gaps - verified
both empirically via lag correlation and arithmetically via the documented
1981-2019 period exactly matching n_z=14244 days). This replaces whatever the
original (lost) extraction script did, rather than patching its output.

Since the original extraction script is unavailable, this rebuilds from the
source rather than trying to shift the existing (wrongly-dated) station.PISCO
series - safer, since we can't be sure the old series had no OTHER issues
beyond the date offset (e.g. wrong lat/lon rounding, unit scaling, etc.) that
a simple date-shift wouldn't catch. This script re-derives PISCO from scratch
for every station and lets you compare against the old series as a sanity
check before trusting the replacement.

Output: data_locked_loaded_clipped_fixed.pkl - same structure as the input
pickle, with only station.PISCO replaced. Everything else (data, rawGPM,
gwrGPM, expGPM, rain4pe, lat/lon/alt) is left untouched.
"""
import os
import copy
import pickle
import numpy as np
import pandas as pd
import xarray as xr

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
input_pickle_path = f'{wkDir}/datasets/data_locked_loaded_clipped.pkl'
output_pickle_path = f'{wkDir}/datasets/data_locked_loaded_clipped_fixed.pkl'
pisco_nc_path = r'C:\Users\jvila\Downloads/PISCOp_daily.nc'
min_range = '2005-01-01'
max_range = '2018-12-31'

PRECIP_VAR = 'pc'
LAT_NAME = 'lat'
LON_NAME = 'lon'
Z_NAME = 'z'
CONFIRMED_START_DATE = '1981-01-01'  # see note above - confirmed both empirically and by day-count arithmetic

# =============================================================================
# 0. Load everything
# =============================================================================
with open(input_pickle_path, 'rb') as f:
    final_data = pickle.load(f)

ds = xr.open_dataset(pisco_nc_path)
n_z = ds.sizes[Z_NAME]
full_date_index = pd.date_range(start=CONFIRMED_START_DATE, periods=n_z, freq='D')
print(f"Rebuilding PISCO with {n_z} days, {full_date_index[0].date()} to {full_date_index[-1].date()}")

def prep_series(df, col, start, end):
    df = df.copy()
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    return df.loc[start:end, col]

# =============================================================================
# 1. Rebuild station.PISCO for every station
# =============================================================================
final_data_fixed = copy.deepcopy(final_data)

verification_rows = []

for station_name, station in final_data_fixed.items():
    lat, lon = float(station.lat), float(station.lon)

    # Extract raw values at nearest grid point, assign the CONFIRMED date index
    point = ds[PRECIP_VAR].sel({LAT_NAME: lat, LON_NAME: lon}, method='nearest')
    raw_values = point.values  # length n_z, in z-order = chronological order from CONFIRMED_START_DATE

    new_pisco_df = pd.DataFrame({'precipitationCal': raw_values}, index=full_date_index)

    # Keep only the project's analysis period, matching how other products are stored
    new_pisco_df = new_pisco_df.loc[min_range:max_range]

    # --- Verification: compare old vs new against observed, before committing ---
    obs = prep_series(station.data, 'Precipitation', min_range, max_range)
    old_pisco = prep_series(station.PISCO, 'precipitationCal', min_range, max_range)
    new_pisco = new_pisco_df['precipitationCal']

    def safe_corr(a, b):
        mask = a.notna() & b.notna()
        if mask.sum() < 30:
            return np.nan
        return a[mask].corr(b[mask])

    old_r = safe_corr(obs, old_pisco)
    new_r = safe_corr(obs, new_pisco)

    verification_rows.append({
        'station': station_name,
        'old_r_vs_observed': old_r,
        'new_r_vs_observed': new_r,
        'improvement': new_r - old_r if pd.notna(old_r) and pd.notna(new_r) else np.nan,
    })

    # --- Replace station.PISCO with the corrected series ---
    station.PISCO = new_pisco_df

verification_df = pd.DataFrame(verification_rows).sort_values('improvement', ascending=False)
print("\n" + "="*90)
print("VERIFICATION - old vs new PISCO correlation with observed, per station")
print("="*90)
print(verification_df.to_string(index=False))

n_improved = (verification_df['improvement'] > 0.05).sum()
n_worse = (verification_df['improvement'] < -0.05).sum()
print(f"\nStations improved (>0.05): {n_improved} / {len(verification_df)}")
print(f"Stations WORSE (<-0.05): {n_worse} / {len(verification_df)}  <- investigate any of these before trusting the fix")

verification_out = os.path.join(wkDir, 'pisco_fix_verification.csv')
verification_df.to_csv(verification_out, index=False)
print(f"\nVerification table saved to {verification_out}")

# =============================================================================
# 2. Save the fixed pickle - only if verification looks right
# =============================================================================
if n_worse > 0:
    print("\nWARNING: some stations got WORSE. Saving anyway, but inspect "
          "'pisco_fix_verification.csv' and the flagged stations before "
          "using this file for the manuscript.")

with open(output_pickle_path, 'wb') as f:
    pickle.dump(final_data_fixed, f)
print(f"\nSaved fixed dataset to {output_pickle_path}")
print("\nNext step: point load_data_path in your Generator-based scripts "
      "(sensitivity_test.py, build_station_geoparquet.py, etc.) at this new "
      "file and rerun the full pipeline.")