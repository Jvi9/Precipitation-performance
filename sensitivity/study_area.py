# -*- coding: utf-8 -*-
"""
Verifies the Study Area section's numeric claims against the actual basin
shapefile and station network, with a focus on PRECISE sparsity: not just
an average density, but which basins have zero station coverage entirely,
and how much of the total area that represents.
"""
import geopandas as gpd
import pandas as pd

uh_shp_path = r'C:\Users\jvila\Desktop\Andean_project\gis\study_uhs\study_uhs.shp'
geoparquet_path = r'C:\Users\jvila\Desktop\Andean_project\outputs\station_metrics_1mm_clipped.parquet'

# =============================================================================
# 1. Basin count and total area
# =============================================================================
uh = gpd.read_file(uh_shp_path)
uh_equal_area = uh.to_crs('ESRI:102033')  # South America Albers Equal Area Conic
uh_equal_area['area_km2'] = uh_equal_area.geometry.area / 1e6

basin_col = 'NOMB_UH_N5' if 'NOMB_UH_N5' in uh.columns else uh.columns[0]
fallback_col = 'NOMB_UH_N4' if 'NOMB_UH_N4' in uh.columns else None
if fallback_col:
    uh_equal_area['basin_name'] = uh_equal_area[basin_col].fillna(uh_equal_area[fallback_col])
else:
    uh_equal_area['basin_name'] = uh_equal_area[basin_col]

print("="*90)
print("BASIN COUNT AND AREA CHECK")
print("="*90)
print(f"Number of basin polygons in shapefile: {len(uh)}")
print(f"Total area: {uh_equal_area['area_km2'].sum():,.0f} km^2")

# =============================================================================
# 2. Spatially join stations to basins - PER-BASIN station count and density,
#    so zero-coverage basins are visible directly rather than averaged away
# =============================================================================
gdf = gpd.read_parquet(geoparquet_path)
uh_latlon = uh.copy()
uh_latlon['basin_name'] = uh_latlon[basin_col].fillna(uh_latlon[fallback_col]) if fallback_col else uh_latlon[basin_col]
uh_latlon = uh_latlon.set_crs(gdf.crs, allow_override=True) if uh_latlon.crs != gdf.crs else uh_latlon

stations_joined = gdf.sjoin(uh_latlon[['basin_name', 'geometry']], how='left', predicate='within')
station_counts = stations_joined.groupby('basin_name').size()

basin_table = uh_equal_area[['basin_name', 'area_km2']].copy()
basin_table['n_stations'] = basin_table['basin_name'].map(station_counts).fillna(0).astype(int)
basin_table['density_per_1000km2'] = basin_table['n_stations'] / basin_table['area_km2'] * 1000
basin_table = basin_table.sort_values('area_km2', ascending=False)

print("\n" + "="*90)
print("PER-BASIN STATION COVERAGE (this is the real sparsity story)")
print("="*90)
for _, row in basin_table.iterrows():
    flag = "  <-- ZERO STATION COVERAGE" if row['n_stations'] == 0 else ""
    print(f"  {row['basin_name']:28s} area={row['area_km2']:>10,.0f} km^2   "
          f"n_stations={row['n_stations']:>3d}   density={row['density_per_1000km2']:.3f}/1000km^2{flag}")

# =============================================================================
# 3. Summary: how much of the total area has ZERO coverage
# =============================================================================
zero_coverage = basin_table[basin_table['n_stations'] == 0]
covered = basin_table[basin_table['n_stations'] > 0]

total_area = basin_table['area_km2'].sum()
zero_area = zero_coverage['area_km2'].sum()
zero_pct = zero_area / total_area * 100

print("\n" + "="*90)
print("SPARSITY SUMMARY")
print("="*90)
print(f"Basins with ZERO station coverage: {len(zero_coverage)} of {len(basin_table)}")
print(f"  ({', '.join(zero_coverage['basin_name'].tolist()) if len(zero_coverage) > 0 else 'none'})")
print(f"Area with zero coverage: {zero_area:,.0f} km^2 ({zero_pct:.1f}% of total study area)")
print(f"\nAmong COVERED basins only (excluding zero-coverage basins):")
print(f"  Total covered area: {covered['area_km2'].sum():,.0f} km^2")
print(f"  Stations in covered basins: {covered['n_stations'].sum()}")
print(f"  Density within covered basins only: "
      f"{covered['n_stations'].sum() / covered['area_km2'].sum() * 1000:.3f} stations per 1000 km^2")
print(f"\n(Compare this to the network-wide average density, which dilutes the real")
print(f"picture by spreading all 70 stations across the FULL area including basins")
print(f"that have no coverage at all.)")

# =============================================================================
# 4. Station elevation range - partial check, as before
# =============================================================================
print("\n" + "="*90)
print("STATION ELEVATION RANGE (lower bound only - see note)")
print("="*90)
print(f"Station elevation range: {gdf['alt'].min():.0f} m to {gdf['alt'].max():.0f} m")
print("NOTE: upper bound (6000+ m) reflects basin terrain from the DEM, not stations.")