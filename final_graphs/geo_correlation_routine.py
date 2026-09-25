# -*- coding: utf-8 -*-
"""
COMPREHENSIVE ROUTINE testing multiple hypotheses about what drives
performance beyond elevation, all in one place, for a chosen set of metrics
and all 5 products:

  A. ELEVATION MAIN EFFECT       - R^2 of metric ~ altitude alone (baseline)
  B. BASIN MAIN EFFECT            - does adding basin identity (categorical)
                                     explain more variance than elevation alone?
  C. ELEVATION x BASIN INTERACTION - does allowing a DIFFERENT elevation slope
                                     per basin explain even more (i.e. is the
                                     elevation effect itself basin-dependent)?
  D. LOW-ELEVATION COVERAGE CHECK  - across basins, does the number/share of
                                     stations below a low-elevation threshold
                                     predict how strong that basin's elevation
                                     slope was (from the per-basin fits)?
  E. ORTHOGONAL GEOGRAPHIC AXIS    - lat/lon are collinear (rho=-0.835) in this
                                     network, so raw lat/lon regression is
                                     unstable (see earlier result). This
                                     computes the first principal component of
                                     (lat, lon) - a single, uncorrelated-by-
                                     construction "geographic axis" - and
                                     tests whether IT predicts the elevation-
                                     adjusted residual, without the
                                     multicollinearity problem.

All model comparisons use R^2 (and adjusted R^2, since models differ in the
number of parameters) - this quantifies, not just visualizes, how much each
additional hypothesis actually adds beyond the previous one.

Basins with fewer than MIN_BASIN_N stations are grouped into a single
"Other" category for the basin-effect models, so no station is dropped,
while keeping the reliability caveat explicit in the printed output.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd

from configurations import PLOT_ORDER, DISPLAY_NAMES

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
geoparquet_path = os.path.join(wkDir, 'outputs', 'station_metrics_1mm_clipped.parquet')

METRICS = ['rmse', 'pbias', 'kge', 'r']   # edit to add/remove metrics
MIN_BASIN_N = 6                            # basins with fewer stations -> "Other"
LOW_ELEV_THRESHOLD = 1000                  # meters, for the low-elevation coverage check

gdf = gpd.read_parquet(geoparquet_path)
gdf['basin'] = gdf['station'].apply(lambda s: s.split('_')[0].strip())

basin_counts = gdf['basin'].value_counts()
small_basins = basin_counts[basin_counts < MIN_BASIN_N].index.tolist()
gdf['basin_grouped'] = gdf['basin'].apply(lambda b: 'Other (small basin)' if b in small_basins else b)

print("Basin grouping for the categorical models (small basins pooled into 'Other'):")
print(gdf['basin_grouped'].value_counts().to_string())
print(f"\n(Pooled basins, kept separate for the low-elevation-coverage check: {small_basins})")


# =============================================================================
# Generic OLS helpers
# =============================================================================
def fit_r2(y, X_columns):
    """X_columns: list of 1D arrays (already includes any dummy columns).
    Returns R^2 and adjusted R^2 for y ~ X_columns (+ intercept)."""
    X = np.column_stack([np.ones(len(y))] + X_columns)
    mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    X, y = X[mask], y[mask]
    n, p = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p) if n > p else np.nan
    return r2, adj_r2, n


def basin_dummies(basin_series, drop_first=True):
    dummies = pd.get_dummies(basin_series, drop_first=drop_first).astype(float)
    return [dummies[col].values for col in dummies.columns], list(dummies.columns)


# =============================================================================
# A/B/C - nested model comparison, per metric per product
# =============================================================================
print("\n" + "="*95)
print("A/B/C: NESTED MODEL COMPARISON (R^2) - does BASIN add anything beyond")
print("ELEVATION, and does an ELEVATION x BASIN INTERACTION add anything beyond that?")
print("="*95)

summary_rows = []
for metric in METRICS:
    print(f"\n{'#'*95}\n# METRIC: {metric.upper()}\n{'#'*95}")
    for code in PLOT_ORDER:
        y = gdf[f'{metric}_{code}'].values
        alt = gdf['alt'].values

        r2_alt, adj_alt, n = fit_r2(y, [alt])

        basin_cols, basin_names = basin_dummies(gdf['basin_grouped'], drop_first=True)
        r2_basin, adj_basin, _ = fit_r2(y, [alt] + basin_cols)

        # Model C properly NESTS model B: same intercept dummies, PLUS a
        # slope-deviation term (dummy * altitude) for every non-reference
        # basin, so the reference basin's slope is the plain 'alt'
        # coefficient and each other basin's slope = alt coef + its own
        # deviation term. This is the standard ANCOVA interaction setup -
        # model C can only fit AT LEAST as well as model B, never worse.
        interaction_cols = [dummy * alt for dummy in basin_cols]
        r2_inter, adj_inter, _ = fit_r2(y, [alt] + basin_cols + interaction_cols)

        delta_basin = r2_basin - r2_alt
        delta_inter = r2_inter - r2_basin

        print(f"\n{DISPLAY_NAMES[code]}:")
        print(f"  (A) elevation only:              R2={r2_alt:.3f}  adjR2={adj_alt:.3f}")
        print(f"  (B) + basin (additive):           R2={r2_basin:.3f}  adjR2={adj_basin:.3f}   "
              f"(delta R2 from adding basin: {delta_basin:+.3f})")
        print(f"  (C) + basin x elevation interact.: R2={r2_inter:.3f}  (relative to B, "
              f"delta R2: {delta_inter:+.3f})")

        summary_rows.append({
            'metric': metric, 'product': DISPLAY_NAMES[code],
            'r2_elevation': r2_alt, 'r2_plus_basin': r2_basin, 'r2_plus_interaction': r2_inter,
            'delta_basin': delta_basin, 'delta_interaction': delta_inter,
        })

summary_df = pd.DataFrame(summary_rows)

# =============================================================================
# D - low-elevation coverage vs. per-basin slope strength
# =============================================================================
print("\n" + "="*95)
print(f"D: LOW-ELEVATION COVERAGE (< {LOW_ELEV_THRESHOLD}m) vs. per-basin elevation-RMSE slope")
print("(exploratory only - very few basins have a reliable slope to compare against)")
print("="*95)

low_elev_counts = gdf.groupby('basin').apply(
    lambda d: (d['alt'] < LOW_ELEV_THRESHOLD).sum(), include_groups=False)
low_elev_share = gdf.groupby('basin').apply(
    lambda d: (d['alt'] < LOW_ELEV_THRESHOLD).mean(), include_groups=False)

print(f"\nStations below {LOW_ELEV_THRESHOLD}m, per basin:")
coverage_table = pd.DataFrame({
    'n_stations': basin_counts,
    f'n_below_{LOW_ELEV_THRESHOLD}m': low_elev_counts,
    f'share_below_{LOW_ELEV_THRESHOLD}m': low_elev_share.round(2),
}).sort_values(f'n_below_{LOW_ELEV_THRESHOLD}m', ascending=False)
print(coverage_table.to_string())

print("\n(Compare this table's ranking directly against the per-basin slope")
print("strength from the previous script - basins near the top here with also")
print("a strong elevation slope support the 'needs low-elevation coverage to")
print("detect the effect' hypothesis; a mismatch would argue against it.)")

# =============================================================================
# E - orthogonal geographic axis (PCA of lat/lon) vs. elevation-adjusted residual
# =============================================================================
print("\n" + "="*95)
print("E: ORTHOGONAL GEOGRAPHIC AXIS (1st principal component of lat/lon)")
print("(avoids the lat/lon collinearity problem - a single composite axis,")
print("uncorrelated with itself by construction, testing whether SOME")
print("direction through lat/lon space explains residual variance)")
print("="*95)

latlon = gdf[['lat', 'lon']].values
latlon_centered = latlon - latlon.mean(axis=0)
U, S, Vt = np.linalg.svd(latlon_centered, full_matrices=False)
geo_axis = latlon_centered @ Vt[0]  # projection onto the 1st principal component
gdf['geo_axis'] = geo_axis
print(f"\n1st principal component explains {S[0]**2 / np.sum(S**2):.1%} of lat/lon's joint variance")
print(f"Loadings (direction of this axis in lat/lon space): lat={Vt[0][0]:+.3f}, lon={Vt[0][1]:+.3f}")

for metric in METRICS:
    print(f"\n--- {metric.upper()}: does geo_axis add anything beyond elevation? ---")
    for code in PLOT_ORDER:
        y = gdf[f'{metric}_{code}'].values
        alt = gdf['alt'].values
        r2_alt, _, _ = fit_r2(y, [alt])
        r2_alt_geo, _, _ = fit_r2(y, [alt, geo_axis])
        delta = r2_alt_geo - r2_alt
        flag = '  <- geo_axis adds meaningfully' if delta > 0.05 else ''
        print(f"  {DISPLAY_NAMES[code]:12s} R2(alt)={r2_alt:.3f}  R2(alt+geo_axis)={r2_alt_geo:.3f}  "
              f"delta={delta:+.3f}{flag}")

# =============================================================================
# Final synthesis table
# =============================================================================
print("\n" + "="*95)
print("SYNTHESIS: mean delta-R2 from adding basin / interaction, averaged across products")
print("(the biggest number tells you which hypothesis matters most, on average)")
print("="*95)
synthesis = summary_df.groupby('metric')[['delta_basin', 'delta_interaction']].mean()
print(synthesis.round(3).to_string())

out_csv = os.path.join(wkDir, 'regional_effects_routine_summary.csv')
summary_df.to_csv(out_csv, index=False)
print(f"\nFull per-metric, per-product results saved to {out_csv}")