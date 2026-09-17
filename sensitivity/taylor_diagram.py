# -*- coding: utf-8 -*-
"""
Taylor diagram: standard climate/hydrology figure for comparing several
products against one reference (observed) on THREE statistics at once -
correlation (angle), normalized standard deviation (radial distance), and
implicitly RMSE (distance from the reference point) - in a single plot.
Replaces separate MAE/RMSE/R/KGE maps or bar charts with one compact figure.

Correlation and std ratio are NOT stored in the GeoParquet (only aggregate
metrics like KGE/PBIAS are), so this reads the raw clipped pickle directly
and computes them per station, then aggregates to one point per product
for the diagram (with station-level spread shown as faint background points).
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.projections as mprojections
from matplotlib.projections.polar import PolarAxes
import mpl_toolkits.axisartist.floating_axes as FA
import mpl_toolkits.axisartist.grid_finder as GF

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
clipped_data_path = f'{wkDir}/datasets/data_locked_loaded_clipped_fixed.pkl'
min_range = '2005-01-01'
max_range = '2018-12-31'
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

products = ['rawGPM', 'gwrGPM', 'expGPM', 'rain4pe', 'PISCO']
PRODUCT_COLORS = {
    'rawGPM': '#1f77b4',
    'gwrGPM': '#2ca02c',
    'expGPM': '#d62728',
    'rain4pe': '#9467bd',
    'PISCO': '#ff7f0e',
}

with open(clipped_data_path, 'rb') as f:
    final_data = pickle.load(f)

# =============================================================================
# 1. Per-station correlation and std ratio, for every product
# =============================================================================
def prep_series(df, col, start, end):
    df = df.copy()
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    return df.loc[start:end, col]

rows = []
for station_name, station in final_data.items():
    obs = prep_series(station.data, 'Precipitation', min_range, max_range)
    for prod in products:
        sim = prep_series(getattr(station, prod), 'precipitationCal', min_range, max_range)
        mask = obs.notna() & sim.notna()
        o, s = obs[mask], sim[mask]
        if len(o) < 30 or o.std() == 0:
            continue
        r = s.corr(o)
        std_ratio = s.std() / o.std()
        rows.append({'station': station_name, 'product': prod, 'r': r, 'std_ratio': std_ratio})

stats_df = pd.DataFrame(rows)

# Aggregate to one point per product (mean across stations) for the main markers
product_summary = stats_df.groupby('product').agg(r=('r', 'mean'), std_ratio=('std_ratio', 'mean'))
product_summary = product_summary.reindex(products)  # keep consistent order/colors
print("--- Network-mean correlation and std ratio per product ---")
print(product_summary.to_string())

# =============================================================================
# 2. Taylor diagram setup (quarter-circle polar plot)
# =============================================================================
class TaylorDiagram(object):
    """Minimal Taylor diagram on a matplotlib figure.
    Angular axis = correlation (0 to 1, mapped to 0 to 90 degrees).
    Radial axis = normalized standard deviation.
    """
    def __init__(self, fig, ref_std=1.0, srange=(0, 1.6), rect=111):
        self.ref_std = ref_std
        tr = PolarAxes.PolarTransform()

        # Correlation ticks (nonlinear angular grid)
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        tlocs = np.arccos(rlocs)
        gl1 = GF.FixedLocator(tlocs)
        tf1 = GF.DictFormatter(dict(zip(tlocs, [f"{x:.2f}" for x in rlocs])))

        smin, smax = srange[0] * ref_std, srange[1] * ref_std

        ghelper = FA.GridHelperCurveLinear(
            tr, extremes=(0, np.pi / 2, smin, smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)
        self.ax = ax
        self.ax.axis["top"].set_axis_direction("bottom")
        self.ax.axis["top"].toggle(ticklabels=True, label=True)
        self.ax.axis["top"].major_ticklabels.set_axis_direction("top")
        self.ax.axis["top"].label.set_axis_direction("top")
        self.ax.axis["top"].label.set_text("Correlation")

        self.ax.axis["left"].set_axis_direction("bottom")
        self.ax.axis["left"].label.set_text("Normalized standard deviation")

        self.ax.axis["right"].set_axis_direction("top")
        self.ax.axis["right"].toggle(ticklabels=True)
        self.ax.axis["right"].major_ticklabels.set_axis_direction("left")

        self.ax.axis["bottom"].set_visible(False)

        self._ax = self.ax
        self.ax = self.ax.get_aux_axes(tr)

        # Reference point (observed) and RMSE contours
        t = np.linspace(0, np.pi / 2)
        r = np.zeros_like(t) + self.ref_std
        self.ax.plot(t, r, 'k--', linewidth=1, label='_nolegend_')
        self.ax.plot([0], [self.ref_std], 'k*', markersize=14, label='Observed')

    def add_sample(self, std_ratio, r, *args, **kwargs):
        theta = np.arccos(np.clip(r, -1, 1))
        return self.ax.plot(theta, std_ratio, *args, **kwargs)

    def add_rmse_contours(self, levels=5, **kwargs):
        rs, ts = np.meshgrid(
            np.linspace(0, self._ax.get_ylim()[1] if hasattr(self._ax, 'get_ylim') else 1.6, 100),
            np.linspace(0, np.pi / 2, 100))
        rms = np.sqrt(self.ref_std ** 2 + rs ** 2 - 2 * self.ref_std * rs * np.cos(ts))
        contours = self.ax.contour(ts, rs, rms, levels, colors='gray', linestyles='dotted', linewidths=0.8)
        self.ax.clabel(contours, inline=1, fontsize=8, fmt='%.1f')


fig = plt.figure(figsize=(9, 9))
taylor = TaylorDiagram(fig, ref_std=1.0, srange=(0, 1.6))
taylor.add_rmse_contours(levels=5)

# Faint station-level points per product (spread)
for prod in products:
    sub = stats_df[stats_df['product'] == prod]
    theta = np.arccos(np.clip(sub['r'], -1, 1))
    taylor.ax.scatter(theta, sub['std_ratio'], color=PRODUCT_COLORS[prod], alpha=0.15, s=25, zorder=2)

# Bold network-mean marker per product
for prod in products:
    row = product_summary.loc[prod]
    taylor.add_sample(row['std_ratio'], row['r'], marker='o', markersize=12,
                       color=PRODUCT_COLORS[prod], markeredgecolor='black',
                       markeredgewidth=1, label=prod, linestyle='none', zorder=5)

fig.legend(loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=11)
fig.suptitle('Taylor diagram: precipitation products vs. observed\n'
             '(faint dots = individual stations, bold markers = network mean)',
             fontsize=13, y=1.02)

out_path = os.path.join(out_dir, 'taylor_diagram.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved {out_path}")

# Save the underlying per-station stats too, for the supplementary material
stats_out_path = os.path.join(wkDir, 'taylor_diagram_stats.csv')
stats_df.to_csv(stats_out_path, index=False)
print(f"Per-station r / std_ratio values saved to {stats_out_path}")