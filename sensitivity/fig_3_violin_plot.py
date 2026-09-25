# -*- coding: utf-8 -*-
"""
Figure 3: MIRRORED (split) violins, LINEAR-space KDE. LEFT half of each
product's panel = Observed, RIGHT half = that product, own color.

Y-axis uses symlog scale: linear from 0 to SYMLOG_LINTHRESH (so dry days
and light rain render correctly - plain log(0) is undefined, which is why
this isn't just 'log'), then logarithmic beyond that, compressing the long
right tail so the full range is visible without an artificial crop.

All days are included (dry days too) - needed for the "small amounts of
rainfall on no-rainfall days" story your text makes; that pattern is only
visible if Observed's true zero-heavy distribution stays in the data.

Median = dotted line, Q1/Q3 = dashed lines, drawn from each half's own
actual pooled data, with each line's length matched to the violin's own
width at that exact value (not a fixed bar width).

BEFORE TRUSTING ANY NUMBER IN YOUR TEXT: this script prints the pooled
median/Q1/Q3 for every dataset before plotting. Check those printed values
directly against your paragraph's specific claims - the figure is only as
trustworthy as those numbers, and they're computed here, not assumed.

Uses the raw daily pickle (every station, every day), not the GeoParquet.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde

from final_graphs.configurations import PLOT_ORDER, DISPLAY_NAMES, PRODUCT_COLORS, OBSERVED_COLOR, OBSERVED_LABEL

wkDir = r'C:\Users\jvila\Desktop\Andean_project'
clipped_fixed_pickle = f'{wkDir}/datasets/data_locked_loaded_clipped_fixed.pkl'
min_range = '2005-01-01'
max_range = '2018-12-31'
out_dir = os.path.join(wkDir, 'graphs')
os.makedirs(out_dir, exist_ok=True)

SYMLOG_LINTHRESH = 1.0  # mm - the linear/log boundary; matches your established wet-day threshold
HALF_WIDTH = 0.42

with open(clipped_fixed_pickle, 'rb') as f:
    final_data = pickle.load(f)

def prep_series(df, col, start, end):
    df = df.copy()
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    return df.loc[start:end, col]

# =============================================================================
# 1. Pool every station's daily values into one array per dataset (all days)
# =============================================================================
pooled = {'Observed': []}
for code in PLOT_ORDER:
    pooled[code] = []

for station_name, station in final_data.items():
    obs = prep_series(station.data, 'Precipitation', min_range, max_range).dropna()
    pooled['Observed'].append(obs.values)
    for code in PLOT_ORDER:
        sim = prep_series(getattr(station, code), 'precipitationCal', min_range, max_range).dropna()
        pooled[code].append(sim.values)

for key in pooled:
    pooled[key] = np.concatenate(pooled[key]).astype(np.float64)

print("Pooled sample sizes (all days, including dry)")
print("CHECK THESE NUMBERS AGAINST YOUR PARAGRAPH'S SPECIFIC CLAIMS BEFORE TRUSTING THE TEXT:")
for key, vals in pooled.items():
    label = OBSERVED_LABEL if key == 'Observed' else DISPLAY_NAMES[key]
    print(f"  {label:12s} n={len(vals):,}  mean={vals.mean():.2f}mm  median={np.median(vals):.2f}mm  "
          f"Q1={np.percentile(vals,25):.2f}  Q3={np.percentile(vals,75):.2f}")

# =============================================================================
# 1b. CONDITIONAL check on days OBSERVED recorded exactly zero: does each
#     product report "light rain" that isn't really there? This is the
#     DIRECT test of the "estimates small amounts of rainfall on no-rainfall
#     days" claim - pooled quartiles alone can't distinguish this from
#     ordinary spread, since they mix wet and dry days together.
# =============================================================================
print("\n" + "="*90)
print("CONDITIONAL CHECK: what does each product report on days OBSERVED = 0mm?")
print("(direct test of the 'light rain on no-rainfall days' claim)")
print("="*90)

station_pairs = {code: [] for code in PLOT_ORDER}
obs_zero_mask_pairs = []
for station_name, station in final_data.items():
    obs = prep_series(station.data, 'Precipitation', min_range, max_range)
    for code in PLOT_ORDER:
        sim = prep_series(getattr(station, code), 'precipitationCal', min_range, max_range)
        common_idx = obs.index.intersection(sim.index)
        obs_aligned = obs.loc[common_idx]
        sim_aligned = sim.loc[common_idx]
        mask = obs_aligned.notna() & sim_aligned.notna() & (obs_aligned == 0)
        station_pairs[code].append(sim_aligned[mask].values.astype(np.float64))

for code in PLOT_ORDER:
    vals_on_obs_zero = np.concatenate(station_pairs[code])
    frac_reports_wet = (vals_on_obs_zero >= 1.0).mean()
    print(f"  {DISPLAY_NAMES[code]:12s} n(obs=0 days)={len(vals_on_obs_zero):,}  "
          f"mean reported={vals_on_obs_zero.mean():.3f}mm  median reported={np.median(vals_on_obs_zero):.3f}mm  "
          f"% of these days product reports \u22651mm: {frac_reports_wet:.1%}")

# =============================================================================
# 1c. Distributional distance to Observed (Wasserstein) - a single honest
#     number summarizing how different each FULL distribution is from
#     Observed, beyond 3 percentile points. Also tests whether GPM-EXP
#     genuinely differs from IMERGF/GWR (their medians/quartiles look
#     nearly identical - this checks the actual shape).
# =============================================================================
from scipy.stats import wasserstein_distance, ks_2samp

print("\n" + "="*90)
print("DISTRIBUTIONAL DISTANCE TO OBSERVED (Wasserstein distance - lower = more similar)")
print("="*90)
for code in PLOT_ORDER:
    wd = wasserstein_distance(pooled['Observed'], pooled[code])
    ks_stat, ks_p = ks_2samp(pooled['Observed'], pooled[code])
    print(f"  {DISPLAY_NAMES[code]:12s} Wasserstein={wd:.3f}mm   KS statistic={ks_stat:.3f} (p={ks_p:.2e})")

print("\n--- Is GPM-EXP's distribution actually different from GPM-IMERGF / GPM-GWR? ---")
for pair in [('rawGPM', 'expGPM'), ('gwrGPM', 'expGPM'), ('rawGPM', 'gwrGPM')]:
    wd = wasserstein_distance(pooled[pair[0]], pooled[pair[1]])
    ks_stat, ks_p = ks_2samp(pooled[pair[0]], pooled[pair[1]])
    print(f"  {DISPLAY_NAMES[pair[0]]} vs {DISPLAY_NAMES[pair[1]]}: "
          f"Wasserstein={wd:.3f}mm  KS statistic={ks_stat:.3f} (p={ks_p:.2e})")

# =============================================================================
# 2. Linear-space KDE, evaluated over each dataset's own actual range
# =============================================================================
def kde_linear(values, y_grid):
    kde = gaussian_kde(values)  # default (Scott's rule) bandwidth, same as seaborn's default
    density = kde(y_grid)
    return density / density.max()  # each half normalized to its own max width

# =============================================================================
# 3. Plot: one split violin per product, Observed on the left half of each
# =============================================================================
fig, ax = plt.subplots(figsize=(24, 12))

for i, code in enumerate(PLOT_ORDER):
    obs_vals = pooled['Observed']
    sim_vals = pooled[code]

    y_grid_obs = np.linspace(0, obs_vals.max(), 400)
    y_grid_sim = np.linspace(0, sim_vals.max(), 400)
    obs_density = kde_linear(obs_vals, y_grid_obs)
    sim_density = kde_linear(sim_vals, y_grid_sim)

    ax.fill_betweenx(y_grid_obs, i - obs_density * HALF_WIDTH, i, color=OBSERVED_COLOR, alpha=0.7, zorder=3)
    ax.fill_betweenx(y_grid_sim, i, i + sim_density * HALF_WIDTH, color=PRODUCT_COLORS[code], alpha=0.7, zorder=3)

    obs_med, obs_q1, obs_q3 = np.median(obs_vals), *np.percentile(obs_vals, [25, 75])
    for val, style, lw in [(obs_med, ':', 2), (obs_q1, '--', 1.2), (obs_q3, '--', 1.2)]:
        w = np.interp(val, y_grid_obs, obs_density) * HALF_WIDTH  # violin's actual width at this exact value
        ax.plot([i - w, i], [val, val], color='black', linestyle=style, linewidth=lw, zorder=5)

    sim_med, sim_q1, sim_q3 = np.median(sim_vals), *np.percentile(sim_vals, [25, 75])
    for val, style, lw in [(sim_med, ':', 2), (sim_q1, '--', 1.2), (sim_q3, '--', 1.2)]:
        w = np.interp(val, y_grid_sim, sim_density) * HALF_WIDTH
        ax.plot([i, i + w], [val, val], color='black', linestyle=style, linewidth=lw, zorder=5)

    ax.axvline(i, color='white', linewidth=1.5, zorder=4)

ax.set_yscale('symlog', linthresh=SYMLOG_LINTHRESH)

# --- Cap the axis at the true maximum found in the data, from zero ---
true_max = max(vals.max() for vals in pooled.values())
ax.set_ylim(0, true_max)

# --- Denser ticks: linear steps within the linear region (0 to linthresh),
#     then 1/2/5-per-decade steps in the log region beyond it - the standard
#     way to get more than just bare powers of 10 on a symlog axis ---
def build_symlog_ticks(linthresh, y_max, n_linear_steps=5):
    ticks = list(np.linspace(0, linthresh, n_linear_steps + 1))
    decade = linthresh
    while decade < y_max:
        for m in (1, 2, 5):
            val = decade * m
            if linthresh < val <= y_max:
                ticks.append(val)
        decade *= 10
    if y_max not in ticks:
        ticks.append(y_max)
    return sorted(set(round(t, 3) for t in ticks))

y_ticks = build_symlog_ticks(SYMLOG_LINTHRESH, true_max)
ax.set_yticks(y_ticks)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x:g}'))
ax.yaxis.set_minor_locator(mticker.NullLocator())
ax.set_xticks(range(len(PLOT_ORDER)))
ax.set_xticklabels([DISPLAY_NAMES[c] for c in PLOT_ORDER], fontsize=17)
ax.set_xlim(-0.6, len(PLOT_ORDER) - 0.4)
ax.set_xlabel('')
ax.set_ylabel('Precipitation (mm/day)', fontsize=21)
ax.set_title('Distribution of daily precipitation values: each dataset (right half)\n'
            'vs. local observations (left half)', fontsize=22)
ax.tick_params(axis='y', labelsize=17)
ax.grid(axis='y', alpha=0.3)

obs_patch = mpatches.Patch(color=OBSERVED_COLOR, alpha=0.7, label='Observed (left half)')
sim_patch = mpatches.Patch(color='gray', alpha=0.7, label='Dataset (right half, own color)')
median_line = plt.Line2D([0], [0], color='black', linestyle=':', linewidth=2, label='Median')
quartile_line = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.2, label='1st / 3rd quartile')
ax.legend(handles=[obs_patch, sim_patch, median_line, quartile_line],
         loc='upper right', fontsize=13, framealpha=0.9)

plt.tight_layout()
out_path = os.path.join(out_dir, 'figure3_daily_distribution_symlog.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved {out_path}")