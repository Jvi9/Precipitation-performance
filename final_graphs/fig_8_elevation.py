"""
File: fig_8_elevation.py
Author: Jose P. Teran
Github: jopator
Date: 2026-10-02
Description: Daily performance statistics vs station elevation
             Figure 8 of manuscript: (a) PBIAS, (b) RMSE, (c) KGE, (d) r vs elevation
             Points = individual stations, lines = LOESS trend per product,
             boxed values = Spearman rho (* = p < 0.05).
             Annex figure 2: station elevation distribution (histogram + KDE).

Modified by: Jhon (added diagnostic printing block below `metrics_df` - no
             changes to the plotting logic itself) - this is NOT testing the
             old paragraph's specific band-based claims (2300m threshold,
             3100-3700m range, etc.) - those are being dropped entirely, not
             rebutted. This prints ONLY the rho-based evidence the new
             section is actually built on: what rho/p-value mean, the full
             per-product-per-metric table, and a synthesis of which
             products/metrics show a real (significant) elevation
             dependence versus none at all.
"""


import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from configurations import DISPLAY_NAMES, PLOT_ORDER, PRODUCT_COLORS
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, spearmanr

repoDir = Path(__file__).resolve().parents[1]

# Dirs
metrics_parquetFN   = repoDir / 'outputs/station_metrics_1mm_clipped.parquet'        # parquet with metrics
plotDir             = repoDir / 'graphs'
os.makedirs(plotDir, exist_ok=True)

# --------------
# Plot settings
# --------------

plt.rcParams.update({
    'xtick.labelsize':  10,     # axis tick labels
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'axes.labelsize':   12,     # axis labels
    'axes.titlesize':   14,     # subplot titles
    'figure.titlesize': 14,     # suptitle
    'savefig.dpi':      300,
})

ELEV_BIN = 250      # elevation histogram bin width (m), annex figure 2
RHO_STRIP = 0.12    # small x-axis extension (fraction of the range) giving the Spearman box some room
# Candidate corners for the Spearman box (axes fraction x, y, ha, va), in order of preference
RHO_CORNERS = [(0.98, 0.97, 'right', 'top'), (0.98, 0.03, 'right', 'bottom'),
               (0.02, 0.97, 'left', 'top'), (0.02, 0.03, 'left', 'bottom')]

# (column metric, axis label, reference line)
panel_specs = [
    ('pbias', 'PBIAS (%)',            0.0),     # unbiased
    ('rmse',  'RMSE (mm day$^{-1}$)', None),
    ('kge',   'KGE',                  1.0),     # perfect score
    ('r',     'r',                    None),
]
panel_labels = ['(a)', '(b)', '(c)', '(d)']


# --------------
# Helpers
# --------------

def simple_lowess(x, y, frac=0.6, n_points=100):
    """Minimal LOWESS using only numpy (no statsmodels dependency). Tricube-
    weighted local linear fit evaluated at n_points across the data range."""
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    n = len(x_sorted)
    k = max(int(np.ceil(frac * n)), 2)

    x_eval = np.linspace(x_sorted.min(), x_sorted.max(), n_points)
    y_eval = np.empty(n_points)

    for i, x0 in enumerate(x_eval):
        dist = np.abs(x_sorted - x0)
        idx = np.argsort(dist)[:k]
        d_max = dist[idx].max()
        weights = np.ones(k) if d_max == 0 else np.clip((1 - (dist[idx] / d_max) ** 3) ** 3, 0, None)
        xi, yi, wi = x_sorted[idx], y_sorted[idx], weights
        W = np.diag(wi)
        X = np.vstack([np.ones(k), xi]).T
        try:
            beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ yi)
            y_eval[i] = beta[0] + beta[1] * x0
        except np.linalg.LinAlgError:
            y_eval[i] = np.average(yi, weights=wi)
    return x_eval, y_eval


def place_rho_box(ax, text, xy):
    """Spearman box in the axes corner covering the fewest data points / LOESS samples (`xy`, data coords)."""
    renderer = ax.figure.canvas.get_renderer()
    pts = ax.transData.transform(xy)
    best = None
    for x, y, ha, va in RHO_CORNERS:
        t = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, multialignment='right',
                    fontsize=plt.rcParams['legend.fontsize'],
                    bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'alpha': 0.8,
                          'edgecolor': 'lightgray', 'linewidth': 0.6})
        n_covered = t.get_window_extent(renderer).expanded(1.1, 1.15).count_contains(pts)
        if best is None or n_covered < best[0]:
            if best is not None:
                best[1].remove()
            best = (n_covered, t)
        else:
            t.remove()


# --------------
# Load data
# --------------

metrics_df = gpd.read_parquet(metrics_parquetFN)

# =============================================================================
# DIAGNOSTIC PRINTING - the rho-based evidence only. No elevation bands, no
# thresholds - the new section is built entirely around whether each
# product/metric shows a REAL (significant) monotonic relationship with
# elevation, and how strong it is where it exists.
# =============================================================================

print("="*95)
print("WHAT THESE NUMBERS MEAN")
print("="*95)
print("Spearman rho: -1 to +1, measures whether a metric consistently rises or falls")
print("with elevation (rank-based - doesn't assume a straight-line relationship).")
print("  |rho| < 0.3  -> weak")
print("  0.3 - 0.5    -> moderate")
print("  > 0.5        -> strong")
print("p-value: given n=70 stations, how likely is a correlation this large to appear")
print("by chance if there were truly NO relationship. p < 0.05 (marked *) = unlikely to")
print("be noise. A high rho with NO * is a pattern that should be treated with caution,")
print("not reported as if it were established.")
print("="*95)

results = []
for metric, ylabel, _ in panel_specs:
    for code in PLOT_ORDER:
        x = metrics_df['alt'].to_numpy()
        y = metrics_df[f'{metric}_{code}'].to_numpy()
        mask = ~np.isnan(x) & ~np.isnan(y)
        rho, pval = spearmanr(x[mask], y[mask])
        strength = 'weak' if abs(rho) < 0.3 else ('moderate' if abs(rho) < 0.5 else 'strong')
        sig = pval < 0.05
        results.append({'metric': metric.upper(), 'product': DISPLAY_NAMES[code],
                        'rho': rho, 'pval': pval, 'strength': strength, 'significant': sig})

results_df = pd.DataFrame(results)

print("\n--- FULL TABLE: rho and significance, every product, every metric ---")
for metric in results_df['metric'].unique():
    print(f"\n{metric} vs elevation:")
    sub = results_df[results_df['metric'] == metric]
    for _, row in sub.iterrows():
        direction = 'increases' if row['rho'] > 0 else 'decreases'
        sig_text = 'significant' if row['significant'] else 'NOT significant'
        print(f"  {row['product']:12s} rho={row['rho']:+.3f}  {row['strength']:8s}  "
              f"{direction} with elevation  - {sig_text}")

# =============================================================================
# SYNTHESIS: which metrics show a real (significant) elevation dependence
# across MOST/ALL products, versus which products show a real dependence
# across MOST/ALL metrics - this is the actual structure the new paragraph
# is built around (e.g. RMSE: universal strong decline; PISCO: no
# significant relationship anywhere; Rain4PE: significant in every metric)
# =============================================================================
print("\n" + "="*95)
print("SYNTHESIS 1: per METRIC, how many of the 5 products show a significant")
print("relationship with elevation (tells you if a metric is universally")
print("elevation-sensitive, or only for specific products)")
print("="*95)
for metric in results_df['metric'].unique():
    sub = results_df[results_df['metric'] == metric]
    n_sig = sub['significant'].sum()
    print(f"  {metric:6s}: {n_sig}/5 products significant  "
          f"({', '.join(sub[sub['significant']]['product'].tolist()) or 'none'})")

print("\n" + "="*95)
print("SYNTHESIS 2: per PRODUCT, how many of the 4 metrics show a significant")
print("relationship with elevation (tells you which products have a coherent,")
print("multi-metric elevation story, versus which show no real pattern at all)")
print("="*95)
for code in PLOT_ORDER:
    sub = results_df[results_df['product'] == DISPLAY_NAMES[code]]
    n_sig = sub['significant'].sum()
    sig_metrics = sub[sub['significant']]['metric'].tolist()
    print(f"  {DISPLAY_NAMES[code]:12s}: {n_sig}/4 metrics significant  "
          f"({', '.join(sig_metrics) or 'none'})")


# --------------
# Figure 8 -> 2x2 metrics vs elevation
# --------------

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
rho_boxes = []

for i, (ax, (metric, ylabel, ref_line)) in enumerate(zip(axs.ravel(), panel_specs)):
    rho_lines, xy = [], []
    for code in PLOT_ORDER:
        x = metrics_df['alt'].to_numpy()
        y = metrics_df[f'{metric}_{code}'].to_numpy()
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid, y_valid = x[mask], y[mask]

        ax.scatter(x_valid, y_valid, color=PRODUCT_COLORS[code], alpha=0.45, s=25,
                   edgecolor='none', label=DISPLAY_NAMES[code])
        x_lo, y_lo = simple_lowess(x_valid, y_valid, frac=0.6)
        ax.plot(x_lo, y_lo, color=PRODUCT_COLORS[code], linewidth=2)
        xy += [np.column_stack([x_valid, y_valid]), np.column_stack([x_lo, y_lo])]

        rho, pval = spearmanr(x_valid, y_valid)
        rho_lines.append(f"{DISPLAY_NAMES[code]}: ρ={rho:+.2f}{'*' if pval < 0.05 else ''}")

    if ref_line is not None:
        ax.axhline(ref_line, color='black', linestyle='--', linewidth=1, alpha=0.6)

    ax.set_xlabel('Elevation (m)' if i >= 2 else '')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + RHO_STRIP * (x1 - x0))
    ax.text(-0.1, 1.05, panel_labels[i], transform=ax.transAxes,
            fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')
    rho_boxes.append((ax, '\n'.join(rho_lines), np.vstack(xy)))

fig.tight_layout()

# Spearman boxes placed after the layout is final, so the overlap check uses the real panel sizes
for ax, text, xy in rho_boxes:
    place_rho_box(ax, text, xy)

# Shared dataset legend, horizontal below the panels
handles = [Line2D([], [], color=PRODUCT_COLORS[code], marker='o', linewidth=2, label=DISPLAY_NAMES[code])
           for code in PLOT_ORDER]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=len(handles), frameon=False)
fig.savefig(plotDir / 'fig8_elevation_continuous.png', bbox_inches='tight')
plt.close(fig)

# --------------
# Annex figure 2 -> Station elevation distribution: histogram + KDE
# --------------

alt = metrics_df['alt'].dropna()
bins = np.arange(np.floor(alt.min() / ELEV_BIN) * ELEV_BIN, alt.max() + ELEV_BIN, ELEV_BIN)
# KDE scaled from density to station counts per bin, so it shares the histogram's y-axis
x_kde = np.linspace(bins[0], bins[-1], 300)
y_kde = gaussian_kde(alt)(x_kde) * len(alt) * ELEV_BIN

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(alt, bins=bins, color='steelblue', edgecolor='black', alpha=0.75, label='Stations')
ax.plot(x_kde, y_kde, color='black', linewidth=2, label='KDE')
ax.set_xlabel('Elevation (m)')
ax.set_ylabel('Number of stations')
ax.grid(axis='y', alpha=0.3)
ax.legend(loc='upper left', framealpha=0.9)

fig.tight_layout()
fig.savefig(plotDir / 'fig_annex_2.png', bbox_inches='tight')
plt.close(fig)