"""
File: fig_9_elevation.py
Author: Jose P. Teran
Github: jopator
Date: 2026-10-02
Description: Categorical (detection) statistics vs station elevation
             Figure 9 of manuscript: (a) CSI, (b) FBI, (c) FAR, (d) POD vs elevation
             Points = individual stations, lines = LOESS trend per product,
             boxed values = Spearman rho (* = p < 0.05).
             CSI is derived from POD and FAR.
"""


import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from configurations import DISPLAY_NAMES, PLOT_ORDER, PRODUCT_COLORS
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

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

# (column metric, axis label, reference line)
panel_specs = [
    ('csi', 'CSI', None),
    ('fbi', 'FBI', 1.0),      # unbiased
    ('far', 'FAR', None),
    ('pod', 'POD', None),
]
panel_labels = ['(a)', '(b)', '(c)', '(d)']
RHO_STRIP = 0.12    # small x-axis extension (fraction of the range) giving the Spearman box some room
# Candidate corners for the Spearman box (axes fraction x, y, ha, va), in order of preference
RHO_CORNERS = [(0.98, 0.97, 'right', 'top'), (0.98, 0.03, 'right', 'bottom'),
               (0.02, 0.97, 'left', 'top'), (0.02, 0.03, 'left', 'bottom')]


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

# CSI is not in the parquet, derived from POD and FAR: CSI = 1 / (1/POD + 1/(1-FAR) - 1)
for code in PLOT_ORDER:
    pod, far = metrics_df[f'pod_{code}'], metrics_df[f'far_{code}']
    with np.errstate(divide='ignore', invalid='ignore'):
        metrics_df[f'csi_{code}'] = 1 / (1 / pod + 1 / (1 - far) - 1)

# --------------
# Figure 9 -> 2x2 metrics vs elevation
# --------------

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
rho_boxes = []

for i, (ax, (metric, ylabel, ref_line)) in enumerate(zip(axs.ravel(), panel_specs)):
    rho_lines, xy = [], []
    for code in PLOT_ORDER:
        x = metrics_df['alt'].to_numpy()
        y = metrics_df[f'{metric}_{code}'].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
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
fig.savefig(plotDir / 'fig9_elevation_categorical.png', bbox_inches='tight')
plt.close(fig)
