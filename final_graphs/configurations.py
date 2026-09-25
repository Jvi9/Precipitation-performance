# -*- coding: utf-8 -*-
"""
configurations.py

Single source of truth for naming and color conventions across every figure
script in this project. Import from here instead of hardcoding product
names/colors per script, so a naming change only has to happen once.

Usage:
    from configurations import PRODUCT_CODES, DISPLAY_NAMES, PRODUCT_COLORS, get_display_name
"""

# =============================================================================
# Product identity
# =============================================================================
# Internal codes - these are the pickle attribute names AND the GeoParquet
# column suffixes (e.g. 'r_rawGPM'). Do not rename these without also
# updating station_class.py / analysis_class.py, since they're tied to the
# actual data attributes.
PRODUCT_CODES = ['rawGPM', 'gwrGPM', 'expGPM', 'rain4pe', 'PISCO']

# Standardized display names for figure legends, axis labels, captions
DISPLAY_NAMES = {
    'rawGPM':  'GPM-IMERGF',
    'gwrGPM':  'GPM-GWR',
    'expGPM':  'GPM-EXP',
    'rain4pe': 'Rain4PE',
    'PISCO':   'PISCO',
}

# Reverse lookup: display name -> internal code
CODE_FROM_DISPLAY = {v: k for k, v in DISPLAY_NAMES.items()}

# Preferred plotting order (legends, bar charts, table columns) - edit this
# single list to reorder every figure consistently
PLOT_ORDER = ['rawGPM', 'gwrGPM', 'expGPM', 'rain4pe', 'PISCO']

# =============================================================================
# Colors - consistent across every figure in the paper
# =============================================================================
# Okabe-Ito palette: the standard colorblind-safe qualitative palette in
# scientific visualization, chosen specifically to remain distinguishable
# under all common forms of color vision deficiency and to degrade more
# gracefully than tab10/Set1 when printed in grayscale. Recommended over a
# default matplotlib palette whenever you have this many categories.
PRODUCT_COLORS = {
    'rawGPM':  '#E69F00',  # orange           -> GPM-IMERGF
    'gwrGPM':  '#56B4E9',  # sky blue         -> GPM-GWR
    'expGPM':  '#D55E00',  # vermillion       -> GPM-EXP
    'rain4pe': '#009E73',  # bluish green     -> Rain4PE
    'PISCO':   '#CC79A7',  # reddish purple   -> PISCO
}

# Marker shapes - a SECOND, color-independent channel for product identity.
# Use both together (color AND shape) in any figure with all 5 products, so
# identity never depends on color alone (grayscale printing, colorblindness,
# or just five colors being a lot to hold in your head at once).
PRODUCT_MARKERS = {
    'rawGPM':  'o',  # circle    -> GPM-IMERGF
    'gwrGPM':  's',  # square    -> GPM-GWR
    'expGPM':  '^',  # triangle  -> GPM-EXP
    'rain4pe': 'D',  # diamond   -> Rain4PE
    'PISCO':   'P',  # plus      -> PISCO
}

# Same colors, keyed by DISPLAY NAME instead - use whichever key style a
# given script already has on hand
PRODUCT_COLORS_BY_DISPLAY = {DISPLAY_NAMES[k]: v for k, v in PRODUCT_COLORS.items()}

OBSERVED_COLOR = '#000000'   # black - always used for the observed/reference series
OBSERVED_LABEL = 'Observed'

def get_display_name(code):
    """Internal code -> standardized display name. Passes through unknown values."""
    return DISPLAY_NAMES.get(code, code)

def get_color(code_or_display):
    """Look up a product's color by either internal code or display name."""
    if code_or_display in PRODUCT_COLORS:
        return PRODUCT_COLORS[code_or_display]
    if code_or_display in PRODUCT_COLORS_BY_DISPLAY:
        return PRODUCT_COLORS_BY_DISPLAY[code_or_display]
    raise KeyError(f"Unknown product identifier: {code_or_display}")

def get_marker(code_or_display):
    """Look up a product's marker shape by either internal code or display name."""
    if code_or_display in PRODUCT_MARKERS:
        return PRODUCT_MARKERS[code_or_display]
    if code_or_display in CODE_FROM_DISPLAY:
        return PRODUCT_MARKERS[CODE_FROM_DISPLAY[code_or_display]]
    raise KeyError(f"Unknown product identifier: {code_or_display}")

def display_columns(metric, codes=None):
    """Given a metric name, return {display_name: geoparquet_column} for all
    products (or a subset via `codes`). Handy for building legends/labels
    straight from a GeoParquet read without repeating the mapping per script."""
    codes = codes or PLOT_ORDER
    return {DISPLAY_NAMES[c]: f"{metric}_{c}" for c in codes}

# =============================================================================
# GeoParquet column reference (station_metrics_1mm_clipped.parquet)
# =============================================================================
# One row per station. Metric columns follow: f"{metric}_{PRODUCT_CODE}"
# using the INTERNAL codes above, never the display names.
# Non-metric columns: station, lat, lon, alt, geometry

CONTINUOUS_METRICS = ['mae', 'pbias', 'rmse', 'r', 'kge']       # threshold-independent
DETECTION_METRICS = ['fbi', 'far', 'pod', 'acc']                # computed at the unified threshold (currently 1.0mm)
EXTREME_INDICES = ['cdd', 'cwd', 'r10', 'r20', 'r95p', 'r99p']  # ETCCDI-style, mean across years

# Extreme indices also exist for the observed record as f"{index}_Observed"
# (e.g. 'cdd_Observed'). There is no _Observed version of continuous or
# detection metrics, since Observed is the reference, not compared to itself.

METRIC_DISPLAY_NAMES = {
    'mae': 'MAE', 'pbias': 'PBIAS', 'rmse': 'RMSE', 'r': 'r', 'kge': 'KGE',
    'fbi': 'FBI', 'far': 'FAR', 'pod': 'POD', 'acc': 'ACC',
    'cdd': 'CDD', 'cwd': 'CWD', 'r10': 'R10mm', 'r20': 'R20mm',
    'r95p': 'R95p', 'r99p': 'R99p', 'csi': 'CSI', 'sr': 'Success Ratio',
}

def metric_column(metric, product_code):
    """Build the GeoParquet column name for a given metric + product code."""
    return f"{metric}_{product_code}"

# Example column names for quick reference:
#   'r_rawGPM', 'kge_PISCO', 'pod_rain4pe', 'cdd_gwrGPM', 'cdd_Observed'