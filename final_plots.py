import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

# -----------------------------
# LOAD DATA
# -----------------------------
gdf = gdf.to_crs(epsg=3857)

boundary_path = r'C:\Users\jvila\Desktop\Andean_project\gis\study_area_shp\study_area.shp'
boundary = gpd.read_file(boundary_path).to_crs(epsg=3857)

# -----------------------------
# DEFINE MODELS
# -----------------------------
pbias_cols = [
    "pbias_rawGPM",
    "pbias_gwrGPM",
    "pbias_PISCO",
    "pbias_rain4pe",
    "pbias_expGPM"
]

# -----------------------------
# ENSEMBLE MEAN
# -----------------------------
gdf["pbias_mean"] = gdf[pbias_cols].mean(axis=1)

# -----------------------------
# DIFFERENCE MAPS
# -----------------------------
for col in pbias_cols:
    gdf[f"diff_{col}"] = gdf[col] - gdf["pbias_mean"]

# -----------------------------
# COLOR RANGE
# -----------------------------
vmin = gdf[[f"diff_{m}" for m in pbias_cols]].min().min()
vmax = gdf[[f"diff_{m}" for m in pbias_cols]].max().max()

# -----------------------------
# FIGURE 2×3
# -----------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=600)
axes = axes.flatten()

models_to_plot = pbias_cols
titles = [m.replace("pbias_", "") for m in models_to_plot]

for ax, model, title in zip(axes[:5], models_to_plot, titles):

    boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8)

    gdf.plot(
        ax=ax,
        column=f"diff_{model}",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        markersize=90,
        edgecolor="white",
        linewidth=0.8,
        legend=False
    )

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()

# Empty last panel
axes[5].set_axis_off()

# -----------------------------
# COLORBAR OUTSIDE GRID
# -----------------------------
cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]

sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("PBIAS Difference from Ensemble Mean", fontsize=12)

plt.subplots_adjust(right=0.90)  # leave space for colorbar
plt.show()




import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

# -----------------------------
# LOAD DATA
# -----------------------------
gdf = gdf.to_crs(epsg=3857)

boundary_path = r'C:\Users\jvila\Desktop\Andean_project\gis\study_area_shp\study_area.shp'
boundary = gpd.read_file(boundary_path).to_crs(epsg=3857)

# -----------------------------
# DEFINE R COLUMNS
# -----------------------------
r_cols = [
    "r_rawGPM",
    "r_gwrGPM",
    "r_PISCO",
    "r_rain4pe",
    "r_expGPM"
]

# -----------------------------
# ENSEMBLE MEAN OF R
# -----------------------------
gdf["r_mean"] = gdf[r_cols].mean(axis=1)

# -----------------------------
# DIFFERENCE MAPS FOR R
# -----------------------------
for col in r_cols:
    gdf[f"diff_{col}"] = gdf[col] - gdf["r_mean"]

# -----------------------------
# COLOR RANGE (same for all)
# -----------------------------
vmin = gdf[[f"diff_{m}" for m in r_cols]].min().min()
vmax = gdf[[f"diff_{m}" for m in r_cols]].max().max()

# -----------------------------
# FIGURE 2×3
# -----------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=600)
axes = axes.flatten()

titles = [m.replace("r_", "") for m in r_cols]

for ax, model, title in zip(axes[:5], r_cols, titles):

    boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8)

    gdf.plot(
        ax=ax,
        column=f"diff_{model}",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        markersize=90,
        edgecolor="white",
        linewidth=0.8,
        legend=False
    )

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()

# Empty last panel
axes[5].set_axis_off()

# -----------------------------
# COLORBAR OUTSIDE GRID
# -----------------------------
# -----------------------------
# COLORBAR OUTSIDE GRID
# -----------------------------
cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])

norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
sm.set_array([])  # <-- ESTA ES LA CLAVE

cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("R Difference from Ensemble Mean", fontsize=12)

# =============================================================================
# map best
# =============================================================================
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

# ============================================================
# 1. Definir columnas R
# ============================================================
r_cols = [
    "r_rawGPM",
    "r_gwrGPM",
    "r_PISCO",
    "r_rain4pe",
    "r_expGPM"
]

# ============================================================
# 2. Matriz R (n_points × n_models)
# ============================================================
r_matrix = gdf[r_cols].values

# ============================================================
# 3. Best model (máximo R)
# ============================================================
best_idx = r_matrix.argmax(axis=1)
gdf["best_model_R"] = [r_cols[i] for i in best_idx]

# ============================================================
# 4. Second best model (segundo mayor R)
# ============================================================
sorted_idx = np.argsort(r_matrix, axis=1)
second_idx = sorted_idx[:, -2]   # segundo mayor
gdf["second_model_R"] = [r_cols[i] for i in second_idx]

# ============================================================
# 5. Colores Glasbey (máxima separación)
# ============================================================
model_colors_R = {
    "r_rawGPM":   "#FF0000",   # rojo
    "r_gwrGPM":   "#00A2FF",   # azul brillante
    "r_PISCO":    "#00CC44",   # verde
    "r_rain4pe":  "#FFAA00",   # naranja
    "r_expGPM":   "#AA00FF"    # púrpura
}

# ============================================================
# 6. Asignar colores
# ============================================================
gdf["color_best"] = gdf["best_model_R"].map(model_colors_R)
gdf["color_second"] = gdf["second_model_R"].map(model_colors_R)

# ============================================================
# 7. FIGURA 1×2 — PAPER READY
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=600)

titles = ["(a) Best Model (R skill)", "(b) Second Best Model (R skill)"]
cols = ["color_best", "color_second"]
labels = ["best_model_R", "second_model_R"]

for ax, title, col, lab in zip(axes, titles, cols, labels):

    # Boundary
    boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8)

    # Points
    gdf.plot(
        ax=ax,
        color=gdf[col],
        markersize=140,
        edgecolor="white",
        linewidth=0.8
    )

    # Legend manual
    for model, color in model_colors_R.items():
        ax.scatter([], [], color=color, label=model.replace("r_", ""), s=140)

    ax.legend(
        title="Model",
        frameon=True,
        fontsize=12,
        title_fontsize=13,
        loc="lower left"
    )

    ax.set_title(title, fontsize=18, pad=15)
    ax.set_axis_off()

plt.tight_layout()
plt.show()


# =============================================================================
# 
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Definir columnas FBI
# ============================================================
fbi_cols = [
    "fbi_rawGPM",
    "fbi_gwrGPM",
    "fbi_PISCO",
    "fbi_rain4pe",
    "fbi_expGPM"
]

# ============================================================
# 2. Matriz FBI (n_points × n_models)
# ============================================================
fbi_matrix = gdf[fbi_cols].values

# ============================================================
# 3. Best model (máximo FBI)
# ============================================================
best_idx = fbi_matrix.argmax(axis=1)
gdf["best_model_FBI"] = [fbi_cols[i] for i in best_idx]

# ============================================================
# 4. Second best model (segundo mayor FBI)
# ============================================================
sorted_idx = np.argsort(fbi_matrix, axis=1)
second_idx = sorted_idx[:, -2]   # segundo mayor
gdf["second_model_FBI"] = [fbi_cols[i] for i in second_idx]

# ============================================================
# 5. Colores Glasbey (máxima separación)
# ============================================================
model_colors_FBI = {
    "fbi_rawGPM":   "#FF0000",   # rojo
    "fbi_gwrGPM":   "#00A2FF",   # azul brillante
    "fbi_PISCO":    "#00CC44",   # verde
    "fbi_rain4pe":  "#FFAA00",   # naranja
    "fbi_expGPM":   "#AA00FF"    # púrpura
}

# ============================================================
# 6. Asignar colores
# ============================================================
gdf["color_best_FBI"] = gdf["best_model_FBI"].map(model_colors_FBI)
gdf["color_second_FBI"] = gdf["second_model_FBI"].map(model_colors_FBI)

# ============================================================
# 7. FIGURA 1×2 — PAPER READY
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=600)

titles = ["(a) Best Model (FBI skill)", "(b) Second Best Model (FBI skill)"]
cols = ["color_best_FBI", "color_second_FBI"]
labels = ["best_model_FBI", "second_model_FBI"]

for ax, title, col, lab in zip(axes, titles, cols, labels):

    # Boundary
    boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8)

    # Points
    gdf.plot(
        ax=ax,
        color=gdf[col],
        markersize=140,
        edgecolor="white",
        linewidth=0.8
    )

    # Legend manual
    for model, color in model_colors_FBI.items():
        ax.scatter([], [], color=color, label=model.replace("fbi_", ""), s=140)

    ax.legend(
        title="Model",
        frameon=True,
        fontsize=12,
        title_fontsize=13,
        loc="lower left"
    )

    ax.set_title(title, fontsize=18, pad=15)
    ax.set_axis_off()

plt.tight_layout()
plt.show()


# =============================================================================
# 
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Definir columnas ACC
# ============================================================
acc_cols = [
    "acc_rawGPM",
    "acc_gwrGPM",
    "acc_PISCO",
    "acc_rain4pe",
    "acc_expGPM"
]

# ============================================================
# 2. Matriz ACC (n_points × n_models)
# ============================================================
acc_matrix = gdf[acc_cols].values

# ============================================================
# 3. Best model (máximo ACC)
# ============================================================
best_idx = acc_matrix.argmax(axis=1)
gdf["best_model_ACC"] = [acc_cols[i] for i in best_idx]

# ============================================================
# 4. Second best model (segundo mayor ACC)
# ============================================================
sorted_idx = np.argsort(acc_matrix, axis=1)
second_idx = sorted_idx[:, -2]   # segundo mayor
gdf["second_model_ACC"] = [acc_cols[i] for i in second_idx]

# ============================================================
# 5. Colores Glasbey (máxima separación)
# ============================================================
model_colors_ACC = {
    "acc_rawGPM":   "#FF0000",   # rojo
    "acc_gwrGPM":   "#00A2FF",   # azul brillante
    "acc_PISCO":    "#00CC44",   # verde
    "acc_rain4pe":  "#FFAA00",   # naranja
    "acc_expGPM":   "#AA00FF"    # púrpura
}

# ============================================================
# 6. Asignar colores
# ============================================================
gdf["color_best_ACC"] = gdf["best_model_ACC"].map(model_colors_ACC)
gdf["color_second_ACC"] = gdf["second_model_ACC"].map(model_colors_ACC)

# ============================================================
# 7. FIGURA 1×2 — PAPER READY
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=600)

titles = ["(a) Best Model (ACC skill)", "(b) Second Best Model (ACC skill)"]
cols = ["color_best_ACC", "color_second_ACC"]
labels = ["best_model_ACC", "second_model_ACC"]

for ax, title, col, lab in zip(axes, titles, cols, labels):

    # Boundary
    boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8)

    # Points
    gdf.plot(
        ax=ax,
        color=gdf[col],
        markersize=140,
        edgecolor="white",
        linewidth=0.8
    )

    # Legend manual
    for model, color in model_colors_ACC.items():
        ax.scatter([], [], color=color, label=model.replace("acc_", ""), s=140)

    ax.legend(
        title="Model",
        frameon=True,
        fontsize=12,
        title_fontsize=13,
        loc="lower left"
    )

    ax.set_title(title, fontsize=18, pad=15)
    ax.set_axis_off()

plt.tight_layout()
plt.show()
