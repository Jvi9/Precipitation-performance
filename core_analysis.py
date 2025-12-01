# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 22:47:44 2025

@author: jvila
"""

""" I used this for a common analysis between stations to find a min threshold
    time range for all stations in this case is '2005-01-01 00:00:00'
list_mins = []
for key in final_data.keys():
    threshold_i=final_data[key].data.index.min()
    threshold_f=final_data[key].data.index.max()
    list_mins.append(threshold_i)
    print(f'{threshold_i} and {threshold_f}')
max(list_mins)
"""
# =============================================================================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import contextily as cx
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# functions to plot maps
def get_pod_far_min_max(df):
    pod_cols = [col for col in df.columns if col.startswith('pod')]
    far_cols = [col for col in df.columns if col.startswith('far')]

    pod_min = round(df[pod_cols].min().min(), 2)
    pod_max = round(df[pod_cols].max().max(), 2)
    
    far_min = round(df[far_cols].min().min(), 2)
    far_max = round(df[far_cols].max().max(), 2)
    
    return pod_min, pod_max, far_min, far_max

def geo_plot(datasummary:pd.DataFrame,
             geosummary:pd.DataFrame,
             colormetric:str,
             sizemetric:str,
             colormin:float,
             colormax:float,
             sizemin:float,
             sizemax:float,
             boundary_path,
             title:str):
    
    boundary = gpd.read_file(boundary_path)
    boundary = boundary.to_crs(epsg=3857)
    # Merge your coordinates with your data
    merged_df = pd.merge(
        geosummary,
        datasummary,
        left_on='station',
        right_index=True,
        how='inner'
    )

    # Create a GeoDataFrame for mapping
    geometry = [Point(lon, lat) for lon, lat in zip(merged_df['lon'], merged_df['lat'])]
    gdf = gpd.GeoDataFrame(merged_df, geometry=geometry, crs="EPSG:4326")

    # Choose which metrics to use
    far_metric = colormetric  # For color
    pod_metric = sizemetric  # For size

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

    # Create a muted colormap
    # colors = ['#f6eff7', '#bdc9e1', '#74a9cf', '#2b8cbe', '#045a8d']  # Muted blue palette
    # n_colors = 256
    # cmap = LinearSegmentedColormap.from_list('muted_blues', colors, N=n_colors)
    from matplotlib import cm
    cmap = cm.get_cmap('coolwarm', 256)
    # Get the min and max values for scaling
    far_min = colormin
    far_max = colormax
    pod_min = sizemin
    pod_max = sizemax

    # Scale the size values
    size_min = 100
    size_max = 500

    # Convert to Web Mercator projection for basemap compatibility
    gdf_webmerc = gdf.to_crs(epsg=3857)

    # Plot the GeoDataFrame with our visualization parameters
    gdf_webmerc.plot(
        ax=ax,
        markersize=gdf_webmerc[pod_metric].apply(lambda x: size_min + (size_max - size_min) * (x - pod_min) / (pod_max - pod_min)),
        column=far_metric,
        cmap=cmap,
        alpha=0.8,
        edgecolor='blue',
        linewidth=0.8,
        vmin=far_min,
        vmax=far_max,
        legend=False,
        zorder=5
    )

    # Add basemap
    # cx.add_basemap(ax, source=cx.providers.OpenTopoMap, zoom=8)
    cx.add_basemap(ax, source=cx.providers.NASAGIBS.ASTER_GDEM_Greyscale_Shaded_Relief, zoom=8)
    boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=3.0, zorder=4)

    # Add color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=far_min, vmax=far_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f'({far_metric})', fontsize=12)

    # Create size legend values (min, middle, max)
    size_legend_values = [round(pod_min, 2), round((pod_min + pod_max) / 2, 2), round(pod_max, 2)]
    size_legend_sizes = [size_min + (size_max - size_min) * (val - pod_min) / (pod_max - pod_min) 
                        for val in size_legend_values]

    # Add a legend for sizes
    for i, (val, size) in enumerate(zip(size_legend_values, size_legend_sizes)):
        ax.scatter([], [], c='gray', s=size, edgecolors='black', linewidths=0.5,
                  label=f'POD: {val:.2f}')

    ax.legend(title=f'({pod_metric})', loc='lower left', frameon=True, framealpha=0.85,labelspacing=1.2)

    # Remove axis ticks and labels (typical for maps)
    ax.set_axis_off()

    # Add title
    fig.suptitle(f'{title}', fontsize=14)

    # Adjust layout
    plt.tight_layout()

    # # Save figure in high-quality formats
    # plt.savefig('precipitation_metrics_georeferenced_map.png', dpi=300, bbox_inches='tight')
    # plt.savefig('precipitation_metrics_georeferenced_map.pdf', bbox_inches='tight')  # Vector format for publication

    plt.show()

def plot_extreme_indices_per_station(generator, station_name):

    sources = {
        "Observed": generator.extremeIndicesObserved,
        "rawGPM"  : generator.extremeIndicesrawGPM,
        "expGPM"  : generator.extremeIndicesexpGPM,
        "gwrGPM"  : generator.extremeIndicesgwrGPM,
        "PISCO"   : generator.extremeIndicesPISCO,
        "rain4pe" : generator.extremeIndicesrain4pe
    }

    index_names = ["cdd", "cwd", "r10", "r20", "r95p", "r99p"]
    index_data = {idx: {} for idx in index_names}

    for source_name, source_dict in sources.items():
        ei = source_dict.get(station_name)
        if ei is None:
            print(f"[Warning] {station_name} not found in {source_name}")
            continue
        for idx in index_names:
            index_series = getattr(ei, idx)
            index_data[idx][source_name] = index_series.sort_index()

    for idx in index_names:
        series_dict = index_data[idx]
        if not series_dict:
            continue

        # Union of all years from all series
        all_years = sorted(set().union(*[s.index for s in series_dict.values()]))
        x = np.arange(len(all_years)) 

        width = 0.12  
        n_sources = len(series_dict)
        offset = -((n_sources - 1) / 2) * width  

        plt.figure(figsize=(12, 5))
        for i, (source, series) in enumerate(series_dict.items()):
            y_vals = [series.get(year, np.nan) for year in all_years]
            plt.bar(x + i * width + offset, y_vals, width=width, label=source)

        plt.title(f"{idx.upper()} for {station_name}")
        plt.xlabel("Year")
        plt.ylabel(idx.upper())
        plt.xticks(x, all_years, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()


# =============================================================================
from analysis_class import *
import matplotlib.pyplot as plt

# Create the Generator instance, that process the data, you can explore them under:
# .statsrawGPM_dict : like generator_test.statsgwrGPM_dict
# .statsgwrGPM_dict 
# .statsPISCO_dict 
# .statsrain4pe_dict
# .statsexpGPM_dict

cwd = os.getcwd()

load_data_path  =   f'{cwd}/datasets/data_locked_loaded.pkl'
min_range       = '2005-01-01'
max_range       = '2018-12-31'
min_threshold   = 2.5

generator_test  = Generator(load_data_path,min_range,max_range, min_threshold)

with open("export_results/generator_export.pkl", "wb") as f:
    pickle.dump(generator_test, f)
# generator_test.plot_violin_stats(stat_name='alt')
# generator_test.plot_violin_precipitation()


# # Columns can be changed to any stat: ['mae', 'pbias', 'rmse', 'r', 'kge', 'fbi', 'far', 'pod', 'acc']
# generator_test.plot_probability_graph(pod_columnx='far', far_columny='pod')
# generator_test.plot_probability_graph(pod_columnx='alt', far_columny='pod')
# generator_test.plot_probability_graph(pod_columnx='min_sim', far_columny='rmse')
# minimos=generator_test.statsrain4pe_dict
# minimosgwr=generator_test.statsgwrGPM_dict

# Generates a summary for station, you can specify or exclude a list of scenarios, by default:
# list_of_dict = ['statsrawGPM_dict','statsgwrGPM_dict','statsPISCO_dict','statsrain4pe_dict','statsexpGPM_dict']  
# .metrics_over_analysis(station_name , list_of_scenarios) 
# station_test = generator_test.metrics_over_analysis('Crisnejas_ Sondor-Matara')
# plot_1station_stats(station_test)

# Generates a temporal plot analysis over the scenarios: 'monthly', 'yearly' or 'both'
# acumulate_comparison(generator_test.final_data, 'Mantaro_ Junin', 'yearly', '2005-01-01','2018-12-31')

# Sorts the stations in function of altitude in a dictionary of stations, it replaces the original sort
# altitude_sort=sort_altitude(generator_test.final_data.keys(), generator_test.final_data)       

# Returns a df summary of all stations over the analysis, it can be filtered with lists
# df = generator_test.join_stats(list_of_dictionaries = None, metrics = ['fbi', 'far', 'pod'])
# #Call a fbi, far, pod 'heat' map over multiple analysis
# heat_map(df)



# Plotting Extreme Indices for a random station
station = 'Perene_ Runatullo'
plot_extreme_indices_per_station(generator_test, station)


#%%


"""Quality check of the precipitation, it looks like aint a big problem"""
# bucket_test = PrecipitationBucket(load_data_path,min_range,max_range)
# bucket_test.checker()
# bucket_test.rain_quality()
# bucket_test.summary_graph()
# bucket_test.normal_graph()

"""
Podriamos utilizar:
    Seasonal filter
    -Probalilidad alta de lluvia de Diciembre a Marzo = 1
    -Probabilidad media de lluvia de Abril Mayo y Noviembre = 0.5
    -Probabilidad zero de lluvia de Junio a Octubre = 0
                Octubre o Mayo 0.5
                othos meses  0
    Weekly cicles: weekend activities, personel does shitty work : CONDITION 1
    - El valor del dia viernes se repite el sabado y domingo? o = 1
      El valor del sabado y domingo son cero = 0.5
      No se cumplen las condiciones anteriores = 0
    
    Poor condition measures: CONDITION 2
    - Hay valores que se repiten tres dias consecutivos differentes de cero? = 1
      Hay valores que se repiten dos dias consecutivos differentes de cero? = 0.5
      No se cumplen las condiciones anteriores = 0
    
    Heavy precipitation truncation: CONDITION 3
    - Buscar el maximo mensual, el valor maximo se repite en el mismo mes +/- un threshold (ejemplo 5 cm) = 1
        Los operarios muchas veces para 'no repetir' numeros varian medidas pasadas un poquito
    - Si la condicion anterior no se cumple = 0

                prob-lluvia     COND1   COND2   COND3
    enero           1           1       1       1
    febrero         1
    marzo           1
    abril           0.5
    mayo            0.5
    junio           0
    julio           0
    agosto          0
    septiembre      0
    octubre         0
    noviembre       0.5
    diciembre       1

entonces si:  
            prob-lluvia = 1 y cualquier condicion es = 1: Es muy probable que el dato es incorrecto Red Flag
            prob-lluvia = 0.5 y cualquier condicion es = 1 o 0.5: Warning, es probable que el dato no sea real Yellow flag
            prob-lluvia = 0 y todas las condiciones = 0 : OK, Green flag

"""

# =============================================================================
# sandbox
# =============================================================================
# patthenr = generator_test.final_data
# from analysis_class import set_pandas_time
# set_pandas_time(patthenr)

# yearly = {}
# for key in patthenr:
#     data = patthenr[key].gwrGPM.loc['2005-01-01':'2018-12-31'].resample("YE").sum()
#     obs[key] = data

# obs = {}

# for key in patthenr:
#     data = patthenr[key].data.loc['2005-01-01':'2018-12-31'].resample("YE").sum()
#     obs[key] = data
    
# def plot_all_dataframes(dictionary_of_dfs):
#     plt.figure(figsize=(12, 6))  # Set the figure size

#     for key, df in dictionary_of_dfs.items():
#         if 'precipitationCal' in df.columns and not df['precipitationCal'].isnull().all():
#             y = df['precipitationCal']
#         else:
#             y = df['Precipitation']

#         plt.plot(df.index, y, label=key)  # Plot with a label for the dictionary key

#     plt.title("Combined Plot of All DataFrames")
#     plt.xlabel("Date")
#     plt.ylabel("Value")
#     # plt.legend()  # Uncomment if you want legend
#     plt.grid(axis='both', linestyle="--", alpha=0.5)
#     plt.tight_layout()
#     plt.show()


# plot_all_dataframes(yearly)
# plot_all_dataframes(obs)


# =============================================================================
# GEO GRAPH
# =============================================================================
# creates geographs
# =============================================================================
boundary_path = r'C:\Users\jvila\Desktop\Andean_project\gis\study_area_shp\study_area.shp'
df = generator_test.join_stats(list_of_dictionaries = None, metrics = ['far', 'pod','acc'])
pod_min, pod_max, far_min, far_max = get_pod_far_min_max(df)
geo_sum = generator_test.geo_summary()

datasummary=df
geosummary=geo_sum
colormetric='far_rain4pe'
sizemetric='pod_rain4pe'
colormin=far_min
colormax=far_max
sizemin=pod_min
sizemax=pod_max
boundary_path=boundary_path
title='Scenario Rain4pe data'
geo_plot(datasummary,geosummary,colormetric,sizemetric,colormin,colormax,
              sizemin,sizemax,boundary_path,title)

# =============================================================================
# 
# =============================================================================

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx

# === Load your data ===
# Main data
df = generator_test.join_stats(list_of_dictionaries = None, metrics = ['far', 'pod','acc','pbias'])

# Geolocation data
geo_df = generator_test.geo_summary()
geo_df = geo_sum

# Merge dataframes
merged = df.reset_index().merge(geo_df, left_on='index', right_on='station')

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(merged, geometry=gpd.points_from_xy(merged.lon, merged.lat), crs="EPSG:4326")
gdf = gdf.to_crs(epsg=3857)  # for basemap compatibility

# Load study area shapefile and reproject
boundary_path = r'C:\Users\jvila\Desktop\Andean_project\gis\study_area_shp\study_area.shp'
boundary = gpd.read_file(boundary_path).to_crs(epsg=3857)

# === Variable groups for each row ===
from matplotlib.colors import LinearSegmentedColormap

# === Variable groups for each row ===
variables = [
    ['far_rawGPM', 'far_gwrGPM', 'far_PISCO', 'far_rain4pe', 'far_expGPM'],
    ['pod_rawGPM', 'pod_gwrGPM', 'pod_PISCO', 'pod_rain4pe', 'pod_expGPM'],
    ['acc_rawGPM', 'acc_gwrGPM', 'acc_PISCO', 'acc_rain4pe', 'acc_expGPM'],
    ['pbias_rawGPM', 'pbias_gwrGPM', 'pbias_PISCO', 'pbias_rain4pe', 'pbias_expGPM']

]

# === Compute row-wise min and max ===
vmins = []
vmaxs = []
for row_vars in variables:
    all_values = gdf[row_vars].values.flatten()
    vmins.append(all_values.min())
    vmaxs.append(all_values.max())

# === Plot setup ===
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(14, 12))
fig.suptitle('Metric Maps', fontsize=16)

# Flatten axes for easier iteration
axes = axes.flatten()

# === Custom colormap ===
cmap = LinearSegmentedColormap.from_list("blue_red", ["red", "blue"])

# === Plot loop ===
for row_idx, row_vars in enumerate(variables):
    for col_idx, metric in enumerate(row_vars):
        i = row_idx * 5 + col_idx
        ax = axes[i]
        
        # Plot boundary
        boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

        # Plot points with unified color scale for the row
        gdf.plot(
            ax=ax,
            column=metric,
            cmap=cmap,
            markersize=10,
            legend=True,
            # vmin=vmins[row_idx],
            # vmax=vmaxs[row_idx]
        )

        # Add basemap (optional)
        # cx.add_basemap(ax, source=cx.providers.OpenTopoMap, zoom=8)
        
        ax.set_title(metric)
        ax.set_axis_off()

plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.3, w_pad=0.05, h_pad=0.05)
plt.show()

for key, obj in generator_test.statsexpGPM_dict.items():
    try:
        obj.alt = pd.to_numeric(obj.alt, errors='coerce')
        print(f"{obj.alt}")
    except Exception as e:
        print(f"{key}: Failed to convert alt ({e})")
    
    
for key, obj in generator_test.statsexpGPM_dict.items():
    print(f"\n{key}:")
    for attr in dir(obj):
        if not attr.startswith('_'):
            try:
                val = getattr(obj, attr)
                print(f"  {attr}: {type(val)}")
            except Exception as e:
                print(f"  {attr}: Error accessing ({e})")

for key, obj in generator_test.statsexpGPM_dict.items():
    if obj.alt is not None and obj.alt < 0:
        print(f"{key}: {obj.alt}")