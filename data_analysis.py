# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 18:30:21 2025

@author: jvila
"""
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# =============================================================================
"""Posible solution for the storage of all the statistics
    Obiously all the data are being compared to the observations""" 
        
class productStats():
    def __init__(self, station_name, mae, pbias, rmse,r, kge, fbi, far, pod, acc, lat, lon, alt):
        self.name = station_name
        self.mae = mae
        self.pbias = pbias
        self.rmse = rmse
        self.r = r
        self.kge = kge
        self.fbi = fbi
        self.far = far
        self.pod = pod
        self.acc = acc
        self.lat = lat
        self.lon = lon
        self.alt = alt
        
def set_pandas_time(dictionary_of_stations):
    for key, station in dictionary_of_stations.items():  # Loop through each Station instance
        # Convert all relevant DataFrame indexes to pandas datetime
        # station.rawGPM.index = pd.to_datetime(station.rawGPM.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        # station.gwrGPM.index = pd.to_datetime(station.gwrGPM.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        # station.PISCO.index = pd.to_datetime(station.PISCO.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        # station.rain4pe.index = pd.to_datetime(station.rain4pe.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        station.data.index = pd.to_datetime(station.data.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        station.expGPM.index = pd.to_datetime(station.expGPM.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    print("All datetime indexes migrated to pandas datetime format for all stations.")
        
def raise_stats(object_creating ,dictionary: dict, obs_attr: str, sim_attr: str, start_date: str, end_date: str, min_obs_threshold: float):
    """
    Uses the inherited methods from Stations Class to invoke all the statistics
    mae, pbias, rmse, fbi, far, pod and acc
    Args:
        object_creating: Create instance of productStats class that has statistics as attributes
        obs_attr (str): Attribute name for observed data, usually as 'data'.
        sim_attr (str): Attribute name for simulated data named in Station class (The product).
        min_obs_threshold (float): Min precipitation analyzed in simulations in mm.

    Returns:
        Dictionary with productStats instances that correspond to a product for each station
    """
    dummy_dictionary = {}
    
    for key in dictionary.keys():
        station_name = key
        pbias, mae, rmse, r, kge = dictionary[key]._performance(obs_attr, sim_attr, start_date, end_date)
        fbi, far, pod, acc  = dictionary[key]._detection_capability(obs_attr, sim_attr, start_date, end_date, min_obs_threshold)
        lat, lon, alt = dictionary[key].lat, final_data[key].lon, final_data[key].alt
        dummy_dictionary[key] = object_creating(station_name, mae, pbias, rmse, r, kge, fbi, far, pod, acc, lat, lon, alt)
        
    return dummy_dictionary

def parameters_report(dictionary: dict):
    """
    Args:
        dictionary: Dictionary full of stats for each Class with the statistics as floats
    Returns:
        A dataframe with the station names as index and the stats results as columns
    """
    # Collect all summaries dynamically
    combined_summary = pd.DataFrame({
        metric: [getattr(dictionary[key], metric) for key in dictionary.keys()]
        for metric in ['mae', 'pbias', 'rmse', 'r', 'kge', 'fbi', 'far', 'pod', 'acc']  # Add more metrics here as needed
    }, index=list(dictionary.keys()))
    
    return combined_summary

def station_summary(dictionary: dict, metrics: list = None):
    """If not metric is provided all the parameters are being calculated
    Default metrics if none are provided
    =['mae', 'pbias', 'rmse', 'fbi', 'far', 'pod', 'acc']"""

    if metrics is None:
        metrics = ['mae', 'pbias', 'rmse', 'r', 'kge', 'fbi', 'far', 'pod', 'acc']
    # Dynamically create a DataFrame for all specified metrics
    
    summary = pd.DataFrame({
        metric: [getattr(dictionary, metric)]
        for metric in metrics
    }, index=[dictionary.name])
    return summary

def metrics_over_analysis(list_of_dictionaries: list, station_name: str):
    station_over_analysis = {}
    
    for item in list_of_dictionaries:
        # Access the dictionary object by its variable name
        try:
            source = globals()[item][station_name]  # Dynamically retrieve dictionary by name
        except KeyError:
            raise KeyError(f"Station name '{station_name}' not found in '{item}' dictionary.")
        except TypeError:
            raise TypeError(f"'{item}' is not a valid dictionary or does not contain the required station name.")
        
        # Generate the summary using the provided function
        summary = station_summary(source)
        station_over_analysis[item] = summary  # Store the summary in the result dictionary
    
    data_joined = pd.concat(station_over_analysis, names=['Key'])
    data_joined.reset_index(level=0, inplace=True)
    
    return data_joined

def plot_1station_stats(df, colors=None):
    """Use it after the metrics_over_analysis function to create a bar plot of the stats."""
    # Extract the common index name (assuming all rows share the same index value)
    common_index = df.index[0]  # Example: "Crisnejas_ Cajabamba"

    plt.figure(figsize=(12, 7))  # Larger figure for clarity
    
    # Number of metrics (columns except 'Key')
    metrics = df.columns[1:]  # Skip 'Key'
    num_metrics = len(metrics)
    x = np.arange(len(df['Key']))  # X-axis positions for each group of bars

    # Bar width to avoid overlap
    bar_width = 0.8 / num_metrics  # Divide the space for each metric group

    # Default color cycle if none provided
    if colors is None:
        colors = plt.cm.tab10.colors[:num_metrics]  # Automatically pick distinct colors

    # Loop through columns and create bars
    for i, column in enumerate(metrics):
        plt.bar(x + i * bar_width, df[column], bar_width, label=column, color=colors[i % len(colors)])
    
    # Add labels and title
    plt.title(f"Station Stats for {common_index}", fontsize=18)
    plt.xlabel("Analysis Result", fontsize=18)
    plt.ylabel("Values", fontsize=18)
    plt.xticks(x + bar_width * (num_metrics / 2 - 0.5), df['Key'],fontsize=16, rotation=0)  # Center the ticks
    plt.yticks(fontsize=16)
    plt.legend(loc='best', fontsize=16)
    
    # Display the grid and tighten layout
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
def key_dicname(full_string:str)->str: #function to extract the keyname
    extracted = full_string.replace("stats", "").replace("_dict", "")
    return extracted

def join_stats(list_of_dictionaries: [dict], metrics: list = None) -> pd.DataFrame:
    """
    Joins DataFrames created from a list of dictionaries by filtering specific metrics.
    
    Args:
        list_of_dictionaries (list[dict]): A list of dictionary names containing data.
        metrics (list): A list of metric names to filter the columns.
                        Defaults to ['mae', 'pbias', 'rmse', 'r', 'kge', 'fbi', 'far', 'pod', 'acc'].
    
    Returns:
        pd.DataFrame: A combined DataFrame with renamed and filtered columns.
    """
    # Set default metrics if none are provided
    if metrics is None:
        metrics = ['mae', 'pbias', 'rmse', 'r', 'kge', 'fbi', 'far', 'pod', 'acc']
    
    di = pd.DataFrame()  # Initialize an empty DataFrame
    
    for item in list_of_dictionaries:
        # Generate a DataFrame for the current dictionary
        df = parameters_report(globals()[item])
        
        # Filter columns to only keep those in the metrics list
        df = df[df.columns.intersection(metrics)]
        
        # Extract a meaningful name for renaming columns
        name = key_dicname(item)
        df = df.rename(columns=lambda col: f"{col}_{name}")
        
        # Concatenate the current DataFrame with the master DataFrame
        di = pd.concat([di, df], axis=1)
    
    return di

# def heat_map(data: pd.DataFrame):
#     # Sort the columns alphabetically
#     data = data.sort_index(axis=1)
#     stations = data.index.tolist()
#     variables = data.columns.tolist()

#     # Create a subplot for each variable
#     fig, axes = plt.subplots(1, len(variables), figsize=(2 * len(variables), 12), sharey=True)

#     for i, var in enumerate(variables):
#         if 'fbi' in var:
#             # Custom discrete colors for FBI
#             colors = data[var].apply(lambda x: 'red' if x > 1 else ('blue' if x < 1 else 'green'))
#             # Add legend for FBI
#             fbi_legend = [
#                 mpatches.Patch(color='red', label='FBI > 1'),
#                 mpatches.Patch(color='blue', label='FBI < 1'),
#                 mpatches.Patch(color='green', label='FBI = 1')
#             ]
#             axes[i].legend(handles=fbi_legend, loc='upper right', fontsize=8)
#         elif 'far' in var:
#             # Gradient (0 green to 1 red) for FAR
#             normalized_far = data[var] / data[var].max()  # Normalize FAR to [0, 1]
#             colors = [plt.cm.RdYlGn(1 - val) for val in normalized_far]
#             # Add legend for FAR
#             far_legend = [
#                 mpatches.Patch(color='red', label='FAR = 1 (red)'),
#                 mpatches.Patch(color='yellow', label='FAR = 0.5 (yellow)'),
#                 mpatches.Patch(color='green', label='FAR = 0 (green)')
#             ]
#             axes[i].legend(handles=far_legend, loc='upper right', fontsize=8)
#         elif 'pod' in var:
#             # Gradient (0 red to 1 green) for POD
#             normalized_pod = data[var] / data[var].max()  # Normalize POD to [0, 1]
#             colors = [plt.cm.RdYlGn(val) for val in normalized_pod]
#             # Add legend for POD
#             pod_legend = [
#                 mpatches.Patch(color='red', label='POD = 0 (red)'),
#                 mpatches.Patch(color='yellow', label='POD = 0.5 (yellow)'),
#                 mpatches.Patch(color='green', label='POD = 1 (green)')
#             ]
#             axes[i].legend(handles=pod_legend, loc='upper right', fontsize=8)
#         else:
#             # Default: No coloring
#             colors = 'gray'

#         # Plot the heatmap-like structure
#         axes[i].barh(stations, np.ones(len(stations)), color=colors)
#         axes[i].set_title(f'{var}')
#         axes[i].set_xlim(0, 1)
#         axes[i].set_xticks([])

#     plt.tight_layout()
#     plt.show()
    
def heat_map(data: pd.DataFrame):
    # Sort the columns alphabetically
    data = data.sort_index(axis=1)
    stations = data.index.tolist()
    variables = data.columns.tolist()

    # Create a subplot for each variable
    fig, axes = plt.subplots(1, len(variables), figsize=(2 * len(variables), 12), sharey=True)

    for i, var in enumerate(variables):
        if 'fbi' in var:
            # Normalize values for FBI to range [0, 1]
            normalized_fbi = (data[var] - 1).abs()  # Shift values around 1 to calculate absolute distance
            normalized_fbi /= normalized_fbi.max()  # Normalize to [0, 1]

            # Use coolwarm colormap (0 = blue, 1 = red)
            colors = [plt.cm.coolwarm(val) if x != 1 else 'green' for x, val in zip(data[var], normalized_fbi)]

            # Add legend for FBI
            fbi_legend = [
                mpatches.Patch(color='blue', label='FBI < 1 (blue)'),
                mpatches.Patch(color='red', label='FBI > 1 (red)'),
                mpatches.Patch(color='green', label='FBI = 1 (green)')
            ]
            axes[i].legend(handles=fbi_legend, loc='upper right', fontsize=8)

        elif 'far' in var:
            # Gradient (0 green to 1 red) for FAR
            normalized_far = data[var] / data[var].max()  # Normalize FAR to [0, 1]
            colors = [plt.cm.RdYlGn(1 - val) for val in normalized_far]
            
            # Add legend for FAR
            far_legend = [
                mpatches.Patch(color='red', label='FAR = 1 (red)'),
                mpatches.Patch(color='yellow', label='FAR = 0.5 (yellow)'),
                mpatches.Patch(color='green', label='FAR = 0 (green)')
            ]
            axes[i].legend(handles=far_legend, loc='upper right', fontsize=8)

        elif 'pod' in var:
            # Gradient (0 red to 1 green) for POD
            normalized_pod = data[var] / data[var].max()  # Normalize POD to [0, 1]
            colors = [plt.cm.RdYlGn(val) for val in normalized_pod]
            
            # Add legend for POD
            pod_legend = [
                mpatches.Patch(color='red', label='POD = 0 (red)'),
                mpatches.Patch(color='yellow', label='POD = 0.5 (yellow)'),
                mpatches.Patch(color='green', label='POD = 1 (green)')
            ]
            axes[i].legend(handles=pod_legend, loc='upper right', fontsize=8)

        else:
            # Default: No coloring
            colors = 'gray'

        # Plot the heatmap-like structure
        axes[i].barh(stations, np.ones(len(stations)), color=colors)
        axes[i].set_title(f'{var}')
        axes[i].set_xlim(0, 1)
        axes[i].set_xticks([])

    plt.tight_layout()
    plt.show()
    
def sort_altitude(stations: list, dictionary: dict) -> list:
    """
    Sorts station names based on their altitude in ascending order.
    
    Args:
        stations (list): A list of station keys (names or identifiers) to be looked up in the dictionary.
        dictionary (dict): A dictionary where the Station instances are stored, keyed by station names.
        
    Returns:
        list: A list of station names sorted by altitude from minimum to maximum.
    """
    # Create a DataFrame to hold the altitudes and names
    data = {'name': [], 'altitude': []}
    
    # Iterate through the station names
    for station_key in stations:
        station = dictionary[station_key]  # Retrieve the Station instance from the dictionary
        data['name'].append(station.name)  # Assume Station class has a `name` attribute
        data['altitude'].append(station.alt)  # Assume Station class has an `alt` attribute
    
    # Convert to a DataFrame
    df = pd.DataFrame(data)
    
    # Sort the DataFrame by altitude
    sorted_df = df.sort_values(by='altitude', ascending=True)
    
    # Return the sorted list of station names
    return sorted_df['name'].tolist()

def acumulate_comparison(station_dict, station_name: str, plotfrequency: str, start_date: str, end_date: str) -> None:
    """
    Generates visual representations (graphs) comparing monthly and/or yearly data
    for the specified station.

    Parameters:
        station_dict (dict): Station data dictionary containing daily values.
        station_name (str): Name of the station for comparison.
        plotfrequency (str): Frequency of the plot - 'monthly', 'yearly', or 'both'.

    Returns:
        None
    """
    # Prepare the data
    new_data = station_dict[station_name].copy()
    new_data.time_selection(start_date, end_date)

    parse_list = ['observations', 'PISCO', 'Rain4pe', 'rawGPM', 'gwrGPM', 'expGPM']
    df = pd.concat([new_data.data, new_data.PISCO, new_data.rain4pe, new_data.rawGPM,
                    new_data.gwrGPM, new_data.expGPM], axis=1)
    df.columns = parse_list
    df = df.applymap(lambda x: float(x) if isinstance(x, str) else x)

    # Resample for monthly and yearly data
    monthly_df = df.resample('ME').sum()
    yearly_df = df.resample('YE').sum()

    # Create combined plot for both
    if plotfrequency == 'both':
        x_monthly = monthly_df['observations']
        x_yearly = yearly_df['observations']
        y_columns = ['PISCO', 'Rain4pe', 'rawGPM', 'gwrGPM', 'expGPM']
        num_cols = len(y_columns)
    
        fig, axes = plt.subplots(nrows=2, ncols=num_cols, figsize=(8 * num_cols, 12), constrained_layout=True)
        
        fig.suptitle(f'Station {station_name.capitalize()} Data Comparison', fontsize=16, y=1.01)
        # Plot monthly data (first row)
        for i, col in enumerate(y_columns):
            y_monthly = monthly_df[col]
            axes[0, i].scatter(x_monthly, y_monthly, alpha=0.7, edgecolors='b')
            axes[0, i].set_title(f'Monthly - Observation vs {col}', fontsize=14)
            axes[0, i].set_xlabel('Observations', fontsize=12)
            axes[0, i].set_ylabel(col, fontsize=12)
            axes[0, i].grid(True)
    
            # Add 45-degree line
            min_val = min(x_monthly.min(), y_monthly.min())
            max_val = max(x_monthly.max(), y_monthly.max())
            axes[0, i].plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', label='45-degree line')
            axes[0, i].legend()
    
        # Plot yearly data (second row)
        for i, col in enumerate(y_columns):
            y_yearly = yearly_df[col]
            axes[1, i].scatter(x_yearly, y_yearly, alpha=0.7, edgecolors='r')
            axes[1, i].set_title(f'Yearly - Observation vs {col}', fontsize=14)
            axes[1, i].set_xlabel('Observations', fontsize=12)
            axes[1, i].set_ylabel(col, fontsize=12)
            axes[1, i].grid(True)
    
            # Add 45-degree line
            min_val = min(x_yearly.min(), y_yearly.min())
            max_val = max(x_yearly.max(), y_yearly.max())
            axes[1, i].plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', label='45-degree line')
            axes[1, i].legend()
    
        plt.show()

    elif plotfrequency == 'monthly':
        # Plot only monthly data
        create_iso_plot(monthly_df, 'monthly', station_name)

    elif plotfrequency == 'yearly':
        # Plot only yearly data
        create_iso_plot(yearly_df, 'yearly', station_name)

    else:
        raise ValueError("Invalid frequency. Choose from 'monthly', 'yearly', or 'both'.")

def create_iso_plot(df, plot_type:str,station_name:str):
    """
    Helper function to create single-row subplots for individual frequencies with a 45-degree reference line.
    """
    x = df['observations']
    y_columns = ['PISCO', 'Rain4pe', 'rawGPM', 'gwrGPM', 'expGPM']
    num_cols = len(y_columns)

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(8 * num_cols, 8), constrained_layout=True)
    
    fig.suptitle(f'Station {station_name.capitalize()} Data Comparison', fontsize=16, y=1.02)

    for i, col in enumerate(y_columns):
        y = df[col]
        # Create scatter plot
        axes[i].scatter(x, y, alpha=0.7, edgecolors='b')
        axes[i].set_title(f'{plot_type.capitalize()} - Observation vs {col}', fontsize=14)
        axes[i].set_xlabel('Observations', fontsize=12)
        axes[i].set_ylabel(col, fontsize=12)
        axes[i].grid(True)

        # Add 45-degree line
        # Ensure min and max values are valid (no NaN)
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        if pd.notnull(min_val) and pd.notnull(max_val):
            axes[i].plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', label='45-degree line')
            axes[i].legend()
        else:
            print(f"Skipping 45-degree line for {col} due to invalid min/max values.")

    plt.show()  
# =============================================================================
# Code execution
# =============================================================================
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
# Calling the data with all dataset sources.
# Care of this because its just how your terminal runs
wkDir = r'C:\Users\jvila\Desktop\Andean_project'
#C:\Users\jvila\Desktop\Andean_project
#'C:\\Users\\joset\\OneDrive - Vrije Universiteit Brussel\\Paper_peru\\Precipitation-performance\\Precipitation-performance'
os.chdir(wkDir)

#Raising final_data , Station class with the ANA data uploaded

save_path = f"{wkDir}/data_locked_loaded.pkl"
# Load the dictionary from the specified path as final_data
with open(save_path, 'rb') as file:
    final_data = pickle.load(file) 
print(f"Dictionary loaded successfully from {save_path}!")

    
#Runs all the scenarios and gets the statistics
statsrawGPM_dict = raise_stats(productStats, final_data, 'data', 'rawGPM', '2005-01-01','2018-12-31', min_obs_threshold=1)
statsgwrGPM_dict = raise_stats(productStats, final_data, 'data', 'gwrGPM', '2005-01-01','2018-12-31', min_obs_threshold=1)
statsPISCO_dict = raise_stats(productStats, final_data, 'data', 'PISCO', '2005-01-01','2018-12-31', min_obs_threshold=1)
statsrain4pe_dict = raise_stats(productStats, final_data, 'data', 'rain4pe', '2005-01-01','2018-12-31', min_obs_threshold=1)
statsexpGPM_dict = raise_stats(productStats, final_data, 'data', 'expGPM', '2005-01-01','2018-12-31', min_obs_threshold=1)

#To visualize a report of one analysis for all the stations
report_expGPM = parameters_report(statsexpGPM_dict)
report_rawGPM = parameters_report(statsrawGPM_dict)

# #Calls one analysis stats
# Cajabamba = station_summary(statsrawGPM_dict['Crisnejas_ Cajabamba'], metrics = ['fbi', 'far', 'pod', 'acc'])

# #Calls one station over multiple analysis 
list_of_dict = ['statsrawGPM_dict','statsgwrGPM_dict','statsPISCO_dict','statsrain4pe_dict','statsexpGPM_dict']
SondorMatara=metrics_over_analysis(list_of_dict, 'Crisnejas_ Sondor-Matara')

# # Calls a Station plot with all the stats | 
# # warning: metrics_over_analysis have to be run before
plot_1station_stats(SondorMatara)

# # Sorting the stations in function of altitude
altitude_sort=sort_altitude(final_data.keys(), final_data)        

# #Call a fbi, far, pod 'heat' map over multiple analysis
heat_map(join_stats(list_of_dict, ['fbi', 'far', 'pod']).reindex(altitude_sort))

acumulate_comparison(final_data, 'Mantaro_ Junin', 'monthly', '2005-01-01','2018-12-31')
acumulate_comparison(final_data, 'Mantaro_ Junin', 'yearly', '2005-01-01','2018-12-31')
acumulate_comparison(final_data, 'Mantaro_ Junin', 'both', '2005-01-01','2018-12-31')
acumulate_comparison(final_data, 'Crisnejas_ Sondor-Matara', 'both', '2005-01-01','2018-12-31')
acumulate_comparison(final_data, 'Mantaro_ Carhuacayan', 'both', '2005-01-01','2018-12-31')

# # =============================================================================
# # Play ground
# # =============================================================================
#WE NEED TO BUILD A GEOMAP GRAPH
# import geopandas as gpd
# from shapely.geometry import Point

# # Extract data (latitude, longitude, accuracy)
# lats = []  # Latitude values
# lons = []  # Longitude values
# accs = []  # Accuracy values

# for key, lil in zip(stats_gwrGPM.keys(), final_data):
#     acc = stats_gwrGPM[key].acc
#     lat = final_data[lil].lat
#     lon = final_data[lil].lon
#     lats.append(lat)
#     lons.append(lon)
#     accs.append(acc)

# # Create a GeoDataFrame
# name = 'stats_gwrGPM'
# geometry = [Point(lon, lat) for lon, lat in zip(lons, lats)]
# geo_df = gpd.GeoDataFrame({'accuracy': accs, 'geometry': geometry})  # Add accuracy as a column

# # Plot the map with color representation for accuracy
# fig, ax = plt.subplots(figsize=(12, 8))

# # Use the GeoDataFrame to plot points with a color map
# geo_df.plot(ax=ax, column='accuracy', cmap='RdYlGn', markersize=50, legend=True)

# # Add labels and title
# plt.title(f'Color Map Accuracy {name}')
# plt.ylabel('Latitudes')
# plt.xlabel('Longitudes')
# plt.show()

# Improving the readibility
# # set_pandas_time(final_data) # To set everything as the same index type
# def run_all_stats(final_data, data_field: str, start_date: str, end_date: str, min_obs_threshold=1):
#     """
#     Runs statistics for multiple products and returns a dictionary of results.

#     Parameters:
#         final_data (dict): Dictionary containing the data for stations.
#         data_field (str): Name of the main field ('data') to compare against other products.
#         start_date (str): Start date for the analysis.
#         end_date (str): End date for the analysis.
#         min_obs_threshold (int, optional): Minimum observation threshold. Default is 1.

#     Returns:
#         dict: A dictionary containing results for all products.
#     """
#     products = ['rawGPM', 'gwrGPM', 'PISCO', 'rain4pe', 'expGPM']
#     stats_dict = {}

#     for product in products:
#         stats_dict[product] = raise_stats(productStats, final_data, data_field, product, start_date, end_date, min_obs_threshold)

#     return stats_dict

# final_stats = run_all_stats(final_data, 'data', '2005-01-01', '2018-12-31', min_obs_threshold=1)
# print(final_stats['rawGPM'])  # Access the stats for 'rawGPM'
# print(final_stats['PISCO'])  # Access the stats for 'PISCO'



import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import gaussian_filter

# Load the DEM data from the TIFF file
tiff_file = r'C:\Users\jvila\Desktop\Andean_project\gis\study_area_1km_res.tif'   # Replace with the path to your TIFF file
with rasterio.open(tiff_file) as src:
    dem_data = src.read(1)  # Read the first band (altitudes)
    transform = src.transform  # Get the transformation metadata

# Smooth the DEM data using Gaussian filter
sigma = 10  # Adjust sigma value to control smoothing level
smoothed_dem_data = gaussian_filter(dem_data, sigma=sigma)

# Create the coordinates (X, Y) based on the transform
rows, cols = smoothed_dem_data.shape
x = np.linspace(transform[2], transform[2] + transform[0] * cols, cols)
y = np.linspace(transform[5], transform[5] + transform[4] * rows, rows)
X, Y = np.meshgrid(x, y)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Surface plot representation
surf = ax.plot_surface(X, Y, smoothed_dem_data, cmap='terrain', edgecolor='none')

# Add color bar for reference (optional)
color_bar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
color_bar.set_label('Altitude (meters)')

# Set labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Altitude')
ax.set_title('Smoothed 3D Surface Plot of DEM')

# Show the plot
plt.show()

import geopandas as gpd
import matplotlib.pyplot as plt

# Create a sample GeoDataFrame with multipoints and additional fields
data = {
    'Longitude': [2.3522, -3.7038, 4.8357, -0.1278],  # Example longitudes
    'Latitude': [48.8566, 40.4168, 45.7640, 51.5074],  # Example latitudes
    'Population': [2148000, 3223000, 513300, 8982000],  # Size field
    'Elevation': [35, 667, 162, 11]  # Color field
}

# Convert to a GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['Longitude'], data['Latitude']))

# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))

# Use 'Population' for marker size and 'Elevation' for color
gdf.plot(
    ax=ax,
    markersize=gdf['Population'] / 10000,  # Scale the population for size
    column='Elevation',  # Use the Elevation column for color mapping
    cmap='viridis',  # Colormap for the Elevation field
    legend=True  # Add a legend for the colormap
)

# Add labels and title
ax.set_title('Map with Multipoints (Size & Color Differentiators)', fontsize=14)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Show the plot
plt.show()

import matplotlib.pyplot as plt

def plot_probability_graph(data_dict, pod_columnx='pod', far_columny='far'):
    """
    Plots multiple datasets on the same graph, differentiating them by color and legend.
    
    Parameters:
    - data_dict: A dictionary where keys are dataset labels and values are DataFrames containing the data.
    - pod_column: Name of the column representing POD values.
    - far_column: Name of the column representing FAR values.
    """
    # Define colors for different datasets
    colors = ['green', 'blue', 'red', 'orange', 'purple']
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Iterate over the datasets and plot each
    for i, (label, data) in enumerate(data_dict.items()):
        if i >= len(colors):  # In case there are more datasets than colors
            color = 'gray'  # Default to gray
        else:
            color = colors[i]
        
        pod = data[pod_columnx].tolist()
        far = data[far_columny].tolist()
        
        plt.scatter(pod, far, color=color, label=label, alpha=0.8)  # Scatter plot for each dataset
    
    # Customize the graph
    plt.xlabel(f'{pod_columnx}')
    plt.ylabel(f'{far_columny}')
    plt.title('Detection capabilities in all stations')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example Usage
# Assuming parameters_report(statsPISCO_dict), parameters_report(statsgwrGPM_dict), etc., return DataFrames
data_dict = {
    'Dataset 1: statsPISCO': parameters_report(statsPISCO_dict),
    'Dataset 2: statsgwrGPM': parameters_report(statsgwrGPM_dict),
    'Dataset 3: statsexpGPM': parameters_report(statsexpGPM_dict),
    'Dataset 4: statsrawGPM': parameters_report(statsrawGPM_dict),
    'Dataset 5: statsrain4pe': parameters_report(statsrain4pe_dict)
    }

plot_probability_graph(data_dict)
plot_probability_graph(data_dict, pod_columnx='r', far_columny='rmse')
