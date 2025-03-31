# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 18:30:21 2025

@author: jvila
"""

import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Care of this because its just how your terminal runs
os.chdir(r"C:\Users\jvila\Desktop\Andean_project")

"""
    Raising final_data , Station class with the ANA data uploaded
    """
save_path = r"C:\Users\jvila\Desktop\Andean_project\final_data_loaded.pkl"
# Load the dictionary from the specified path as final_data
with open(save_path, 'rb') as file:
    final_data = pickle.load(file) 
print(f"Dictionary loaded successfully from {save_path}!")

"""Posible solution for the storage of all the statistics
    Obiously all the data are being compared to the observations""" 
class statsPISCO():
    def __init__(self, station_name, mae, pbias, rmse, fbi, far, pod, acc, lat, lon, alt):
        self.name = station_name
        self.mae = mae
        self.pbias = pbias
        self.rmse = rmse
        self.fbi = fbi
        self.far = far
        self.pod = pod
        self.acc = acc
        self.lat = lat
        self.lon = lon
        self.alt = alt

class statsrain4pe():
    def __init__(self, station_name, mae, pbias, rmse, fbi, far, pod, acc, lat, lon, alt):
        self.name = station_name
        self.mae = mae
        self.pbias = pbias
        self.rmse = rmse    
        self.fbi = fbi
        self.far = far
        self.pod = pod
        self.acc = acc
        self.lat = lat
        self.lon = lon
        self.alt = alt
        
class statsrawGPM():
    def __init__(self, station_name, mae, pbias, rmse, fbi, far, pod, acc, lat, lon, alt):
        self.name = station_name
        self.mae = mae
        self.pbias = pbias
        self.rmse = rmse
        self.fbi = fbi
        self.far = far
        self.pod = pod
        self.acc = acc
        self.lat = lat
        self.lon = lon
        self.alt = alt
        
class statsgwrGPM():
    def __init__(self, station_name, mae, pbias, rmse, fbi, far, pod, acc, lat, lon, alt):
        self.name = station_name
        self.mae = mae
        self.pbias = pbias
        self.rmse = rmse
        self.fbi = fbi
        self.far = far
        self.pod = pod
        self.acc = acc
        self.lat = lat
        self.lon = lon
        self.alt = alt
        
class statsdmultiGPM():
    def __init__(self, station_name, mae, pbias, rmse, fbi, far, pod, acc, lat, lon, alt):
        self.name = station_name
        self.mae = mae
        self.pbias = pbias
        self.rmse = rmse
        self.fbi = fbi
        self.far = far
        self.pod = pod
        self.acc = acc
        self.lat = lat
        self.lon = lon
        self.alt = alt

class statsdexpGPM():
    def __init__(self, station_name, mae, pbias, rmse, fbi, far, pod, acc, lat, lon, alt):
        self.name = station_name
        self.mae = mae
        self.pbias = pbias
        self.rmse = rmse
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
        station.rawGPM.index = pd.to_datetime(station.rawGPM.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        station.gwrGPM.index = pd.to_datetime(station.gwrGPM.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        station.PISCO.index = pd.to_datetime(station.PISCO.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        station.rain4pe.index = pd.to_datetime(station.rain4pe.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        station.data.index = pd.to_datetime(station.data.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    print("All datetime indexes migrated to pandas datetime format for all stations.")
        
def raise_stats(object_creating ,dictionary: dict, obs_attr: str, sim_attr: str, start_date: str, end_date: str, min_obs_threshold: float):
    """
    Uses the inherited methods from Stations Class to invoke all the stadistics
    mae, pbias, rmse, fbi, far, pod and acc
    Args:
        object_creating: Specific Class created to store the attribute
                         choose from:  statsPISCO, statsrain4pe, statsrawGPM
                         statsgwrGPM, statsdmultiGPM, statsdexpGPM
        obs_attr (str): Attribute name for observed data, usually as 'data'.
        sim_attr (str): Attribute name for simulated data named in Station class.
        min_obs_threshold (float): Min precipitation analyzed in simulations in mm.

    Returns:
        tuple: FBI, FAR, POD and Accuracy values as floats.
    """
    dummy_dictionary = {}
    
    for key in dictionary.keys():
        station_name = key
        pbias, mae, rmse = dictionary[key]._performance(obs_attr, sim_attr, start_date, end_date)
        fbi, far, pod, acc  = dictionary[key]._detection_capability(obs_attr, sim_attr, start_date, end_date, min_obs_threshold)
        lat, lon, alt = dictionary[key].lat, final_data[key].lon, final_data[key].alt
        dummy_dictionary[key] = object_creating(station_name, mae, pbias, rmse, fbi, far, pod, acc, lat, lon, alt)
        
    return dummy_dictionary


def parameters_report(dictionary: dict):
    """
    Args:
        dictionary: Dictionary full of stats?Class with the statistics as floats
    Returns:
        A dataframe with the station names as index and the stats results as columns
    """
    # Collect all summaries dynamically
    combined_summary = pd.DataFrame({
        metric: [getattr(dictionary[key], metric) for key in dictionary.keys()]
        for metric in ['mae', 'pbias', 'rmse', 'fbi', 'far', 'pod', 'acc']  # Add more metrics here as needed
    }, index=list(dictionary.keys()))
    
    return combined_summary

def station_summary(dictionary: dict, metrics: list = None):
    """If not metric is provided all the parameters are being calculated
    Default metrics if none are provided
    =['mae', 'pbias', 'rmse', 'fbi', 'far', 'pod', 'acc']"""

    if metrics is None:
        metrics = ['mae', 'pbias', 'rmse', 'fbi', 'far', 'pod', 'acc']
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
# =============================================================================
# Code execution
# =============================================================================
set_pandas_time(final_data) # To set everything as the same index type

#Runs all the scenarios
statsrawGPM_dict = raise_stats(statsrawGPM, final_data, 'data', 'rawGPM', '2005-01-01','2018-12-31', min_obs_threshold=1)
statsgwrGPM_dict = raise_stats(statsrawGPM, final_data, 'data', 'gwrGPM', '2005-01-01','2018-12-31', min_obs_threshold=1)
statsPISCO_dict = raise_stats(statsrawGPM, final_data, 'data', 'PISCO', '2005-01-01','2018-12-31', min_obs_threshold=1)
statsrain4pe_dict = raise_stats(statsrawGPM, final_data, 'data', 'rain4pe', '2005-01-01','2018-12-31', min_obs_threshold=1)
"""Pending to start the exponential regression"""

#To visualize a report of one analysis for all the stations
report_rawGPM = parameters_report(statsrawGPM_dict)

#Calls one analysis stats
Cajabamba = station_summary(statsrawGPM_dict['Crisnejas_ Cajabamba'], metrics = ['fbi', 'far', 'pod', 'acc'])

#Calls one station over multiple analysis 
list_of_dict = ['statsrawGPM_dict','statsgwrGPM_dict','statsPISCO_dict','statsrain4pe_dict']
SondorMatara=metrics_over_analysis(list_of_dict, 'Crisnejas_ Sondor-Matara')

# Calls a Station plot with all the stats | 
# warning: metrics_over_analysis have to be run before
plot_1station_stats(SondorMatara)

# =============================================================================
# Play ground
# =============================================================================
""" I used this for a common analysis between stations to find a min threshold
    time range for all stations in this case is '2005-01-01 00:00:00'
"""
list_mins = []
for key in final_data.keys():
    threshold_i=final_data[key].data.index.min()
    threshold_f=final_data[key].data.index.max()
    list_mins.append(threshold_i)
    print(f'{threshold_i} and {threshold_f}')
max(list_mins) #I used this for a common analysis between stations to find 
               # in this case is '2005-01-01 00:00:00' givin a 

#TESTING GRAPHS
import geopandas as gpd
from shapely.geometry import Point

# Extract data (latitude, longitude, accuracy)
lats = []  # Latitude values
lons = []  # Longitude values
accs = []  # Accuracy values

for key, lil in zip(stats_gwrGPM.keys(), final_data):
    acc = stats_gwrGPM[key].acc
    lat = final_data[lil].lat
    lon = final_data[lil].lon
    lats.append(lat)
    lons.append(lon)
    accs.append(acc)

# Create a GeoDataFrame
name = 'stats_gwrGPM'
geometry = [Point(lon, lat) for lon, lat in zip(lons, lats)]
geo_df = gpd.GeoDataFrame({'accuracy': accs, 'geometry': geometry})  # Add accuracy as a column

# Plot the map with color representation for accuracy
fig, ax = plt.subplots(figsize=(12, 8))

# Use the GeoDataFrame to plot points with a color map
geo_df.plot(ax=ax, column='accuracy', cmap='RdYlGn', markersize=50, legend=True)

# Add labels and title
plt.title(f'Color Map Accuracy {name}')
plt.ylabel('Latitudes')
plt.xlabel('Longitudes')
plt.show()
