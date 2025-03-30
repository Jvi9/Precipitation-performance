# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 18:30:21 2025

@author: jvila
"""

import os
import pandas as pd
import pickle

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

def set_pandas_time(dictionary_of_stations):
    for key, station in dictionary_of_stations.items():  # Loop through each Station instance
        # Convert all relevant DataFrame indexes to pandas datetime
        station.rawGPM.index = pd.to_datetime(station.rawGPM.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        station.gwrGPM.index = pd.to_datetime(station.gwrGPM.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        station.PISCO.index = pd.to_datetime(station.PISCO.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        station.rain4pe.index = pd.to_datetime(station.rain4pe.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        station.data.index = pd.to_datetime(station.data.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    print("All datetime indexes migrated to pandas datetime format for all stations.")

"""Posible solution for the storage of all the statistics""" 
class statsPISCO():
    def __init__(self, station_name, mae, pbias, rmse):
        self.name = station_name
        self.mae = mae
        self.pbias = pbias
        self.rmse = rmse
        
class statsrawGPM():
    def __init__(self, station_name, mae, pbias, rmse):
        self.name = station_name
        self.mae = mae
        self.pbias = pbias
        self.rmse = rmse
        
class statsgwrGPM():
    def __init__(self, station_name, mae, pbias, rmse):
        self.name = station_name
        self.mae = mae
        self.pbias = pbias
        self.rmse = rmse
        
class statsdmultiGPM():
    def __init__(self, station_name, mae, pbias, rmse):
        self.name = station_name
        self.mae = mae
        self.pbias = pbias
        self.rmse = rmse
        
class statsrain4pe():
    def __init__(self, station_name, mae, pbias, rmse):
        self.name = station_name
        self.mae = mae
        self.pbias = pbias
        self.rmse = rmse       
    
# =============================================================================
# Code execution
# =============================================================================
set_pandas_time(final_data) # To set everything as the same index type

explore1=  final_data['Crisnejas_ La Encañada']
pbias, mae, rmse = explore1._performance('data','gwrGPM','2010-01-01','2015-06-30')
pbias, mae, rmse = explore1._performance('data','rawGPM','2010-01-01','2015-06-30')

stats_rawGPM = {}
for key in final_data.keys():
    station_name = key
    pbias, mae, rmse = final_data[key]._performance('data','rawGPM','2010-01-01','2016-12-31')
    stats_rawGPM[key] = statsrawGPM(station_name, mae, pbias, rmse)

mae_summary1 = pd.DataFrame({'mae':[stats_rawGPM[key].mae for key, value in stats_rawGPM.items()]}, index=list(final_data.keys()))

stats_gwrGPM = {}
for key in final_data.keys():
    station_name = key
    pbias, mae, rmse = final_data[key]._performance('data','gwrGPM','2010-01-01','2016-12-31')
    stats_gwrGPM[key] = statsgwrGPM(station_name, mae, pbias, rmse)

mae_summary2 = pd.DataFrame({'mae':[stats_gwrGPM[key].mae for key, value in stats_gwrGPM.items()]}, index=list(final_data.keys()))


stats_rawGPM['Crisnejas_ La Encañada'].mae
stats_gwrGPM['Crisnejas_ La Encañada'].mae
