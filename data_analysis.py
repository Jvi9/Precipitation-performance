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


# =============================================================================
# Play ground
# =============================================================================
explore1=  final_data['Crisnejas_ La Encañada']
pbias, mae, rmse = explore1._performance('data','gwrGPM','2010-01-01','2015-06-30')
pbias, mae, rmse = explore1._performance('data','rawGPM','2010-01-01','2015-06-30')

list_mins = []
for key in final_data.keys():
    threshold_i=final_data[key].data.index.min()
    threshold_f=final_data[key].data.index.max()
    list_mins.append(threshold_i)
    print(f'{threshold_i} and {threshold_f}')
max(list_mins) #threshold to use for a common analysis between stations
                # in this case is '2005-01-01 00:00:00' givin a 
stats_rawGPM = {}
for key in final_data.keys():
    station_name = key
    pbias, mae, rmse = final_data[key]._performance('data','rawGPM','2005-01-01','2018-12-31')
    stats_rawGPM[key] = statsrawGPM(station_name, mae, pbias, rmse)

mae_summary1 = pd.DataFrame({'mae':[stats_rawGPM[key].mae for key, value in stats_rawGPM.items()]}, index=list(final_data.keys()))

stats_gwrGPM = {}
for key in final_data.keys():
    station_name = key
    pbias, mae, rmse = final_data[key]._performance('data','gwrGPM','2005-01-01','2018-12-31')
    stats_gwrGPM[key] = statsgwrGPM(station_name, mae, pbias, rmse)

mae_summary2 = pd.DataFrame({'mae':[stats_gwrGPM[key].mae for key, value in stats_gwrGPM.items()]}, index=list(final_data.keys()))

stats_PISCO = {}
for key in final_data.keys():
    station_name = key
    pbias, mae, rmse = final_data[key]._performance('data','PISCO','2005-01-01','2018-12-31')
    stats_PISCO[key] = statsPISCO(station_name, mae, pbias, rmse)

mae_summary3 = pd.DataFrame({'mae':[stats_PISCO[key].mae for key, value in stats_PISCO.items()]}, index=list(final_data.keys()))

stats_rain4pe = {}
for key in final_data.keys():
    station_name = key
    pbias, mae, rmse = final_data[key]._performance('data','PISCO','2005-01-01','2018-12-31')
    stats_rain4pe[key] = statsrain4pe(station_name, mae, pbias, rmse)

mae_summary4 = pd.DataFrame({'mae':[stats_rain4pe[key].mae for key, value in stats_rain4pe.items()]}, index=list(final_data.keys()))

print(f'MAE rawGPM {stats_rawGPM['Crisnejas_ La Encañada'].mae}')
print(f'MAE gwrGPM {stats_gwrGPM['Crisnejas_ La Encañada'].mae}')
print(f'MAE PISCO {stats_PISCO['Crisnejas_ La Encañada'].mae}')
print(f'MAE rain4pe {stats_rain4pe['Crisnejas_ La Encañada'].mae}')

print(f'pbias rawGPM {stats_rawGPM['Crisnejas_ La Encañada'].pbias}')
print(f'pbias gwrGPM {stats_gwrGPM['Crisnejas_ La Encañada'].pbias}')
print(f'pbias PISCO {stats_PISCO['Crisnejas_ La Encañada'].pbias}')
print(f'pbias rain4pe {stats_rain4pe['Crisnejas_ La Encañada'].pbias}')

print(f'rmse rawGPM {stats_rawGPM['Crisnejas_ La Encañada'].rmse}')
print(f'rmse gwrGPM {stats_gwrGPM['Crisnejas_ La Encañada'].rmse}')
print(f'rmse PISCO {stats_PISCO['Crisnejas_ La Encañada'].rmse}')
print(f'rmse rain4pe {stats_rain4pe['Crisnejas_ La Encañada'].rmse}')

#Calculate the detection capability
# FBI / Frecuency bias index
# FAR / False Alarm ratio
# POD / Probably of detection

#I need to calculate:
    #                   Gauges rain     Gauges no rain
    #Satellite rain         a               b
    #satellite no rain      c               d

#FBI = (a+b)/(a+c)
#FAR = b/(a+c)
#POD = a/(a+c)
    
# Using summary_dictionary comparing the GPM raw and the stations
# a= if 'Average' >0 and 'Ground_based'>0 ,True
# b= if 'Average' >0 and 'Ground_based'=0 ,True
# c= if 'Average' =0 and 'Ground_based'>0 ,True
# d= if 'Average' =0 and 'Ground_based'=0 ,True

summary_dictionary
for data in summary_dictionary.values():
    data['a'] = (data['Average'] > 1) & (data['station_groundbased'] > 0) 
    data['b'] = (data['Average'] > 1) & (data['station_groundbased'] == 0)  #1mm threshold
    data['c'] = (data['Average'] == 0) & (data['station_groundbased'] > 0) 
    data['d'] = (data['Average'] == 0) & (data['station_groundbased'] > 0)
for key, data in summary_dictionary.items():
    # Count the number of True values in columns 'a', 'b', and 'c'
    a_count = data['a'].sum()
    b_count = data['b'].sum()
    c_count = data['c'].sum()
    
    # Calculate the indicators
    FBI = (a_count + b_count) / (a_count + c_count)
    FAR = b_count / (a_count + c_count)
    POD = a_count / (a_count + c_count)
    
    # Print or store the indicators
    print(f"For {key}:")
    print(f"FBI: {FBI}")
    print(f"FAR: {FAR}")
    print(f"POD: {POD}")