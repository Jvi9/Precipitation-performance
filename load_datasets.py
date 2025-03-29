# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 20:28:06 2025

@author: jvila
"""
import os
import xarray as xr
import pandas as pd
import pickle

"""
    Raising final_data , Station class with the ANA data uploaded
    """
save_path = r"C:\Users\jvila\Desktop\Andean_project\final_data.pkl"
# Load the dictionary from the specified path
with open(save_path, 'rb') as file:
    final_data = pickle.load(file)
print(f"Dictionary loaded successfully from {save_path}!")


'''Calling PISCO DATA'''
pisco_path = os.path.join(r'D:\Thesis_IUPWARE\Thesis file\2.0 Data\Input\PISCOp_daily.nc').replace(os.sep, '/')
x = xr.open_dataset(pisco_path)
#To use pisco with the funtions like GPM, it needs to modify the variable names af follows in the 5 lines
pisco_daily=xr.open_dataset(pisco_path)
pisco_daily=pisco_daily.rename({'pc':'precipitationCal'})
pisco_daily=pisco_daily.rename({'z':'time'})
pisco_daily['time'] = pd.to_datetime('1981-01-01') + pd.to_timedelta(pisco_daily['time'], 'D')
pisco_daily['time'] = pisco_daily['time'].dt.strftime('%Y-%m-%d')
pisco_daily['time'] = pd.to_datetime(pisco_daily['time'].values)

for key in final_data.keys():
    final_data[key]._point_pp_extr(pisco_daily, 'precipitationCal', attribute_name= 'PISCO',time='time')

'''Calling GPM raw data'''
GPM_path = os.path.join(r'D:\Thesis_IUPWARE\Thesis file\2.0 Data\Input\GPM_day_2000_2020.nc').replace(os.sep, '/')
gpm_daily = xr.open_dataset(GPM_path)
for key in final_data.keys():
    final_data[key]._point_pp_extr(gpm_daily, 'precipitationCal', attribute_name= 'rawGPM',time='time')



# =============================================================================
#   Playing ground

dummy1 = final_data['Crisnejas_ San Marcos']
dummy1.plot_ana()
dummy1.PISCO.loc['2015-01-01':'2015-06-30']
pbias, mae, rmse = dummy1._performance('data','PISCO','2015-01-01','2015-06-30')
dummy1.plot_pisco('precipitationCal', pointname=dummy1.name)
pbias1, mae1, rmse1 = dummy1._performance('data','rawGPM','2015-01-01','2015-06-30')

"""Posible solution for the storage of all the statistics""" 
class statsPISCO():
    def __init__(self, station_name):
        self.name = station_name
        self.mae = None
        self.pbias = None
        self.rmse = None
        
class statsrawGPM():
    def __init__(self, station_name):
        self.name = station_name
        self.mae = None
        self.pbias = None
        self.rmse = None
        
class statsgwrGPM():
    def __init__(self, station_name):
        self.name = station_name
        self.mae = None
        self.pbias = None
        self.rmse = None
        
class statsdmultiGPM():
    def __init__(self, station_name):
        self.name = station_name
        self.mae = None
        self.pbias = None
        self.rmse = None
        
class statsrain4pe():
    def __init__(self, station_name):
        self.name = station_name
        self.mae = None
        self.pbias = None
        self.rmse = None        
# =============================================================================
