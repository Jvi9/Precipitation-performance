# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 20:28:06 2025

@author: jvila

Script to load datasets from different products for each station
Saved in pickle: final_data_loaded.pkl

Project: Evaluating precipitation datasets on the Andean region of Peru
"""
import os
import xarray as xr
import pandas as pd
import pickle

# Care of this because its just how your terminal runs

# Jhon's WD: "C:\Users\jvila\Desktop\Andean_project"
# Jose's WD: 'C:\\Users\\joset\\OneDrive - Vrije Universiteit Brussel\\Paper_peru\\Precipitation-performance\\Precipitation-performance'

wkDir = r'C:\\Users\\joset\\OneDrive - Vrije Universiteit Brussel\\Paper_peru\\Precipitation-performance\\Precipitation-performance'
os.chdir(wkDir)


#
"""
    Raising final_data , Station class with the ANA data uploaded
    """

final_data_file = "final_data.pkl"
save_path = f"{wkDir}/{final_data_file}"

# Load the dictionary from the specified path as final_data
with open(save_path, 'rb') as file:
    final_data = pickle.load(file) 
print(f"Dictionary loaded successfully from {save_path}!")


'''Calling PISCO DATA'''
# Extracts the lan|lon point for every station from the dictionary of Stations
def get_PISCO(dataset_path, stations_dictionary): 
    #To use pisco with the funtions like GPM, it needs to modify the variable names af follows in the 5 lines
    pisco_daily=xr.open_dataset(dataset_path)
    pisco_daily=pisco_daily.rename({'pc':'precipitationCal'})
    pisco_daily=pisco_daily.rename({'z':'time'})
    pisco_daily['time'] = pd.to_datetime('1981-01-01') + pd.to_timedelta(pisco_daily['time'], 'D')
    pisco_daily['time'] = pisco_daily['time'].dt.strftime('%Y-%m-%d')
    pisco_daily['time'] = pd.to_datetime(pisco_daily['time'].values)

    for key in stations_dictionary.keys():
        stations_dictionary[key]._point_pp_extr(pisco_daily, 'precipitationCal', attribute_name= 'PISCO',time='time')
    return

'''Calling GPM raw data'''
def get_rawGPM(dataset_path, stations_dictionary):
    gpm_daily = xr.open_dataset(dataset_path)
    for key in stations_dictionary.keys():
        stations_dictionary[key]._point_pp_extr(gpm_daily, 'precipitationCal', attribute_name= 'rawGPM',time='time')
    return
'''
    Calling the rain4pe data    
'''
def get_rain4pe(dataset_path, stations_dictionary):
    dataset_opened = xr.open_dataset(dataset_path)
    dataset_opened = dataset_opened.rename({
        'Longitude': 'lon',
        'Latitude': 'lat',
        'pcp': 'precipitationCal'
    })
    for key in stations_dictionary.keys():
        station =  stations_dictionary[key]
        selection = dataset_opened.sel(lat=station.lat, lon=station.lon, method='nearest')
        precipitation = selection.precipitationCal.values
        timestamps = selection.time.values
        df_extracted = pd.DataFrame({'precipitationCal': precipitation}, index=timestamps)    
        stations_dictionary[key].rain4pe = df_extracted
        print(f'Rain4Pe data loaded for station {key}')
    return
'''
    Calling the gwr data    
'''
def get_gwrGPM(dataset_path, stations_dictionary):
    # Define the directory path
    input_dir4 = dataset_path
    # Get a list of all NetCDF files in the directory
    nc_files = [os.path.join(input_dir4, file) for file in os.listdir(input_dir4) if file.endswith('.nc')]
    for key in stations_dictionary.keys():
        df_dict ={}
        for i, index in enumerate(nc_files):
            station =  stations_dictionary[key]
            data_opened = xr.open_dataarray(index)
            selection = data_opened.sel(lat=station.lat, lon=station.lon, method='nearest')
            precipitation = selection.values
            timestamps = selection.time.values
            df_dict[i] = pd.DataFrame({'precipitationCal': precipitation}, index=timestamps)
        
        # Concatenate all DataFrames in the dictionary
        stations_dictionary[key].gwrGPM = pd.concat(df_dict.values())    
        print(f'GWR data loaded for station {key}')
    return
             
'''
    Calling the exponential data    
'''
def get_expGPM(dataset_path, stations_dictionary):
    # Define the directory path
    input_dir4 = dataset_path
    # Get a list of all NetCDF files in the directory
    nc_files = [os.path.join(input_dir4, file) for file in os.listdir(input_dir4) if file.endswith('.nc')]
    for key in stations_dictionary.keys():
        df_dict ={}
        for i, index in enumerate(nc_files):
            station =  stations_dictionary[key]
            data_opened = xr.open_dataarray(index)
            selection = data_opened.sel(lat=station.lat, lon=station.lon, method='nearest')
            precipitation = selection.values
            timestamps = selection.time.values
            df_dict[i] = pd.DataFrame({'precip': precipitation}, index=timestamps)
        
        # Concatenate all DataFrames in the dictionary
        stations_dictionary[key].expGPM = pd.concat(df_dict.values())    
        print(f'expGPM data loaded for station {key}')
    return    

# =============================================================================
# Code execution
# =============================================================================
# get_PISCO(r'D:\Thesis_IUPWARE\Thesis file\2.0 Data\Input\PISCOp_daily.nc', final_data)
# get_rawGPM(r'D:\Thesis_IUPWARE\Thesis file\2.0 Data\Input\GPM_day_2000_2020.nc', final_data)
# get_gwrGPM(r"D:\Thesis_IUPWARE\Final_pp_daily", final_data)    
# get_rain4pe(r'C:\Users\jvila\Downloads\RAIN4PE_daily_0.1d_1981_2015_v1.0.nc', final_data)
#for jose to execute
expoGPM_path = r"D:\GPM_downscaled_peru\gpm_dwsc_daily"
get_expGPM(expoGPM_path, final_data)    

#Save the data selected for the study to not re-process it everytime
final_data_loaded_file = "final_data_loaded.pkl"
save_path = f"{wkDir}/{final_data_loaded_file}"
with open(save_path, 'wb') as file:
    pickle.dump(final_data, file)
print(f"Dictionary saved successfully to {save_path}!")

# =============================================================================
# Playing ground
# rain4pe_opened = xr.open_dataset(r'C:\Users\jvila\Downloads\RAIN4PE_daily_0.1d_1981_2015_v1.0.nc')
# rain4pe_opened = rain4pe_opened.rename({
#     'Longitude': 'lon',
#     'Latitude': 'lat',
#     'pcp': 'precipitationCal'
# })

# selection = rain4pe_opened.sel(lat=final_data['Crisnejas_ San Marcos'].lat, lon=final_data['Crisnejas_ San Marcos'].lon, method='nearest')
# selection.precipitationCal.values
# selection.time.values

# dummy1 = final_data['Crisnejas_ San Marcos']
# dummy1.plot_ana()
# dummy1.PISCO.loc['2015-01-01':'2015-06-30']
# pbias, mae, rmse = dummy1._performance('data','PISCO','2015-01-01','2015-06-30')
# dummy1.plot_pisco('precipitationCal', pointname=dummy1.name)
# pbias1, mae1, rmse1 = dummy1._performance('data','rawGPM','2015-01-01','2015-06-30')

# #Open one by one this is working
# dummy1 = final_data['Crisnejas_ San Marcos']
# gpmgwr = xr.open_dataarray(nc_files[2])
# y_variable1 = gpmgwr.sel(lat=dummy1.lat, lon=dummy1.lon, method='nearest')
# precipitation = y_variable1.values
# times = y_variable1.time.values
# df1 = pd.DataFrame(precipitation,times)

# gpmgwr2 = xr.open_dataarray(nc_files[3])
# y_variable2 = gpmgwr2.sel(lat=dummy1.lat, lon=dummy1.lon, method='nearest')
# precipitation = y_variable2.values
# times = y_variable2.time.values
# df2 = pd.DataFrame(precipitation,times)
# result = pd.concat([df1,df2])

# gpmgwr3 = xr.open_dataarray(nc_files[4])
# y_variable3 = gpmgwr3.sel(lat=dummy1.lat, lon=dummy1.lon, method='nearest')
# precipitation = y_variable3.values
# times = y_variable3.time.values
# df3 = pd.DataFrame(precipitation,times)

# result = pd.concat([df1, df2, df3]).sort_index()
# dummy1.gwrGPM = result

# df_dict ={}
# for i, index in enumerate(nc_files):
#     station =  final_data['Crisnejas_ San Marcos']
#     data_opened = xr.open_dataarray(index)
#     selection = data_opened.sel(lat=station.lat, lon=station.lon, method='nearest')
#     precipitation = selection.values
#     timestamps = selection.time.values
#     df_dict[i] = pd.DataFrame({'precipitationCal': precipitation}, index=timestamps)
# =============================================================================
