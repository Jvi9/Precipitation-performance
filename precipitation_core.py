"""
    The goal here will be create a class call station that can store
    all datasets in an attribute to later just cast the stats needed
    between them
"""
import os
import csv
import pandas as pd
import numpy as np
import xarray as xr
# import requests 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# import geopandas as gpd
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from shapely.geometry import Polygon
# from tqdm import tqdm  #status bar
# from rasterio.warp import calculate_default_transform, reproject, Resampling
# from rasterio.windows import from_bounds
# from sklearn.linear_model import LinearRegression
# from rasterio.transform import from_origin
# from rasterio.enums import Resampling
# from scipy.ndimage import generic_filter
# from sklearn.metrics import r2_score
# from scipy.optimize import curve_fit
# import rasterio
# from rasterio.plot import show
# import re
from datetime import datetime, timedelta
# from dateutil.relativedelta import relativedelta
# import pysal
# from mpl_toolkits.mplot3d import Axes3D
# from mgwr.gwr import GWR, MGWR
# from mgwr.sel_bw import Sel_BW
# from scipy.interpolate import griddata
# import cftime
# from scipy.interpolate import RectBivariateSpline
# import rioxarray
# from shapely.geometry import mapping
import simplekml

class Station(): 
    def __init__(self, name, file_path):
        self.name = name
        self.path = file_path
        self.lat = None
        self.lon = None
        self.alt = None
        self.data = None
        self.rawGPM = None  # To contain the GPM raw data of the location
        self.gwrGPM = None  # To contain the GPM geographically weighted regression
        self.multiGPM = None # To contain the GPM multilinear regression
        self.rain4pe = None # To contain the rain4pe data
        self.PISCO = None # To contain PISCO data per pixel
        
        self.__station_location()
        self.__station_data()
        
    "Funtions for Precipitation Stations dataset from ANA"
    def __station_location(self):
        station=pd.read_excel(self.path+'/'+self.name+'.xlsx')
        station.columns=range(station.shape[1])
        geo_features=station[2].iloc[6]       #WGS84
        self.lat = geo_features[10:19] #Latitud in degrees
        self.lon = geo_features[32:41] #Longitud in degrees
        self.alt = geo_features[60:] #Altitude in m.s.n.m
        return 

    "Function to get the data itself in mm daily"
    def __station_data(self):# Only for ANDREA-ANA format
        station_name=pd.read_excel(self.path+'/'+self.name+'.xlsx').replace(os.sep, '/')
        station_name=station_name.drop(station_name.index[:12])
        station_name.columns=station_name.iloc[0]
        station_name=station_name.drop(station_name.index[0])
        station_name=station_name.reset_index(drop=True)
        station_name=station_name.drop(station_name.index[-1])
        melted_df = pd.melt(station_name, id_vars=['Año', 'Día'], var_name='Month', value_name='Precipitation')
        month_mapping = {
            'Ene': '01',
            'Feb': '02',
            'Mar': '03',
            'Abr': '04',
            'May': '05',
            'Jun': '06',
            'Jul': '07',
            'Ago': '08',
            'Set': '09',
            'Oct': '10',
            'Nov': '11',
            'Dic': '12'
        }
        melted_df['Month'] = melted_df['Month'].map(month_mapping)
        # melted_df['Date'] = pd.to_datetime(melted_df['Año'].astype(str) + '-' + melted_df['Month'] + '-' + melted_df['Día'].astype(str), format='%Y-%m-%d', errors='coerce')
        melted_df['Date'] = pd.to_datetime(melted_df['Año'].astype(str) + '-' + melted_df['Month'] + '-' + melted_df['Día'].astype(str), format='ISO8601', errors='coerce')
        melted_df = melted_df.dropna(subset=['Date'])
        result_df = melted_df.set_index('Date')[['Precipitation']]
        result_df.sort_index(inplace=True)
        self.data = result_df
        return
    
    def plot_ana(self):
    # To plot Ana data timeseries
        plt.figure(figsize=(10, 6))
        plt.plot(self.data.index, self.data['Precipitation'], label='Precipitation (mm)')
        plt.title("Data available from " + str(self.data.index[0])+' to '
                    + str(self.data.index[-1])+'\n'+ self.name+' Station' ,fontweight="bold")    
        plt.xlabel("Time")
        plt.ylabel("Precipitation (mm)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def _point_pp_extr(self, dataset_path, variable, time=None, pointname=None):
        """The dataset will be charged in the memory as an xarray"""
        dataset = xr.open_dataset(dataset_path)
        # Extracting data
        y_variable = dataset[variable].sel(lat=self.lat, lon=self.lon, method='nearest').values.flatten()
        if time is not None:
            x_variable = dataset[time].values.astype('datetime64[D]')
            # Plotting with datetime x-axis
            plt.plot(x_variable, y_variable, label=variable, color='blue', linestyle='-')
            # Customize plot appearance
            plt.xlabel('Date')
            plt.ylabel(variable)
            plt.title(f'{variable} values at {pointname} ({self.lat}, {self.lon})')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            # Add legend
            plt.legend()
        else:
            # Plotting without datetime x-axis
            plt.plot(y_variable, label=variable, color='blue', linestyle='-')
            # Customize plot appearance
            plt.xlabel('Data Index')
            plt.ylabel(variable)
            plt.title(f'{variable} values at {pointname} ({self.lat}, {self.lon})')
            plt.grid(True)
            plt.tight_layout()
            plt.legend()            # Add legend

        # Convert data to DataFrame for return
        if time is not None:
            dataframe_result = pd.DataFrame({'Date': pd.to_datetime(x_variable), variable: y_variable})
            dataframe_result.set_index('Date', inplace=True)
        else:
            dataframe_result = pd.DataFrame({variable: y_variable}, columns=[variable])

        return dataframe_result
    
        #Funtion to calculate the PBIAS,MAE and RMSE in a dataframe using 2 columns
        
    def _performance(self, obs_attr: str, sim_attr: str, point_name: str):
        """
        Calculates PBIAS, MAE, and RMSE between simulated and observed data.
    
        Args:
            obs_attr (str): Attribute name for observed data.
            sim_attr (str): Attribute name for simulated data.
            point_name (str): Name of the point/station being analyzed.
    
        Returns:
            tuple: PBIAS, MAE, RMSE values as floats.
        """
        # Fetch attributes dynamically
        sim_data = getattr(self, sim_attr)
        obs_data = getattr(self, obs_attr)
    
        # Ensure the data has valid entries
        mask = obs_data.notna()
        sim_data = sim_data[mask]
        obs_data = obs_data[mask]
    
        # PBIAS Calculation
        numerator = (sim_data - obs_data).sum()
        denominator = obs_data.sum()
        pbias = (numerator / denominator) * 100
        pbias = pbias.item() if isinstance(pbias, pd.Series) else float(pbias)  # Ensure float type
    
        # MAE Calculation
        mae = (abs(sim_data - obs_data)).mean()
        mae = mae.item() if isinstance(mae, pd.Series) else float(mae)  # Ensure float type
    
        # RMSE Calculation
        squared_diff = (sim_data - obs_data) ** 2
        rmse = np.sqrt(squared_diff.mean().iloc[0])   # Explicitly convert mean to float
        rmse = float(rmse)
        # Print the results
        print(f"PBIAS is {pbias:.2f} in {point_name}")
        print(f"MAE is {mae:.2f} in {point_name}")
        print(f"RMSE is {rmse:.2f} in {point_name}")
    
        return pbias, mae, rmse

        
""" 
Functions to retrive some imformation from dataframes, which are contained
in the .data of the dummy
"""
    
def dummy_comparison(data1, data2, parameter_name = 'Precipitation'):
    df1= data1.data.copy()
    df2= data2.data.copy()
    df1.index = pd.to_datetime(df1.index)
    df2.index = pd.to_datetime(df2.index)    
    plt.figure(figsize=(10, 5))
    # Plot both DataFrames on the same y-axis
    plt.plot(df1.index, df1[parameter_name], label=data1.name, linestyle='--', color='blue')
    plt.plot(df2.index, df2[parameter_name], label=data2.name, linestyle='-.', color='green')
    plt.xlabel('Time')
    plt.ylabel(f'{parameter_name} in (mm)')
    plt.title(f'Comparison of:  {data1.name} vs {data2.name}')
    plt.legend()
    plt.grid(True)
    plt.show()      
    
def station_monthly_sum(dataframe_ini):
    dataframe = dataframe_ini.copy()
    dataframe.index = pd.to_datetime(dataframe.index)
    dataframe['Year'] = dataframe.index.year
    dataframe['Month'] = dataframe.index.month
    dataframe['Year-Month'] = dataframe['Year'].astype(str) + '-' + dataframe['Month'].astype(str).str.zfill(2)
    monthly_sum = dataframe.groupby('Year-Month')['Precipitation'].sum()
    monthly_sum = monthly_sum.reset_index()
    monthly_sum['Year-Month'] = pd.to_datetime(monthly_sum['Year-Month'], format='%Y-%m')
    monthly_sum = monthly_sum.set_index('Year-Month')
    monthly_sum.index = monthly_sum.index.date
    return monthly_sum

def station_yearly_sum(dataframe_init):
    dataframe = dataframe_init.copy()
    dataframe.index = pd.to_datetime(dataframe.index)
    dataframe['Year'] = dataframe.index.year
    yearly_sum = dataframe.groupby('Year')['Precipitation'].sum()
    yearly_sum = yearly_sum.reset_index()
    yearly_sum = yearly_sum.set_index('Year')
    yearly_sum.index = pd.to_datetime(yearly_sum.index.astype(str) + '-12-31')
    yearly_sum.index = yearly_sum.index.date
    return yearly_sum

def month_mean_oyears(dataframe_init,parameter): #Dataframe index(time format),Parameter should be the name of the column
    dataframe = dataframe_init.copy()    
    dataframe.index = pd.to_datetime(dataframe.index)
    dataframe['year'] = dataframe.index.year
    dataframe['month'] = dataframe.index.month
    mean_per_month_across_years = dataframe.groupby('month')[parameter].mean().reset_index()    
    return mean_per_month_across_years 
   
def initial_charge(data_directory):
    # #Append the name of all the stations int the list
    station_list = []
    for filename in os.listdir(data_directory):
        if filename.endswith(".xlsx"):
            station_name = os.path.splitext(filename)[0]  # Remove the extension
            station_list.append(station_name)
    return station_list   
    
def dic_to_KML(dictionary_name,save_path): #Exporting dictionary to KML file
    # Initialize the KML object
    kml = simplekml.Kml()
    
    # Add each location as a point in the KML file
    for key,data in dictionary_name.items():
        # Extract latitude, longitude, and altitude
        lat = data.lat
        lon = data.lon
        alt = data.alt
        name=data.name
        # Add to KML
        pnt = kml.newpoint(name=name)
        pnt.coords = [(lon, lat, alt)]
        pnt.altitudemode = simplekml.AltitudeMode.absolute  # Use absolute altitude
        pnt.description = f"{name}: Altitude = {alt} meters"
    
    # Save the KML file
    kml.save(save_path) 

def core_station_data(folder_path = str):
    station_list = initial_charge(data_directory)
    # Creation of a dictionary that contains Station objects with initially metadata from Ana
    dummy_dict = {}
    for i in station_list:  # Look for the images
        dummy_dict[i] = Station(i,data_directory)

    # Save entire dictionary to a kml file
    save_path=r"C:\Users\jvila\Desktop\Andean_project\final_stations_locations.kml"
    dic_to_KML(dummy_dict, save_path)  
    return dummy_dict

def dict_data_filtered(data_dict: dict, years: int = 10):
    """
    Filters data along GPM range based on a given number of years.

    Args:
        data_dict (dict): Dictionary containing station data with timestamps.
        years (int): Number of years to calculate the GPM range.
        start_date (str): Starting date for GPM filtering.

    Returns:
        list: Filtered keys that meet the GPM range criteria.
    GPM data starts from '2000-01-01'

    """
    # Function to calculate the date key
    def date_key_forgpm( years: int) -> str:
        start_gpm_date = datetime.strptime('2000-01-01', '%Y-%m-%d')
        final_date = start_gpm_date.replace(year=start_gpm_date.year + years)
        return final_date.strftime('%Y-%m-%d')
    
    # Calculate the final date based on the given years
    final_date = date_key_forgpm(years)

    # Function to filter keys based on timestamp
    def filter_keys(data_dict: dict, timestamp_min=final_date) -> list:
        filtered_keys = [
            key for key, df in data_dict.items()
            if (df.data.index > pd.Timestamp(timestamp_min)).any()
        ]
        print(f"Number of stations within the range > {timestamp_min} are: {len(filtered_keys)}")
        return filtered_keys

    # Apply filtering and return the results
    filter_keys = filter_keys(data_dict)        
    
    # Create a new dictionary according to the filtered keys selected 
    final_filtered_dict = {
        key: data_dict[key] for key in filter_keys  if key in data_dict}
    
    return final_filtered_dict
# =============================================================================
# Command execution
# =============================================================================
data_directory=os.path.join(r'C:\Users\jvila\Desktop\Andean_project\data').replace(os.sep, '/')

stations = core_station_data(data_directory) #Set the folder where the ANA data is
final_data = dict_data_filtered(stations, 18) # Set the #years to filter after 2000
        
# Save selected stations to a kml file
save_path=r"C:\Users\jvila\Desktop\Andean_project\selected_stations_locations.kml"
dic_to_KML(final_data, save_path)         
        
# =============================================================================
# Play ground
dummy1 = final_data['Crisnejas_ San Marcos']
a=dummy1.rain4pe = dummy1.data+10
pbias, mae, rmse = dummy1._performance('data','rain4pe','tryout')


# class Station_stads():
#     def __init__(self, name, pbias, mae, rmse):
#         self.name = name
#         self.data = data
#         self.pbias = None
#         self.mae = None
#         self.rmse = None

# dummy_dict[station_list[0]].plot_ana() # to plot all the data
# station_monthly_sum(dummy_dict[station_list[0]].data) # to sum monthly data
# dummy_comparison(dummy_dict[station_list[0]],dummy_dict[station_list[1]]) # to compare dataframes

# for name in station_list: #To plot the yearly sum for each station
#     plt.figure()
#     plt.plot(station_yearly_sum(dummy_dict[name].data), label=f"{name}")
#     plt.ylabel('Precipitation (mm)')
#     plt.legend()
#     plt.title(f"Precipitation for {name}")
#     plt.show()
# =============================================================================
