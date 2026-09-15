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
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import simplekml

# import geopandas as gpd
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
# import pysal
# import rioxarray
# from shapely.geometry import mapping

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
        self.exprGPM = None  # To contain the GPM exponential regression
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

    def _point_pp_extr(self, dataset, variable, attribute_name: str, time=None):
        """
        Extracts data from a dataset and dynamically stores it in the class instance.
    
        Args:
            dataset (-): Dataset (NetCDF or similar format).
            variable (str): Name of the variable to extract.
            time (str, optional): Name of the time dimension variable. Defaults to None.
            attribute_name (str): Attribute name to store the extracted data in the class.
    
        Returns:
            pd.DataFrame: The extracted data as a DataFrame.
        """
        # Copy dataset using xarray
        dataset = dataset.copy()
    
        # Extract data for the given variable at the nearest lat/lon
        y_variable = dataset[variable].sel(lat=self.lat, lon=self.lon, method='nearest').values.flatten()
    
        # Extract time variable if provided
        if time is not None:
            x_variable = dataset[time].values.astype('datetime64[D]')
            dataframe_result = pd.DataFrame({'Date': pd.to_datetime(x_variable), variable: y_variable})
            dataframe_result.set_index('Date', inplace=True)
        else:
            dataframe_result = pd.DataFrame({variable: y_variable}, columns=[variable])
    
        # Dynamically set the attribute in the class
        setattr(self, attribute_name, dataframe_result)
        print(f"Data extracted for {self.name}")
        return dataframe_result
    
    def plot_pisco(self, variable: str, pointname=None):
        """
        Plots data stored in a class attribute.
    
        Args:
            attribute_name (str): Name of the class attribute containing the data to plot.
            variable (str): Name of the variable to display on the plot.
            pointname (str, optional): Name of the point for labeling. Defaults to None.
        """
        # Get the data from the class attribute
        data = getattr(self, 'PISCO')
    
        # Plot the data
        if 'Date' in data.columns:
            # Plot with datetime x-axis
            plt.plot(data.index, data[variable], label=variable, color='blue', linestyle='-')
            plt.xlabel('Date')
        else:
            # Plot without datetime x-axis
            plt.plot(data[variable], label=variable, color='blue', linestyle='-')
            plt.xlabel('Data Index')
    
        plt.ylabel(variable)
        plt.title(f'{variable} values at {pointname or "Unknown Point"}')
        plt.grid(True)
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()    
    #Function to cut all the datasets according to a timerange
    def time_selection(self, start_date, end_date):
        # List of parameters to process
        parameters = ['data', 'gwrGPM', 'rawGPM', 'multiGPM', 'rain4pe', 'expGPM', 'PISCO']
        
        for parameter in parameters:
            try:
                if parameter == 'data':  # Process observed data
                    obs_data = getattr(self, parameter)  # Access the attribute
                    cut_data = obs_data.copy()
                    cut_data.index = pd.to_datetime(obs_data.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
                    cut_data = cut_data.loc[start_date:end_date, 'Precipitation']

                    # Replace the original attribute with the updated data
                    setattr(self, parameter, cut_data)

                else:  # Process other parameters (simulated data)
                    sim_data = getattr(self, parameter)  # Access the attribute
                    cut_data = sim_data.copy()
                    cut_data.index = pd.to_datetime(sim_data.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
                    cut_data = cut_data.loc[start_date:end_date, 'precipitationCal']

                    # Replace the original attribute with the updated data
                    setattr(self, parameter, cut_data)
            
            except AttributeError:
                # If the parameter attribute is not found, skip
                print(f"Attribute for {parameter} not found, skipping...")

    #Make a hard copy of the class
    def copy(self):
        # Create a new instance of the class, passing the required arguments
        new_instance = Station(self.name, self.path)
        
        # Copy all attributes from the current instance to the new one
        for attr, value in self.__dict__.items():
            if isinstance(value, pd.DataFrame):  # Make a deep copy for DataFrames
                new_instance.__dict__[attr] = value.copy()
            else:  # Shallow copy for other types
                new_instance.__dict__[attr] = value
        
        return new_instance
    
    #Funtion to calculate the PBIAS,MAE and RMSE in a dataframe using 2 columns
    def _performance(self, obs_attr: str, sim_attr: str, start_date: str, end_date: str):
        """
        Calculates PBIAS, MAE, RMSE, R, and KGE between simulated and observed data,
        along with count of zeros, min/max values, and zero_match indicator.
    
        Args:
            obs_attr (str): Attribute name for observed data.
            sim_attr (str): Attribute name for simulated data.
            start_date (str): Start date of the period.
            end_date (str): End date of the period.
    
        Returns:
            tuple: PBIAS, MAE, RMSE, R, KGE, zero_sim, min_sim, max_sim, zero_match
        """
        import numpy as np
        import pandas as pd
    
        obs_data = getattr(self, obs_attr).copy()
        obs_data.index = pd.to_datetime(obs_data.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        obs_data = obs_data.loc[start_date:end_date, 'Precipitation']
    
        sim_data = getattr(self, sim_attr).copy()
        sim_data.index = pd.to_datetime(sim_data.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        sim_data = sim_data.loc[start_date:end_date, 'precipitationCal']
    
        print(f'Data was homogenized to the same time range {start_date}:{end_date}')
        
        point_name = self.name
    
        # Filter valid values
        mask = obs_data.notna() & sim_data.notna()
        obs_data = obs_data[mask]
        sim_data = sim_data[mask]
    
        # PBIAS
        pbias = ((sim_data - obs_data).sum() / obs_data.sum()) * 100
        pbias = float(pbias)
    
        # MAE
        mae = abs(sim_data - obs_data).mean()
        mae = float(mae)
    
        # RMSE
        rmse = np.sqrt(((sim_data - obs_data) ** 2).mean())
        rmse = float(rmse)
    
        # Pearson R
        r = sim_data.corr(obs_data)
        r = float(r)
    
        # KGE
        mean_sim = sim_data.mean()
        mean_obs = obs_data.mean()
        std_sim = sim_data.std()
        std_obs = obs_data.std()
    
        beta = mean_sim / mean_obs
        gamma = (std_sim / mean_sim) / (std_obs / mean_obs)
        kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
        kge = float(kge)
        
        threshold = 0.2  # Define near-zero threshold    
        # Additional metrics
        zero_obs = int((obs_data == 0).sum())
        zero_sim = int((sim_data == 0).sum())
        close_zero = int(((sim_data > 0) & (sim_data < threshold)).sum())
    
        min_obs = float(obs_data.min())
        max_obs = float(obs_data.max())
        min_sim = float(sim_data.min())
        max_sim = float(sim_data.max())
    
        # Zero match: where sim_data == 0 and obs_data is near-zero
        nonzero_match_mask = (obs_data > 0) & (sim_data > 0) & (sim_data < threshold)
        min_sim = int(nonzero_match_mask.sum())
        numbdata1 = float(obs_data.count())
        numbdata2 = float(sim_data.count())
        # Print results
        print(f" number of data {numbdata1} and {numbdata2}")
        print(f"PBIAS is {pbias:.2f} in {point_name}")
        print(f"MAE is {mae:.2f} in {point_name}")
        print(f"RMSE is {rmse:.2f} in {point_name}")
        print(f"R is {r:.2f} in {point_name}")
        print(f"KGE is {kge:.2f} in {point_name}")
        print(f"Zero count - Observed: {zero_obs}, Simulated: {zero_sim}")
        print(f"Min/Max - Observed: ({min_obs}, {max_obs}), Simulated: ({min_sim}, {max_sim})")
        # print(f"Zero match count (sim=0 & obs≈0): {zero_match}")
    
        return pbias, mae, rmse, r, kge, zero_sim, min_sim, max_sim


    
    def _detection_capability(self, obs_attr: str, sim_attr: str, start_date: str, end_date: str, min_obs_threshold: float):
        """
        Calculates # FBI|Frecuency bias index, FAR|False Alarm ratio,
        POD|Probably of detection and accuracy between simulated and observed data.
    
        Args:
            obs_attr (str): Attribute name for observed data.
            sim_attr (str): Attribute name for simulated data.
            min_obs_threshold (float): Min precipitation analyzed in simulations.
    
        Returns:
            tuple: FBI, FAR, POD and Accuracy values as floats.
        """
        obs_data = getattr(self, obs_attr)
        obs_data = obs_data.copy()
        obs_data.index = pd.to_datetime(obs_data.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        obs_data = obs_data.loc[start_date:end_date, 'Precipitation']
        
        sim_data = getattr(self, sim_attr)
        sim_data = sim_data.copy()
        sim_data.index = pd.to_datetime(sim_data.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        sim_data = sim_data.loc[start_date:end_date, 'precipitationCal']
    
        hits = ((obs_data > 0) & (sim_data >= min_obs_threshold)).sum()    # Satellite rain and gauge rain
        false_alarms = ((obs_data == 0) & (sim_data >= min_obs_threshold)).sum()
        # misses = ((obs_data > 0) &  (sim_data == 0)).sum()
        misses = ((obs_data > 0) & (sim_data <= min_obs_threshold)).sum()
        correct_negatives = ((obs_data == 0) & (sim_data == 0)).sum()
        
        FBI = (hits + false_alarms) / (hits + misses)
        FAR = (false_alarms) / (hits + false_alarms)
        POD = (hits) / (hits + misses)
        accuracy = (hits + correct_negatives) / (hits + false_alarms + misses +correct_negatives)
        print(f"Stats are: FBI {FBI}, FAR:{FAR}, POD {POD}, Accuracy {accuracy} for {self.name}")
        return FBI, FAR, POD, accuracy

    def _extreme_indices(self, self_attr:str, start_date: str, end_date: str):
        """
        Calculates extreme indicators between simulated and observed data.
        The indicators are based on Expert Team on Climate Change Detection and Indices (ETCCDI)  http://etccdi.pacificclimate.org.
        
        ** This is a bit different because we are not calculating error or performance metrics
        ** We do not load observed and simulated everytime, they are independent
        ** Functions and classess later are created later to account for this
        
        Args:
            self_attr (str)   : Attribute name for source of data (e.g., "data","rawGPM")
            start_date (str) : Start date of the period.
            end_date (str)   : End date of the period.
            
        Calculated parameters:
        
            *Counter Indices*
            Consecutive dry days (CDD) : Number of consecutive days with precipitation < 1mm.
            Consecutive wet days (CWD) : Number of consecutive days with precipitation > 1mm.
            Number of heavy precipitation days (R10)      : Number of days with precipitation > 10mm.
            Number of very heavy precipitation days (R20) : Number of days with precipitation > 20mm.
            
            *Percentile Indices*
            Very wet days (R95p)      : Annual total precipitation when daily precipitation > 95th percentile.    
            Extremely wet days (R99p) : Annual total precipitation when daily precipitation > 99th percentile.
            
            
        returns:
            Dictionary with the calculated indices per year
            
        """
        
        # Reading data for given source

        data = getattr(self, self_attr)  # or sim_attr, since you’re now using just one
        data = data.copy()
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        if self_attr == "data":
            data = data.loc[start_date:end_date, 'Precipitation']
        else:
            data = data.loc[start_date:end_date, 'precipitationCal']
        
        ext_ind_dict = {}

        # ------ Calculating Counter Indices
        years = data.index.year.unique()
        ext_ind_dict['cwd'] = pd.Series({
            year: (group := data[data.index.year == year])  
                .pipe(lambda x: (x > 1).astype(int)) #Count wet days
                .groupby((group <= 1).cumsum())      #Stop count when dry day is found
                .sum()                               #Sum counts
                .max(skipna=True)                    #Get max count of wet days  for the year -->cwd
            for year in years
        })
        
        ext_ind_dict['cdd'] = pd.Series({
            year: (group := data[data.index.year == year])  
                .pipe(lambda x: (x < 1).astype(int)) #Count dry days
                .groupby((group >= 1).cumsum())      #Stop count when wet day is found
                .sum()                               #Sum counts
                .max(skipna=True)                    #Get max count of dry days for the year -->cdd   
            for year in years
        })
        
        ext_ind_dict['r10'] = data.groupby(data.index.year).apply(lambda x: (x > 10).sum())
        ext_ind_dict['r20'] = data.groupby(data.index.year).apply(lambda x: (x > 20).sum())

        # ------ Calculating Percentile Indices
        ext_ind_dict['r95p'] = data.groupby(data.index.year).apply(lambda x: x[x > data.quantile(0.95)].sum())
        ext_ind_dict['r99p'] = data.groupby(data.index.year).apply(lambda x: x[x > data.quantile(0.99)].sum())
        
        return ext_ind_dict
# =============================================================================
        
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
        print(f'ANA precipitation loaded for station {i}')
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
if __name__ == '__main__':
    """
        I am setting this convention in the DB, to be considered for the data filling
        self.data = None, Data from ANA (Autoridad Nacional del Agua)
        self.rawGPM = None, Data from GPM raw data of the location
        self.gwrGPM = None, Data from GPM geographically weighted regression
        self.multiGPM = None, Data from GPM multilinear regression
        self.expGPM = None, Data from GPM exponential regression
        self.rain4pe = None, Data from rain4pe data
        self.PISCO = None, Data from PISCO data 
    """
    data_directory=os.path.join(r'C:\Users\jvila\Desktop\Andean_project\data').replace(os.sep, '/')
    
    stations = core_station_data(data_directory) #Set the folder where the ANA data is
    final_data = dict_data_filtered(stations, 18) # Set the #years to filter after 2000
        
# Save selected stations to a kml file, Optional
# save_path=r"C:\Users\jvila\Desktop\Andean_project\selected_stations_locations.kml"
# dic_to_KML(final_data, save_path)

# import pickle
# #Save the data selected for the study to not re-process it everytime
# save_path = r"C:\Users\jvila\Desktop\Andean_project\final_data.pkl"

# with open(save_path, 'wb') as file:
#     pickle.dump(final_data, file)
# print(f"Dictionary saved successfully to {save_path}!")
