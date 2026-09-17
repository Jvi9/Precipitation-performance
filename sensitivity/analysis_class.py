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
import seaborn as sns

# =============================================================================
"""Posible solution for the storage of all the statistics
    Obiously all the data are being compared to the observations""" 
      
class productStats():
    def __init__(self, station_name, mae, pbias, rmse,r, kge, zero_sim, min_sim, max_sim, fbi, far, pod, acc, lat, lon, alt):
        self.name = station_name
        self.mae = mae
        self.pbias = pbias
        self.rmse = rmse
        self.r = r
        self.kge = kge
        self.zero_sim = zero_sim
        self.min_sim = min_sim
        self.max_sim = max_sim
        self.fbi = fbi
        self.far = far
        self.pod = pod
        self.acc = acc
        self.lat = lat
        self.lon = lon
        self.alt = alt

class extremeIndices():
    def __init__(self, station_name, cdd, cwd, r10, r20, r95p, r99p):
        '''
        Type of indices is Pandas Series
        '''
        self.name = station_name
        self.cdd = cdd
        self.cwd = cwd
        self.r10 = r10
        self.r20 = r20
        self.r95p = r95p
        self.r99p = r99p



        
def set_pandas_time(dictionary_of_stations):
    for key, station in dictionary_of_stations.items():  # Loop through each Station instance
        # Convert all relevant DataFrame indexes to pandas datetime
        station.rawGPM.index = pd.to_datetime(station.rawGPM.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        station.gwrGPM.index = pd.to_datetime(station.gwrGPM.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        station.PISCO.index = pd.to_datetime(station.PISCO.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        station.rain4pe.index = pd.to_datetime(station.rain4pe.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
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
        pbias, mae, rmse, r, kge, zero_sim, min_sim, max_sim = dictionary[key]._performance(obs_attr, sim_attr, start_date, end_date)
        fbi, far, pod, acc  = dictionary[key]._detection_capability(obs_attr, sim_attr, start_date, end_date, min_obs_threshold)
        lat, lon, alt = dictionary[key].lat, dictionary[key].lon, float(dictionary[key].alt)
        dummy_dictionary[key] = object_creating(station_name, mae, pbias, rmse, r, kge, zero_sim, min_sim, max_sim, fbi, far, pod, acc, lat, lon, alt)
        
    return dummy_dictionary

def raise_extremeIndices(object_creating, dictionary:dict, self_attr:str, start_date: str, end_date: str, wet_threshold: float = 1.0):
    """
    Similar to raise_stats, but for extreme indices
    Not using raise_stats because here we run analysis independently (obs and sim are not loaded everytime)

    CHANGED: added wet_threshold (default 1.0, so old call sites that don't pass it
    keep the original behaviour). Pass this through to Station._extreme_indices so
    CDD/CWD use the SAME threshold as the FBI/FAR/POD/ACC detection stats
    (i.e. Generator.min_threshold) - this is the fix for the incoherent-thresholds
    issue raised by the editor/reviewers.
    
    Returns:
        Dictionary with extremeIndices instance that corresponds to each station
    """
    station_dictionary = {}
    
    for key in dictionary.keys():
        ExtInd = dictionary[key]._extreme_indices(self_attr, start_date, end_date, wet_threshold) #This is also a dictionary
        station_dictionary[key] = object_creating(key,ExtInd["cdd"],ExtInd["cwd"],ExtInd["r10"],ExtInd["r20"],ExtInd["r95p"],ExtInd["r99p"])
        
    return station_dictionary


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
        for metric in ['mae', 'pbias', 'rmse', 'r', 'kge', 'zero_sim', 'min_sim', 'max_sim', 'fbi', 'far', 'pod', 'acc','alt']  # Add more metrics here as needed
    }, index=list(dictionary.keys()))
    
    return combined_summary

def station_summary(dictionary: dict, metrics: list = None):
    """If not metric is provided all the parameters are being calculated
    Default metrics if none are provided
    =['mae', 'pbias', 'rmse', 'fbi', 'far', 'pod', 'acc']"""

    if metrics is None:
        metrics = ['mae', 'pbias', 'rmse', 'r', 'kge', 'zero_sim', 'min_sim', 'max_sim', 'fbi', 'far', 'pod', 'acc']
    # Dynamically create a DataFrame for all specified metrics
    
    summary = pd.DataFrame({
        metric: [getattr(dictionary, metric)]
        for metric in metrics
    }, index=[dictionary.name])
    return summary

def metrics_over_analysis(station_name:str, list_of_dictionaries: list = None):
    station_over_analysis = {}
    if list_of_dictionaries is None:
        list_of_dictionaries = ['statsrawGPM_dict','statsgwrGPM_dict','statsPISCO_dict','statsrain4pe_dict','statsexpGPM_dict']
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
        metrics = ['mae', 'pbias', 'rmse', 'r', 'kge', 'zero_sim', 'min_sim', 'max_sim', 'fbi', 'far', 'pod', 'acc']
    
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

class Generator():
    def __init__(self, load_data_path:str, min_range:str, max_range:str, min_threshold:float)->None:
        """
        min_threshold is used for BOTH the detection-capability stats
        (FBI/FAR/POD/ACC, via _run_scenarios) AND the CDD/CWD ETCCDI indices
        (via _run_extremeIndices) - i.e. this is now your single unified
        threshold across both analyses. To run the 1 vs 2.5 mm/day
        sensitivity test, instantiate this class twice with min_threshold=1.0
        and min_threshold=2.5 and compare the resulting stats/indices.
        """
        self.load_data_path = load_data_path
        self.min_threshold = min_threshold
        self.min_range = min_range
        self.max_range = max_range
        
        self._load_data()
        self._set_pandas_time()
        self._run_scenarios()
        self._run_extremeIndices()
        print("Generator generated successfully!")
    def _load_data(self):
        # Load the dictionary from the specified path as final_data
        with open(self.load_data_path, 'rb') as file:
            final_data = pickle.load(file) 
            self.final_data = final_data
        print(f"Dictionary loaded successfully from {self.load_data_path}!")
    
    def _set_pandas_time(self):
        for key, station in self.final_data.items():  # Loop through each Station instance
            # Convert all relevant DataFrame indexes to pandas datetime
            station.rawGPM.index = pd.to_datetime(station.rawGPM.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
            station.gwrGPM.index = pd.to_datetime(station.gwrGPM.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
            station.PISCO.index = pd.to_datetime(station.PISCO.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
            station.rain4pe.index = pd.to_datetime(station.rain4pe.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
            station.data.index = pd.to_datetime(station.data.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
            station.expGPM.index = pd.to_datetime(station.expGPM.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        print("All datetime indexes migrated to pandas datetime format for all stations.")
     
    def _run_scenarios(self):
        self.statsrawGPM_dict  = raise_stats(productStats, self.final_data, 'data', 'rawGPM', self.min_range, self.max_range, min_obs_threshold=self.min_threshold)
        self.statsgwrGPM_dict  = raise_stats(productStats, self.final_data, 'data', 'gwrGPM', self.min_range, self.max_range, min_obs_threshold=self.min_threshold)
        self.statsPISCO_dict   = raise_stats(productStats, self.final_data, 'data', 'PISCO', self.min_range, self.max_range, min_obs_threshold=self.min_threshold)
        self.statsrain4pe_dict = raise_stats(productStats, self.final_data, 'data', 'rain4pe', self.min_range, self.max_range, min_obs_threshold=self.min_threshold)
        self.statsexpGPM_dict  = raise_stats(productStats, self.final_data, 'data', 'expGPM', self.min_range, self.max_range, min_obs_threshold=self.min_threshold)
        
    def _run_extremeIndices(self):
        # CHANGED: wet_threshold=self.min_threshold added to every call below so CDD/CWD
        # use the same threshold as the detection-capability stats above.
        self.extremeIndicesrawGPM     = raise_extremeIndices(extremeIndices, self.final_data,'rawGPM' , self.min_range, self.max_range, wet_threshold=self.min_threshold)
        self.extremeIndicesgwrGPM     = raise_extremeIndices(extremeIndices, self.final_data,'gwrGPM' , self.min_range, self.max_range, wet_threshold=self.min_threshold)
        self.extremeIndicesPISCO      = raise_extremeIndices(extremeIndices, self.final_data,'PISCO'  , self.min_range, self.max_range, wet_threshold=self.min_threshold)
        self.extremeIndicesrain4pe    = raise_extremeIndices(extremeIndices, self.final_data,'rain4pe', self.min_range, self.max_range, wet_threshold=self.min_threshold)
        self.extremeIndicesexpGPM     = raise_extremeIndices(extremeIndices, self.final_data,'expGPM' , self.min_range, self.max_range, wet_threshold=self.min_threshold)
        self.extremeIndicesObserved   = raise_extremeIndices(extremeIndices, self.final_data,'data'   , self.min_range, self.max_range, wet_threshold=self.min_threshold)
        
     
    def plot_probability_graph(self, pod_columnx='pod', far_columny='far'):
        """
        Plots multiple datasets on the same graph, differentiating them by color and legend.
        
        Parameters:
        - data_dict: A dictionary where keys are dataset labels and values are DataFrames containing the data.
        - pod_column: Name of the column representing POD values.
        - far_column: Name of the column representing FAR values.
        """
        data_dict = {
            'Dataset 1: statsPISCO': parameters_report(self.statsPISCO_dict),
            'Dataset 2: statsgwrGPM': parameters_report(self.statsgwrGPM_dict),
            'Dataset 3: statsexpGPM': parameters_report(self.statsexpGPM_dict),
            'Dataset 4: statsrawGPM': parameters_report(self.statsrawGPM_dict),
            'Dataset 5: statsrain4pe': parameters_report(self.statsrain4pe_dict)
            }
        
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
        plt.title(f'Detection capabilities in all stations {self.min_threshold} threshold')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_violin_precipitation(self):
        print("Preparing data for violin plot...")
    
        # Create a list to hold all the data
        all_data = []
    
        # Loop through each station
        for key, station in self.final_data.items():
            # For each dataset, get daily precipitation and add to the list
            for dataset_name in ['data','rawGPM', 'gwrGPM', 'PISCO', 'rain4pe', 'expGPM']:
                dataset = getattr(station, dataset_name)
                daily_precip = dataset.resample('D').sum()  # Resample to daily totals
                for date, value in daily_precip.iterrows():
                    all_data.append({
                        'Station': key,
                        'Date': date,
                        'Precipitation': value.values[0],  # Assuming single column
                        'Dataset': dataset_name
                    })
    
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
    
        # Drop NaNs just in case
        df.dropna(subset=['Precipitation'], inplace=True)
    
        print("Plotting violin graph...")
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='Dataset', y='Precipitation', data=df, inner='box', palette='Set2')
        plt.title('Daily Precipitation Distribution by Dataset')
        plt.ylabel('Precipitation (mm)')
        plt.xlabel('Dataset')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_violin_stats(self, stat_name='kge'):
        print(f"Preparing violin plot for statistic: {stat_name}")
    
        # Create a list to hold all the data
        all_stats = []
    
        # Define your datasets and their corresponding stats dictionaries
        dataset_stats = {
            'rawGPM': self.statsrawGPM_dict,
            'gwrGPM': self.statsgwrGPM_dict,
            'PISCO': self.statsPISCO_dict,
            'rain4pe': self.statsrain4pe_dict,
            'expGPM': self.statsexpGPM_dict
        }
    
        # Loop through each dataset and each station
        for dataset_name, stats_dict in dataset_stats.items():
            for station_name, stats_obj in stats_dict.items():
                # Get the desired stat from the stats object
                value = getattr(stats_obj, stat_name, None)
                if value is not None:
                    all_stats.append({
                        'Station': station_name,
                        'Dataset': dataset_name,
                        'Statistic': float(value)
                    })
    
        # Convert to DataFrame
        df = pd.DataFrame(all_stats)
    
        print("Plotting violin graph...")
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Dataset', y='Statistic', data=df, inner='box', palette='Set3')
        plt.title(f'Distribution of {stat_name.upper()} Across Datasets')
        plt.ylabel(stat_name.upper())
        plt.xlabel('Dataset')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def metrics_over_analysis(self, station_name: str, list_of_dictionaries: list = None):
        station_over_analysis = {}
        
        # Default to specific class attributes if no list is provided
        if list_of_dictionaries is None:
            list_of_dictionaries = [
                'statsrawGPM_dict',
                'statsgwrGPM_dict',
                'statsPISCO_dict',
                'statsrain4pe_dict',
                'statsexpGPM_dict'
            ]
        
        for item in list_of_dictionaries:
            # Access the dictionary object as an attribute of the class
            try:
                dictionary = getattr(self, item)  # Retrieve the dictionary attribute by name
                source = dictionary[station_name]  # Access the dictionary's value by station name
            except AttributeError:
                raise AttributeError(f"'{item}' is not a valid attribute of the class.")
            except KeyError:
                raise KeyError(f"Station name '{station_name}' not found in '{item}' dictionary.")
            except TypeError:
                raise TypeError(f"'{item}' is not a valid dictionary or does not contain the required station name.")
    
            # Generate the summary using the provided function
            summary = station_summary(source)
            station_over_analysis[item] = summary  # Store the summary in the result dictionary
    
        # Combine data into a single DataFrame
        data_joined = pd.concat(station_over_analysis, names=['Key'])
        data_joined.reset_index(level=0, inplace=True)
    
        return data_joined
    
    def geo_summary(self):
        rows = []
        # Loop through the data
        for key, other in self.final_data.copy().items():
            name = key
            lat = self.final_data[key].lat
            lon = self.final_data[key].lon
            alt = self.final_data[key].alt
            # Append a dictionary for each row
            rows.append({'station': name, 'lat': lat, 'lon': lon, 'alt': alt})

        # Create a DataFrame from the list of dictionaries
        geo_sum = pd.DataFrame(rows)
        
        self.geo_summary = geo_sum
        return self.geo_summary
    
    def join_stats(self, list_of_dictionaries: list = None, metrics: list = None) -> pd.DataFrame:
        """
        Joins DataFrames created from a list of class attributes (dictionaries) by filtering specific metrics.
        
        Args:
            list_of_dictionaries (list[str]): A list of attribute names referring to dictionaries in the class.
            metrics (list): A list of metric names to filter the columns.
                            Defaults to ['mae', 'pbias', 'rmse', 'r', 'kge', 'zero_sim', 'min_sim', 'max_sim', 'fbi', 'far', 'pod', 'acc'].
        
        Returns:
            pd.DataFrame: A combined DataFrame with renamed and filtered columns.
        """
        # Default to specific class attributes if no list is provided
        if list_of_dictionaries is None:
            list_of_dictionaries = [
                'statsrawGPM_dict',
                'statsgwrGPM_dict',
                'statsPISCO_dict',
                'statsrain4pe_dict',
                'statsexpGPM_dict'
            ]
            
        # Set default metrics if none are provided
        if metrics is None:
            metrics = ['mae', 'pbias', 'rmse', 'r', 'kge', 'zero_sim', 'min_sim', 'max_sim', 'fbi', 'far', 'pod', 'acc']
        
        di = pd.DataFrame()  # Initialize an empty DataFrame
        
        for item in list_of_dictionaries:
            # Access the dictionary attribute dynamically using getattr
            try:
                dictionary = getattr(self, item)  # Retrieve dictionary from class attributes
                df = parameters_report(dictionary)  # Generate DataFrame from the dictionary
            except AttributeError:
                raise AttributeError(f"'{item}' is not a valid attribute of the class.")
            except TypeError:
                raise TypeError(f"'{item}' is not a valid dictionary or does not contain the required data.")
            
            # Filter columns to only keep those in the metrics list
            df = df[df.columns.intersection(metrics)]
            
            # Extract a meaningful name for renaming columns
            name = key_dicname(item)
            df = df.rename(columns=lambda col: f"{col}_{name}")
            
            # Concatenate the current DataFrame with the master DataFrame
            di = pd.concat([di, df], axis=1)
        
        return di


class PrecipitationQualityChecker:
    def __init__(self, df):
        self.df = df.copy()
        self.df.index = pd.to_datetime(self.df.index)
        self.df["weekday"] = self.df.index.weekday
        self.add_prob_lluvia()

    def add_prob_lluvia(self):
        def get_prob(month):
            if month in [12, 1, 2, 3]:
                return 1.0
            elif month in [4, 5, 11]:
                return 0.5
            else:
                return 0.0
        self.df["prob_lluvia"] = self.df.index.month.map(get_prob)

    def apply_condition_1(self):
        cond1 = [0.0] * len(self.df)
        for i in range(len(self.df) - 2):
            if self.df["weekday"].iloc[i] == 4:  # viernes
                v = self.df["Precipitation"].iloc[i]
                s = self.df["Precipitation"].iloc[i + 1]
                d = self.df["Precipitation"].iloc[i + 2]

                if pd.notna(v) and v == s == d:
                    cond1[i] = 1.0
                elif pd.notna(s) and pd.notna(d) and s == 0 and d == 0:
                    cond1[i] = 0.5
                else:
                    cond1[i] = 0.0
        self.df["COND1"] = cond1

    def apply_condition_2(self):
        precip = self.df["Precipitation"].fillna(-9999)
        cond2 = [0.0] * len(self.df)
        for i in range(len(self.df)):
            val = precip.iloc[i]
            if val <= 0:
                continue
            if i >= 2 and val == precip.iloc[i - 1] == precip.iloc[i - 2] and val != 0:
                cond2[i] = 1.0
            elif i >= 1 and val == precip.iloc[i - 1] and val != 0:
                cond2[i] = 0.5
        self.df["COND2"] = cond2

    def apply_condition_3(self, threshold=5.0):
        def cond3_group(group):
            max_val = group["Precipitation"].max()
            if pd.isna(max_val) or max_val == 0:
                return pd.Series([0.0] * len(group), index=group.index)
            return group["Precipitation"].apply(
                lambda x: 1.0 if pd.notna(x) and abs(x - max_val) <= threshold and x != 0 else 0.0
            )
        self.df["COND3"] = self.df.groupby(self.df.index.to_period("M")).apply(cond3_group).reset_index(level=0, drop=True)

    def classify_flags(self):
        def classify(row):
            prob = row["prob_lluvia"]
            conds = [row["COND1"], row["COND2"], row["COND3"]]

            if prob == 1.0 and any(c == 1.0 for c in conds):
                return "Red rain"
            elif prob == 0.5 and any(c >= 0.5 for c in conds):
                return "Yellow rain"
            elif prob == 0.0 and all(c == 0.0 for c in conds):
                return "Blue rain"
            else:
                return "Blue rain"
        self.df["Flag"] = self.df.apply(classify, axis=1)

    def run_all_checks(self):
        self.apply_condition_1()
        self.apply_condition_2()
        self.apply_condition_3()
        self.classify_flags()

    def get_result(self):
        return self.df[["Precipitation", "prob_lluvia", "COND1", "COND2", "COND3", "Flag"]]
    
    # Inside PrecipitationQualityChecker class
    import matplotlib.pyplot as plt

    def plot_flag_summary(self):
        flag_counts = self.df["Flag"].value_counts().sort_index()
        
        # Print the counts
        print("Flag Summary:")
        print(flag_counts)
        
        # Plot
        plt.figure(figsize=(8, 5))
        flag_counts.plot(kind="bar", color=["red", "gold", "blue"])
        plt.title("Flag Distribution Summary")
        plt.xlabel("Flag Type")
        plt.ylabel("Number of Days")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

class PrecipitationBucket:
    def __init__(self, load_data_path:str, min_range:str, max_range:str)->None:
        self.load_data_path = load_data_path
        self.min_range = min_range
        self.max_range = max_range
        
        self._load_data()
        
    def _load_data(self):
        # Load the dictionary from the specified path as final_data
        with open(self.load_data_path, 'rb') as file:
            final_data = pickle.load(file) 
            self.final_data = final_data
        print(f"Dictionary loaded successfully from {self.load_data_path}!")
        
    def checker(self):
        self.results = {}
        for key, data in self.final_data.items():
            obs_data = data.data.copy()
            obs_data.index = pd.to_datetime(obs_data.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
            obs_data = obs_data.loc[self.min_range:self.max_range]
            checker = PrecipitationQualityChecker(obs_data)
            checker.run_all_checks()
            self.results[key] = checker.get_result()
    
    def rain_quality(self):
        summaries = []

        for key, df in self.results.items():
            flag_counts = df["Flag"].value_counts()
            summary_item = {
                "Key": key,
                "Red rain": flag_counts.get("Red rain", 0),
                "Yellow rain": flag_counts.get("Yellow rain", 0),
                "Blue rain": flag_counts.get("Blue rain", 0),
            }
            summaries.append(summary_item)

        data_summary = pd.DataFrame(summaries).set_index("Key")
        self.summary = data_summary

    def summary_graph(self):    
        # Assuming: self.summary is your DataFrame (as shown)
        df = self.summary.copy()
        
        # Plot settings
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Stacked bar plot
        df[['Blue rain', 'Yellow rain', 'Red rain']].plot(kind='bar', stacked=True, ax=ax, 
                                                          color=['blue', 'gold', 'red'])
        
        # Title and labels
        ax.set_title('Rain Flag Summary by Station')
        ax.set_ylabel('Days Count')
        ax.set_xlabel('Station')
        ax.legend(title='Rain Flag')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        plt.show()

    def normal_graph(self):
        # Copy and calculate row-wise percentages
        df_percent = self.summary.copy()
        df_percent = df_percent.div(df_percent.sum(axis=1), axis=0) * 100  # Convert to %
        df_percent = df_percent.sort_values(by='Red rain', ascending=False) #sorted

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        df_percent[['Blue rain', 'Yellow rain', 'Red rain']].plot(
            kind='bar', 
            stacked=True, 
            ax=ax, 
            color=['blue', 'gold', 'red']
        )
        
        # Title and labels
        ax.set_title('Percentage of Rain Flags by Station')
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Station')
        ax.legend(title='Rain Flag')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        plt.show()

    
def plot_flag_summary(dataflagged):
    flag_counts = dataflagged["Flag"].value_counts().sort_index()
    
    # Print the counts
    print("Flag Summary:")
    print(flag_counts)
    
    # Plot
    plt.figure(figsize=(8, 5))
    flag_counts.plot(kind="bar", color=["blue", "gold", "red"])
    plt.title("Flag Distribution Summary")
    plt.xlabel("Flag Type")
    plt.ylabel("Number of Days")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_precipitation_colored(dataflagged):
    # Ensure datetime index for plotting
    df = dataflagged.copy()
    df.index = pd.to_datetime(df.index)

    # Assign colors based on the Flag column
    color_map = {
        "Blue rain": "blue",
        "Yellow rain": "gold",
        "Red rain": "red"
    }
    colors = df["Flag"].map(color_map)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(df.index, df["Precipitation"].fillna(0), color=colors, width=1.0)
    plt.title("Daily Precipitation Colored by Flag")
    plt.xlabel("Date")
    plt.ylabel("Precipitation (mm)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    legend_handles = [
        plt.Line2D([0], [0], color="blue", lw=4, label="Blue rain"),
        plt.Line2D([0], [0], color="gold", lw=4, label="Yellow rain"),
        plt.Line2D([0], [0], color="red", lw=4, label="Red rain")
    ]
    plt.legend(handles=legend_handles)
    plt.show()