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

from analysis_class import *

# Create the Generator instance, that process the data, you can explore them under:
# .statsrawGPM_dict : like generator_test.statsgwrGPM_dict
# .statsgwrGPM_dict 
# .statsPISCO_dict 
# .statsrain4pe_dict
# .statsexpGPM_dict

load_data_path=r'C:\Users\jvila\Desktop\Andean_project/data_locked_loaded.pkl'
min_range = '2005-01-01'
max_range = '2018-12-31'
min_threshold = 1.0

generator_test = Generator(load_data_path,min_range,max_range, min_threshold)

# Columns can be changed to any stat: ['mae', 'pbias', 'rmse', 'r', 'kge', 'fbi', 'far', 'pod', 'acc']
generator_test.plot_probability_graph(pod_columnx='r', far_columny='kge')

# Generates a summary for station, you can specify or exclude a list of scenarios, by default:
# list_of_dict = ['statsrawGPM_dict','statsgwrGPM_dict','statsPISCO_dict','statsrain4pe_dict','statsexpGPM_dict']  
# .metrics_over_analysis(station_name , list_of_scenarios) 
station_test = generator_test.metrics_over_analysis('Crisnejas_ Sondor-Matara')
plot_1station_stats(station_test)

# Generates a temporal plot analysis over the scenarios: 'monthly', 'yearly' or 'both'
acumulate_comparison(generator_test.final_data, 'Mantaro_ Junin', 'monthly', '2005-01-01','2018-12-31')

# Sorts the stations in function of altitude in a dictionary of stations, it replaces the original sort
altitude_sort=sort_altitude(generator_test.final_data.keys(), generator_test.final_data)       

# Returns a df summary of all stations over the analysis, it can be filtered with lists
df = generator_test.join_stats(list_of_dictionaries = None, metrics = ['fbi', 'far', 'pod'])
# #Call a fbi, far, pod 'heat' map over multiple analysis
heat_map(df)