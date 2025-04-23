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
import matplotlib.pyplot as plt

# Create the Generator instance, that process the data, you can explore them under:
# .statsrawGPM_dict : like generator_test.statsgwrGPM_dict
# .statsgwrGPM_dict 
# .statsPISCO_dict 
# .statsrain4pe_dict
# .statsexpGPM_dict

load_data_path=r'C:\Users\jvila\Desktop\Andean_project\datasets/data_locked_loaded.pkl'
min_range = '2005-01-01'
max_range = '2018-12-31'
min_threshold = 1.0

generator_test = Generator(load_data_path,min_range,max_range, min_threshold)

# Columns can be changed to any stat: ['mae', 'pbias', 'rmse', 'r', 'kge', 'fbi', 'far', 'pod', 'acc']
generator_test.plot_probability_graph(pod_columnx='pod', far_columny='far')

# Generates a summary for station, you can specify or exclude a list of scenarios, by default:
# list_of_dict = ['statsrawGPM_dict','statsgwrGPM_dict','statsPISCO_dict','statsrain4pe_dict','statsexpGPM_dict']  
# .metrics_over_analysis(station_name , list_of_scenarios) 
station_test = generator_test.metrics_over_analysis('Crisnejas_ Sondor-Matara')
plot_1station_stats(station_test)

# Generates a temporal plot analysis over the scenarios: 'monthly', 'yearly' or 'both'
acumulate_comparison(generator_test.final_data, 'Mantaro_ Junin', 'yearly', '2005-01-01','2018-12-31')

# Sorts the stations in function of altitude in a dictionary of stations, it replaces the original sort
altitude_sort=sort_altitude(generator_test.final_data.keys(), generator_test.final_data)       

# Returns a df summary of all stations over the analysis, it can be filtered with lists
df = generator_test.join_stats(list_of_dictionaries = None, metrics = ['fbi', 'far', 'pod'])
# #Call a fbi, far, pod 'heat' map over multiple analysis
heat_map(df)


"""Quality check of the precipitation, it looks like aint a big problem"""
bucket_test = PrecipitationBucket(load_data_path,min_range,max_range)
bucket_test.checker()
bucket_test.rain_quality()
bucket_test.summary_graph()
bucket_test.normal_graph()

"""
Podriamos utilizar:
    Seasonal filter
    -Probalilidad alta de lluvia de Diciembre a Marzo = 1
    -Probabilidad media de lluvia de Abril Mayo y Noviembre = 0.5
    -Probabilidad zero de lluvia de Junio a Octubre = 0
                Octubre o Mayo 0.5
                othos meses  0
    Weekly cicles: weekend activities, personel does shitty work : CONDITION 1
    - El valor del dia viernes se repite el sabado y domingo? o = 1
      El valor del sabado y domingo son cero = 0.5
      No se cumplen las condiciones anteriores = 0
    
    Poor condition measures: CONDITION 2
    - Hay valores que se repiten tres dias consecutivos differentes de cero? = 1
      Hay valores que se repiten dos dias consecutivos differentes de cero? = 0.5
      No se cumplen las condiciones anteriores = 0
    
    Heavy precipitation truncation: CONDITION 3
    - Buscar el maximo mensual, el valor maximo se repite en el mismo mes +/- un threshold (ejemplo 5 cm) = 1
        Los operarios muchas veces para 'no repetir' numeros varian medidas pasadas un poquito
    - Si la condicion anterior no se cumple = 0

                prob-lluvia     COND1   COND2   COND3
    enero           1           1       1       1
    febrero         1
    marzo           1
    abril           0.5
    mayo            0.5
    junio           0
    julio           0
    agosto          0
    septiembre      0
    octubre         0
    noviembre       0.5
    diciembre       1

entonces si:  
            prob-lluvia = 1 y cualquier condicion es = 1: Es muy probable que el dato es incorrecto Red Flag
            prob-lluvia = 0.5 y cualquier condicion es = 1 o 0.5: Warning, es probable que el dato no sea real Yellow flag
            prob-lluvia = 0 y todas las condiciones = 0 : OK, Green flag

"""

# =============================================================================
# sandbox
# =============================================================================
patthenr = generator_test.final_data
from analysis_class import set_pandas_time
set_pandas_time(patthenr)

yearly = {}
for key in patthenr:
    data = patthenr[key].gwrGPM.loc['2005-01-01':'2018-12-31'].resample("YE").sum()
    obs[key] = data

obs = {}

for key in patthenr:
    data = patthenr[key].data.loc['2005-01-01':'2018-12-31'].resample("YE").sum()
    obs[key] = data
    
def plot_all_dataframes(dictionary_of_dfs):
    plt.figure(figsize=(12, 6))  # Set the figure size

    for key, df in dictionary_of_dfs.items():
        if 'precipitationCal' in df.columns and not df['precipitationCal'].isnull().all():
            y = df['precipitationCal']
        else:
            y = df['Precipitation']

        plt.plot(df.index, y, label=key)  # Plot with a label for the dictionary key

    plt.title("Combined Plot of All DataFrames")
    plt.xlabel("Date")
    plt.ylabel("Value")
    # plt.legend()  # Uncomment if you want legend
    plt.grid(axis='both', linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


plot_all_dataframes(yearly)
plot_all_dataframes(obs)