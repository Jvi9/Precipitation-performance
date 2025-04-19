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

import pandas as pd
import numpy as np

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
                return "Red Flag"
            elif prob == 0.5 and any(c >= 0.5 for c in conds):
                return "Yellow Flag"
            elif prob == 0.0 and all(c == 0.0 for c in conds):
                return "Green Flag"
            else:
                return "OK"
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
        flag_counts.plot(kind="bar", color=["red", "grey", "green", "gold"])
        plt.title("Flag Distribution Summary")
        plt.xlabel("Flag Type")
        plt.ylabel("Number of Days")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


# Crear el objeto con tu dataframe original
dummy_copy = pd.DataFrame(test_df.data.copy().astype(float))

checker = PrecipitationQualityChecker(dummy_copy)

# Ejecutar todos los chequeos
checker.run_all_checks()

# Obtener los resultados
result_df = checker.get_result()
checker.plot_flag_summary()
# Mostrar los últimos valores
print(result_df.tail(15))
