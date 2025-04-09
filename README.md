<h3>Evaluating Precipitation Datasets in the Andes of Peru</h3>
```
Data sources:
- ANA data for ground based stations
- PISCO
- RAIN4PE
- GPM
- GPM Downscaled with geographical weighted regression
- GPM Downscaled with exponential regression

Schema of the code
precipitation-performance/
├── data/                       # Contains ground station information in csv
├── graphs/                     # Relevant comparison graphs
├── loader/                    
│   ├── precipitation_core.py  	# Process ground station information
│   ├── load_datasets.py  	    # Load the data sources
├── data_analysis.py            # Calls the statistics along the sources
├── gb_stations_data.pkl        # Contains only the station information
├── data_locked_loaded.pkl      # Contains ground station information in csv
├── all_stations_locations.kml  # all the stations in the area
├── selected_stations.kml       # stations kml multipoint file filtered

## Preview view: multipoint maps
<img src="https://github.com/user-attachments/assets/9ba6cc9b-9694-41a0-b658-8d2b7b0be9de" width="150">
<img src="https://github.com/user-attachments/assets/c5a3d9ff-84e1-48ce-b59c-cc7d3c89b91d" width="150">
<img src="https://github.com/user-attachments/assets/9a99e5d7-ff4f-4d42-8e97-7b3dfaa994e4" width="150">
<img src="https://github.com/user-attachments/assets/7a01044e-722e-4478-a3d2-7b6e22ff0e10" width="150">