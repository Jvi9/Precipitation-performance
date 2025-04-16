
<h2>Evaluating Precipitation Datasets in the Andes of Peru</h2>

## Data sources:
```
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
├── datasets/                     # Pickle format of the loaded precipitation datasets
│   ├── gb_stations_data.pkl        # Contains only the station information
│   ├── data_locked_loaded.pkl      # Contains all the datasets added
├── gis/                    
│   ├── ...  	              # shps, kml, and other auxiliar info
│   ├── all_stations_locations.kml  # all the stations in the area
│   ├── selected_stations.kml       # stations kml multipoint file filtered
│ 
├── loader/                    
│   ├── load_datasets.py  	    # Load the data sources
├── precipitation_core.py  	# Process ground station information retrieved from ANA (Peru) platform
├── analysis_class.py            # Functions and classes to process the information
├── core_analysis.py            # Process the information in a easy and interactive way


```

<h2>Preview statistics from the scenarios ran</h2>

<img src="https://github.com/user-attachments/assets/d9fc7cbb-1b5e-48a8-88d3-f221691becfe" width="250">
<img src="https://github.com/user-attachments/assets/87570789-a8ef-4130-be44-cca42866a308" width="250">
<img src="https://github.com/user-attachments/assets/3a57b238-8d3d-4c1f-ab2d-4c4abfcdbacc" width="250">
<img src="https://github.com/user-attachments/assets/52f3ead7-cc2f-401c-bc32-edf8278df8be" width="250">

<h2>Preview view: multipoint maps</h2>

<img src="https://github.com/user-attachments/assets/9ba6cc9b-9694-41a0-b658-8d2b7b0be9de" width="250">
<img src="https://github.com/user-attachments/assets/c5a3d9ff-84e1-48ce-b59c-cc7d3c89b91d" width="250">
<img src="https://github.com/user-attachments/assets/9a99e5d7-ff4f-4d42-8e97-7b3dfaa994e4" width="250">
<img src="https://github.com/user-attachments/assets/7a01044e-722e-4478-a3d2-7b6e22ff0e10" width="250">