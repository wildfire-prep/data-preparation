# Data Preparation

Authors: Joshua Cohen, Ryan Green, Leilanie Rubinstein, Rachel Swick

## Repository Structure
```
📦 
├─ .gitignore
├─ README.md
├─ code
│  ├─ 00_label_data
│  │  ├─ 00_clean_inspections.ipynb
│  │  ├─ 01_join_parcels.ipynb
│  │  ├─ 02_clean_buildings_data.ipynb
│  │  ├─ 03_buildings_exploration.ipynb
│  │  ├─ 04_buffer_geometry.ipynb
│  │  ├─ 05_county_boundary.ipynb
│  │  └─ 06_interactive_map.ipynb
│  ├─ 01_satellite_imagery
│  │  ├─ 00_download_basemaps.ipynb
│  │  └─ 01_plot_imagery.ipynb
│  ├─ 02_clipping
│  │  ├─ 00_imagery_clipping.ipynb
│  │  ├─ 01_analytic_imagery.ipynb
│  │  ├─ imagery_clipping.ipynb
│  │  ├─ mosaic.tif
│  │  └─ mosaic_clipped_2022.tif
│  ├─ 03_rainfall
│  │  └─ clean_rainfall.ipynb
│  ├─ planet_tests
│  │  ├─ archive
│  │  │  ├─ planet_api_test.ipynb
│  │  │  ├─ planet_data_test.ipynb
│  │  │  ├─ planet_orders_test_split_implem.ipynb
│  │  │  ├─ polygon_split.ipynb
│  │  │  └─ udm2_test.ipynb
│  │  ├─ greater_UCSB-campus-aoi.geojson
│  │  ├─ planet_data_test_jj_copy.ipynb
│  │  ├─ planet_orders_jj_3.ipynb
│  │  ├─ planet_orders_test.ipynb
│  │  ├─ planet_orders_test_jj.ipynb
│  │  ├─ planet_orders_test_tool_implem.ipynb
│  │  └─ single_poly_test.ipynb
│  └─ utils
│     ├─ config.py
│     ├─ data_utils.py
│     └─ planet_utils.py
└─ environment
   ├─ environment.yml
   └─ requirements.txt
```
