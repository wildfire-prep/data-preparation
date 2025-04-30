# Data Preparation

This repository houses data preparation notebooks for our project, titled: Remote Sensing of Brush Clearance to Enhance Wildfire Preparedness.

## Authors

Joshua Cohen
Ryan Green
Leilanie Rubinstein
Rachel Swick

## Data

Data is stored separately from our repository for the purposes of this analysis.

## Repository Structure

```
📦 
├─ .gitignore
├─ README.md
├─ code
│  ├─ 00_label_data
│  │  ├─ 00_clean_inspections.ipynb
│  │  ├─ 01_join_parcels.ipynb
│  │  ├─ 02_append_inspections_data.ipynb
│  │  ├─ 02_clean_buildings_data.ipynb
│  │  ├─ 02_data_cleaning_2.ipynb
│  │  ├─ 02_inspection_data_cleaning.ipynb
│  │  ├─ 03_buildings_exploration.ipynb
│  │  ├─ 04_buffer_geometry.ipynb
│  │  └─ inspections_master_stats.ipynb
│  ├─ 01_satellite_imagery
│  │  ├─ 00_download_basemaps.ipynb
│  │  └─ 01_plot_imagery.ipynb
│  ├─ 02_clipping
│  │  ├─ 00_imagery_clipping.ipynb
│  │  ├─ 01_analytic_imagery.ipynb
│  │  ├─ 02_single_scene_NDVI.ipynb
│  │  ├─ 03_clip_basemap_images.ipynb
│  │  ├─ imagery_clipping.ipynb
│  │  └─ single_parcel_clip.ipynb
│  ├─ 03_rainfall
│  │  ├─ clean_rainfall.ipynb
│  │  └─ join_rainfall.ipynb
│  ├─ 04_landcover_stats
│  │  ├─ landcover_data
│  │  │  └─ GAP_National_Terrestrial_Ecosystems.csv
│  │  └─ landcover_stats.ipynb
│  ├─ 05_randomforest
│  │  ├─ randomforest_attempt.ipynb
│  │  └─ randomforest_attempt2.ipynb
│  ├─ 06_model_development
│  │  ├─ 00_mosaiks.ipynb
│  │  ├─ 01_mosaiks.py
│  │  ├─ 02_ridge_classifier.ipynb
│  │  └─ 03_random_forest.ipynb
│  ├─ planet_tests
│  │  ├─ archive
│  │  │  ├─ planet_api_test.ipynb
│  │  │  ├─ planet_data_test.ipynb
│  │  │  ├─ planet_data_test_jj_copy.ipynb
│  │  │  ├─ planet_orders_test_jj.ipynb
│  │  │  ├─ planet_orders_test_split_implem.ipynb
│  │  │  ├─ polygon_split.ipynb
│  │  │  └─ udm2_test.ipynb
│  │  ├─ buffer_aoi_ordering.ipynb
│  │  ├─ greater_UCSB-campus-aoi.geojson
│  │  ├─ planet_orders_jj_3.ipynb
│  │  ├─ planet_orders_test.ipynb
│  │  ├─ planet_orders_test_tool_implem.ipynb
│  │  └─ single_poly_test.ipynb
│  └─ utils
│     ├─ data_utils.py
│     └─ planet_utils.py
└─ environment
   ├─ environment.yml
   └─ requirements.txt
```

## License

MIT Commons