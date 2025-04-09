#####################################################
##################### USER 
#####################################################

user = "leilanie"
# user = "ryan"
# user = "josh"
# user = "rachel"

#####################################################
##################### DIRECTORIES & FILE PATHS 
#####################################################

data_dir = "/capstone/wildfire_prep/data"
cleaned_inspections_dir = f"{data_dir}/inspections_data/cleaned_status"
basemap_dir = f"{data_dir}/basemaps"

repo_dir = f"/capstone/wildfire_prep/{user}/data-preparation"

fig_dir = f"{repo_dir}/figures"

#####################################################
##################### PARAMETERS
#####################################################

geodetic_crs = "EPSG:4326"
mercator_crs = "EPSG:3857"
albers_crs = "EPSG:3310" # equal area projection - use this for calculating buffer areas
mollweide_crs = "ESRI:54009"



