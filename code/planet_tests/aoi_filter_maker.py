import geopandas as gpd

from pyproj import Transformer

import json

from shapely.geometry import shape, mapping, MultiPolygon
from shapely.validation import explain_validity
from shapely.ops import unary_union


"""
Recursively converts tuples to lists in a nested dictionary or list structure. Use in the following functions.

Parameters:
    data (any): The input data (dict, list, or tuple).
    
Returns:
    any: The modified data with tuples converted to lists.
"""

def convert_tuples_to_lists(data):

        if isinstance(data, tuple):
            return [convert_tuples_to_lists(item) for item in data]
        elif isinstance(data, list):
            return [convert_tuples_to_lists(item) for item in data]
        elif isinstance(data, dict):
            return {key: convert_tuples_to_lists(value) for key, value in data.items()}
        else:
            return data  # Base case: return the item if it's not a tuple/list/dict



"""
Convex Hull GeoJSON Creator
write docs later
"""


def make_convex_hull(buffers_set_df, month_col = "month", month_vals = range(1,13)):


    # create coordinate converter
    transformer = Transformer.from_crs("EPSG:3310", "EPSG:4326", always_xy=True)

    # filter the geojson for parcels inspected on a given month
    filtered_df = buffers_set_df.loc[buffers_set_df[month_col].isin(month_vals)]

    # exit func if month has no inspections
    if len(filtered_df) < 1:
        return None

    convex_hull = filtered_df.unary_union.convex_hull
    geodict_hull = gpd.GeoDataFrame(geometry=[convex_hull], crs=filtered_df.crs).to_geo_dict()["features"][0]["geometry"]

    # tuples to lists
    geodict_hull = convert_tuples_to_lists(geodict_hull)
    # # give cords extra nesting
    # geodict_hull["coordinates"] = [geodict_hull["coordinates"]]

    # init geojson structure for master list
    geojson = {
        "type": "MultiPolygon", 
        "coordinates": []
    }

    curr_hull = geodict_hull["coordinates"][0]

    vert_count = 0

    curr_hull = [list(transformer.transform(*coord)) for coord in curr_hull]

    # round
    for coord in curr_hull: 
        coord[0] = round(coord[0],6)
        coord[1] = round(coord[1],6)
    
    # print vert count and area
    print(f"Vert count: {len(curr_hull)}")
    print(f"Hull area: {convex_hull.area}")

    # append to geojson
    geojson["coordinates"].append([curr_hull])


    # repair geometries if necessary
    geojson_shp = shape(geojson) # converts json to shapely

    if not geojson_shp.is_valid:
        geojson_shp = geojson_shp.buffer(0) # fixes geom
        geojson = mapping(geojson_shp) # converts back to geojson

    return geojson





"""
This function makes a single geojson containing a set amount of polygons, vertices, or both for use of stress testing the Planet orders API for it's capabilities with filtering AOI.

____Parameters____
buffers_set: a "buffer_geometries" geojson created by Lei
max_verts: the maximum number of vertices allowed in the json, summed from all polygons
begin: first polygon in the geojson to include, beginning of the inclusion window
end: last polygon in the geojson to include, end of the inclusion window
"""

def make_aoi_geojson(buffers_set_df, month_col = "month", month_vals = range(1,13), max_verts = float("inf"), begin = 0, end = -1):

    # create coordinate converter
    transformer = Transformer.from_crs("EPSG:3310", "EPSG:4326", always_xy=True)

    if month_col is not None:
        # filter the geojson for parcels inspected on a given month
        filtered_json = buffers_set_df.loc[buffers_set_df[month_col].isin(month_vals)].to_geo_dict()["features"]
    else:
        filtered_json = buffers_set_df.to_geo_dict()["features"]

    # reformat the coordinates list
    for i in range(len(filtered_json)):
        filtered_json[i]["geometry"]["coordinates"] = [[list(coord) for coord in list(filtered_json[i]["geometry"]["coordinates"][0])]]

    # init geojson structure for master list
    geojson = {
        "type": "MultiPolygon", 
        "coordinates": []
    }

    vert_count = 0

    # for every buffer within a window of the buffer collection
    for buffer in filtered_json[begin:end]:

        curr_buffer = (
                        buffer
                        ["geometry"]["coordinates"] # grab the coords specifically
                        [0] # unlists one stage, for formatting
                    )

        # convert the current buffer polygon to 4326
        curr_buffer = [list(transformer.transform(*coord)) for coord in curr_buffer]

        # round
        for coord in curr_buffer: 
            coord[0] = round(coord[0],6)
            coord[1] = round(coord[1],6)
        
        # close the polygon
        curr_buffer.append(curr_buffer[0])

        
        # append to master list
        geojson["coordinates"].append([curr_buffer])

        # repair geometries if necessary
        geojson_shp = shape(geojson) # converts json to shapely
        if not geojson_shp.is_valid:
            geojson_shp = geojson_shp.buffer(0) # fixes geom
            geojson = mapping(geojson_shp) # converts back to geojson


        vert_count = vert_count + len(curr_buffer)
        if vert_count > max_verts:
            vert_count = vert_count - len(curr_buffer)
            geojson = convert_tuples_to_lists(geojson)
            geojson["coordinates"].pop(-1)
            print(f"Max vert limit reached. Total vertices in this AOI filter: {vert_count}")
            return geojson

    geojson = convert_tuples_to_lists(geojson)

    print(f"Total vertices in this AOI filter: {vert_count}")
        

    return geojson


"""
Depricated versions
"""

# def make_aoi_geojson(buffers_set, max_verts = float("inf"), begin = 0, end = -1):

#     # create coordinate converter
#     transformer = Transformer.from_crs("EPSG:3310", "EPSG:4326", always_xy=True)

#     # init geojson structure for master list
#     geojson = {
#         "type": "MultiPolygon", 
#         "features": []
#     }

#     vert_count = 0

#     # for every buffer within a window of the buffer collection
#     for buffer in buffers_set[begin:end]:

#         curr_buffer = (
#                         buffer
#                         ["geometry"]["coordinates"] # grab the coords specifically
#                         [0] # unlists one stage, for formatting
#                     )
        
#         curr_buffer = [transformer.transform(*coord) for coord in curr_buffer]

#         feature = {
#             "type": "Feature",
#             "properties": {},
#             "geometry": {
#                 "type": "Polygon",
#                 "coordinates": [
#                     curr_buffer
#                 ]
#             }
#         }

#         # append to master list
#         geojson["features"].append(feature)

#         vert_count = vert_count + len(curr_buffer)
#         if vert_count > max_verts:
#             vert_count = vert_count - len(curr_buffer)
#             geojson["features"].pop(-1)
#             print(f"Max vert limit reached. Total vertices in this AOI filter: {vert_count}")
#             return geojson

#     print(f"Total vertices in this AOI filter: {vert_count}")
        

#     return geojson





# def make_aoi_geojson(buffers_set, max_verts = float("inf"), begin = 0, end = -1):

#     # create coordinate converter
#     transformer = Transformer.from_crs("EPSG:3310", "EPSG:4326", always_xy=True)

#     # init geojson structure for master list
#     geojson = {
#         "type": "MultiPolygon", 
#         "coordinates": []
#     }

#     vert_count = 0

#     # for every buffer within a window of the buffer collection
#     for buffer in buffers_set[begin:end]:

#         curr_buffer = (
#                         buffer
#                         ["geometry"]["coordinates"] # grab the coords specifically
#                         [0] # unlists one stage, for formatting
#                     )
#         # print(curr_buffer)

#         # convert the current buffer polygon to 4326
#         curr_buffer = [list(transformer.transform(*coord)) for coord in curr_buffer]
#         # print(curr_buffer)

#         # round
#         for coord in curr_buffer: 
#             coord[0] = round(coord[0],6)
#             coord[1] = round(coord[1],6)

#         # close the polygon
#         curr_buffer.append(curr_buffer[0])


#         # append to master list
#         geojson["coordinates"].append([curr_buffer])


#         # repair geometries if necessary
#         geojson_shp = shape(geojson) # converts json to shapely
#         if not geojson_shp.is_valid:
#             geojson_shp = geojson_shp.buffer(0) # fixes geom
#             geojson = mapping(geojson_shp) # converts back to geojson
            

#         vert_count = vert_count + len(curr_buffer)
#         if vert_count > max_verts:
#             vert_count = vert_count - len(curr_buffer)
#             geojson["coordinates"].pop(-1)
#             print(f"Max vert limit reached. Total vertices in this AOI filter: {vert_count}")
#             return geojson

#     print(f"Total vertices in this AOI filter: {vert_count}")
        

#     return geojson




"""
This function makes a list of AOI geojsons, such that they can be fed into the orders API iteratively.

____Parameters____
buffers_set: a "buffer_geometries" geojson created by Lei
groups_of: how many groups of polygons to split the geojson into
"""

# def make_aoi_geojson_collection(buffers_set, groups_of = 1):


#     # create coordinate converter
#     transformer = Transformer.from_crs("EPSG:3310", "EPSG:4326", always_xy=True)


#     # init
#     geojson_collection = [] # master list for all feature collections
#     begin = 0 # starting index of window
#     end = groups_of # ending index of window

#     # loop will repeat until the top of the window is greater than the actual number of polygons in the buffer set
#     while end < len(buffers_set) + groups_of:

#         # and when that happens, force the top of the window to equal the total number of polygons
#         if end > len(buffers_set):
#             print(f"cutting short now by {end - len(buffers_set)}")
#             end = len(buffers_set)

#         print(f"step: begin = {begin}, end = {end}")

#         # init geojson structure for collection
#         geojson = {
#         "type": "MultiPolygon", 
#         "coordinates": []
#         }

#         # for every buffer within a window of the buffer collection
#         for buffer in buffers_set[begin:end]:

#             curr_buffer = (
#                 buffer
#                 ["geometry"]["coordinates"] # grab the coords specifically
#                 [0] # unlists one stage, for formatting
#                 )
            
#             # convert the current buffer polygon to 4326
#             curr_buffer = [list(transformer.transform(*coord)) for coord in curr_buffer]

#             # close the polygon
#             curr_buffer.append(curr_buffer[0])

#             # # round
#             # for coord in curr_buffer: 
#             #     coord[0] = round(coord[0],6)
#             #     coord[1] = round(coord[1],6)
            

#             # append to current feature collection
#             geojson["coordinates"].append([curr_buffer])


#             # repair geometries if necessary
#             geojson_ch = unary_union(
#                             MultiPolygon(
#                                     [
#                                     shape({"type": "Polygon", "coordinates": coords}) for coords in geojson["coordinates"]
#                                 ]
#                             )
#                         )



#         # print(geojson_shp)
#         if not geojson_ch.is_valid:
#             geojson_ch = geojson_ch.buffer(1e-9) # fixes geom
#             geojson = mapping(geojson_ch) # converts back to geojson
#             geojson["coordinates"] = list(geojson["coordinates"])
#             # print([[[list(geojson) for geojson in geojson["coordinates"][0][0]]]])
                
#             # # print(geojson)


#             # # geojson_shp = shape(geojson)
#             # # geojson_ch = geojson_shp.convex_hull
#             # # geojson = mapping(geojson_ch)
#             # # print(geojson)
        
#         # update window for next set of polygons
#         begin = begin + groups_of
#         end = end + groups_of

#         # append current feature collection to master collection
#         geojson_collection.append(geojson)
        
#         # convert all tuples to lists
#         geojson_collection = convert_tuples_to_lists(geojson_collection)

#     # print(f"total fails {fail_counter} out of {len(buffers_set)}")

#     return geojson_collection





"""
Deprecated versions
"""



"""v2"""
def make_aoi_geojson_collection(buffers_set, groups_of = 1):


    # create coordinate converter
    transformer = Transformer.from_crs("EPSG:3310", "EPSG:4326", always_xy=True)


    # init
    geojson_collection = [] # master list for all feature collections
    begin = 0 # starting index of window
    end = groups_of # ending index of window

    # loop will repeat until the top of the window is greater than the actual number of polygons in the buffer set
    while end < len(buffers_set) + groups_of:

        # and when that happens, force the top of the window to equal the total number of polygons
        if end > len(buffers_set):
            print(f"cutting short now by {end - len(buffers_set)}")
            end = len(buffers_set)

        print(f"step: begin = {begin}, end = {end}")

        # init geojson structure for collection
        geojson = {
        "type": "MultiPolygon", 
        "coordinates": []
        }

        # for every buffer within a window of the buffer collection
        for buffer in buffers_set[begin:end]:

            curr_buffer = (
                buffer
                ["geometry"]["coordinates"] # grab the coords specifically
                [0] # unlists one stage, for formatting
                )
            
            # convert the current buffer polygon to 4326
            curr_buffer = [list(transformer.transform(*coord)) for coord in curr_buffer]

            # close the polygon
            curr_buffer.append(curr_buffer[0])

            # round
            # for coord in curr_buffer: 
            #     coord[0] = round(coord[0],6)
            #     coord[1] = round(coord[1],6)
            

            # append to current feature collection
            geojson["coordinates"].append([curr_buffer])


        # repair geometries if necessary
        geojson_shp = shape(geojson) # converts json to shapely

        # print(geojson_shp)
        if not geojson_shp.is_valid:
            geojson_shp = geojson_shp.buffer(0) # fixes geom
            geojson = mapping(geojson_shp) # converts back to geojson
            geojson["coordinates"] = list(geojson["coordinates"])
            # print([[[list(geojson) for geojson in geojson["coordinates"][0][0]]]])
                
            # # print(geojson)


            # # geojson_shp = shape(geojson)
            # # geojson_ch = geojson_shp.convex_hull
            # # geojson = mapping(geojson_ch)
            # # print(geojson)
        
        # update window for next set of polygons
        begin = begin + groups_of
        end = end + groups_of

        # append current feature collection to master collection
        geojson_collection.append(geojson)
        
        # convert all tuples to lists
        geojson_collection = convert_tuples_to_lists(geojson_collection)

    # print(f"total fails {fail_counter} out of {len(buffers_set)}")

    return geojson_collection



"""v1"""
# def make_aoi_geojson_collection(buffers_set, groups_of = 1):

#     # buffers_set is one of the buffers geojsons that Lei generated
#     # groups_of is the number of polygons per feature collection

#     # init
#     geojson_collection = [] # master list for all feature collections
#     begin = 0 # starting index of window
#     end = groups_of # ending index of window

#     # loop will repeat until the top of the window is greater than the actual number of polygons in the buffer set
#     while end < len(buffers_set) + groups_of:
#         print(f"step: begin = {begin}, end = {end}")

#         # and when that happens, force the top of the window to equal the total number of polygons
#         if end > len(buffers_set) + groups_of:
#             end = len(buffers_set)

#         # declare feature collection structure
#         geojson = {
#             "type": "MultiPolygon", 
#             "features": []
#         }

#         # for every buffer within a window of the buffer collection
#         for buffer in buffers_set[begin:end]:

#             feature = {
#                 "type": "Feature",
#                 "properties": {},
#                 "geometry": {
#                     "type": "Polygon",
#                     "coordinates": [
#                         (
#                             buffer
#                             ["geometry"]["coordinates"] # grab the coords specifically
#                             [0] # unlists one stage, for formatting
#                         )
#                     ]
#                 }
#             }

#             # append to current feature collection
#             geojson["features"].append(feature)
    
#         # update window for next set of polygons
#         begin = begin + groups_of
#         end = end + groups_of

#         # append current feature collection to master collection
#         geojson_collection.append(geojson)
        
        

#     return geojson_collection