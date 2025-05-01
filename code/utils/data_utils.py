"""
This module contains utility functions for modeling artisanal scale mining,
in particular, manipulating data.

Functions:
----------
    - create_country_grid
    - map_columns
    - assign_overlap_group_ids
    - create_buffer_column
    - data_confidence
    - stratified_group_split
    - split_data
        - extract_features
    - filter_mining_sites
    - prepend_keys
    - chunks
    - extract_features_fn
    - reorder_columns
    - contains_nan
    - convert_list_columns_to_float

Classes:
----------
    - VisualBasemapDataset
        - __init__
        - __len__
        - __getitem__


Author: Cullen Molitor, Cullen_Molitor@bren.ucsb.edu
Date: 2023-10-20
"""

import os
import torch
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio.features

from operator import gt, ge
from ast import literal_eval
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import nearest_points
from shapely.geometry import box
from torch.utils.data import Dataset
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Union, Callable, Any, Optional


def create_grid(
    borders,
    resolution: float = 0.01,
    geometry_col: str = "geometry",
    id_col: str = "NAME",
    return_ids: bool = False,
) -> pd.DataFrame:
    """
    Create a grid of latitude and longitude coordinates for one or more geometries.
    It can accept a bounding box, a single polygon (or other Shapely geometry),
    or a GeoDataFrame with a geometry column.

    Parameters
    ----------
    borders : list or tuple or shapely.geometry.BaseGeometry or geopandas.GeoDataFrame
        - If list/tuple of length 4, interpreted as a bounding box: [minx, miny, maxx, maxy].
        - If a Shapely geometry (Polygon, MultiPolygon, etc.), creates a single-row GeoDataFrame.
        - If a GeoDataFrame, the function iterates over its rows.
    resolution : float, optional
        Grid resolution in degrees, default 0.01.
    geometry_col : str, optional
        Column name for the geometry in the resulting GeoDataFrame, by default "geometry".
    id_col : str, optional
        Column name in the GeoDataFrame to use as the ID column, or the name
        for the new column if bounding box / single polygon is provided. Default is "NAME".
    return_ids : bool, optional
        If True, generate and return the unique IDs for each grid cell. This will create a
        column 'unique_id' which follows the pattern 'lon_{lon}__lat_{lat}'. This option slows
        down the overall operation. Default is False.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'lat': latitude values (Y)
        - 'lon': longitude values (X)
        - `[id_col]`: the identifier for each geometry feature
        - 'unique_id': a string combining [id_col] + lon/lat for uniqueness (optional)
    """

    # 1. Convert input to a GeoDataFrame
    gdf = _to_geodataframe(borders, geometry_col, id_col)

    # 2. Ensure there's an ID column in the GeoDataFrame
    if id_col not in gdf.columns:
        # If user didn't provide an ID col for bounding box or single geometry,
        # assign a placeholder ID. For a multi-row GDF, user is expected to pass
        # an existing column name.
        gdf[id_col] = [f"feature_{i}" for i in range(len(gdf))]

    # 3. Rasterize each geometry and collect points
    result_list = []
    for _, row in gdf.iterrows():
        geom = row[geometry_col]
        this_id = row[id_col]

        if geom.is_empty:
            # Skip empty geometries
            continue

        minx, miny, maxx, maxy = geom.bounds

        # ---- Create arrays for lat and lon values (Note: lat reversed) ----
        # The 0.005 shift ensures that coordinates align on .005
        lats = np.arange(
            np.ceil(maxy / resolution) * resolution - 0.005, miny, -resolution
        )
        lons = np.arange(
            np.ceil(minx / resolution) * resolution + 0.005, maxx, resolution
        )

        if len(lats) == 0 or len(lons) == 0:
            # If bounding box is too small or resolution is large, might be empty
            continue

        # ---- Create a meshgrid ----
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # ---- Rasterize the geometry ----
        out_shape = (len(lats), len(lons))
        transform = rasterio.transform.from_bounds(
            minx, miny, maxx, maxy, out_shape[1], out_shape[0]
        )

        mask = rasterio.features.rasterize(
            [(geom, 1)],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        # ---- Extract the lat and lon values using the mask ----
        lat_values = lat_grid[mask == 1]
        lon_values = lon_grid[mask == 1]

        # ---- Create a DataFrame and append to the result list ----
        temp_df = pd.DataFrame({"lat": lat_values, "lon": lon_values})
        temp_df[id_col] = this_id
        result_list.append(temp_df)

    # 4. Concatenate the results
    if len(result_list) == 0:
        final_result = pd.DataFrame(columns=["lat", "lon", id_col, "unique_id"])
    else:
        final_result = pd.concat(result_list, ignore_index=True)
        if return_ids:
            # --- Create the unique_id column ---
            # e.g. 'lon_-10.005__lat_9.995'
            final_result["lon_rounded"] = final_result["lon"].round(3).astype(str)
            final_result["lat_rounded"] = final_result["lat"].round(3).astype(str)

            final_result["unique_id"] = (
                "lon_"
                + final_result["lon_rounded"]
                + "__lat_"
                + final_result["lat_rounded"]
            )

            final_result.drop(["lon_rounded", "lat_rounded"], axis=1, inplace=True)

    return final_result


def _to_geodataframe(borders, geometry_col: str, id_col: str) -> gpd.GeoDataFrame:
    """
    Internal helper that converts various input types into a standardized GeoDataFrame.

    Parameters
    ----------
    borders : list/tuple, shapely geometry, or GeoDataFrame
        Bounding box (list/tuple of length 4),
        single Shapely geometry (Polygon, MultiPolygon, etc.),
        or a GeoDataFrame.
    geometry_col : str
        The name of the geometry column to use or create.
    id_col : str
        The column in which to store or look for an ID (if relevant).

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with columns [id_col, geometry_col].
    """
    # Case 1: bounding box
    if isinstance(borders, (list, tuple)) and len(borders) == 4:
        minx, miny, maxx, maxy = borders
        geom = box(minx, miny, maxx, maxy)
        gdf = gpd.GeoDataFrame(
            {id_col: ["bbox_1"], geometry_col: [geom]}, crs="EPSG:4326"
        )

    # Case 2: single shapely geometry
    elif isinstance(borders, (Polygon, MultiPolygon, GeometryCollection)):
        gdf = gpd.GeoDataFrame(
            {id_col: ["geom_1"], geometry_col: [borders]}, crs="EPSG:4326"
        )

    # Case 3: GeoDataFrame
    elif isinstance(borders, gpd.GeoDataFrame):
        # If geometry_col does not exist, rename the current geometry column
        # so everything is consistent
        if geometry_col not in borders.columns:
            borders = borders.rename(columns={borders.geometry.name: geometry_col})

        gdf = borders.copy()
        gdf = gdf.set_geometry(geometry_col)

    else:
        raise ValueError(
            "Unsupported input for 'borders'. Must be one of:\n"
            "1) [minx, miny, maxx, maxy]\n"
            "2) A Shapely geometry (Polygon, MultiPolygon, etc.)\n"
            "3) A GeoDataFrame"
        )

    return gdf


def create_country_grid(borders, resolution=0.01):
    """
    Create a grid of latitude and longitude coordinates within each country's borders.

    Parameters
    ----------
    borders : geopandas.GeoDataFrame
        GeoDataFrame containing country borders with ISO3 and geometry columns.
    resolution : float, optional
        Resolution of the grid, default is 0.01.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing latitude, longitude, and country information.
    """

    result_list = []

    for _, row in borders.iterrows():
        country = row["ISO3"]
        geom = row["geometry"]

        # Get bounding box of the geometry
        minx, miny, maxx, maxy = geom.bounds

        # Create arrays for latitude and longitude values, reverse latitude
        lats = np.arange(
            np.ceil(maxy / resolution) * resolution - 0.005, miny, -resolution
        )
        lons = np.arange(
            np.ceil(minx / resolution) * resolution + 0.005, maxx, resolution
        )

        # Create a meshgrid for the longitude and latitude values
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Transform the geometry into the raster space
        out_shape = (len(lats), len(lons))
        transform = rasterio.transform.from_bounds(
            minx, miny, maxx, maxy, out_shape[1], out_shape[0]
        )

        # Rasterize the polygon
        mask = rasterio.features.rasterize(
            [(geom, 1)],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        # Extract the lat and lon values using the mask
        lat_values = lat_grid[mask == 1]
        lon_values = lon_grid[mask == 1]

        # Create a DataFrame and append to the result list
        country_df = pd.DataFrame(
            {"lat": lat_values, "lon": lon_values, "country": country}
        )
        result_list.append(country_df)

    # Concatenate the results
    final_result = pd.concat(result_list, ignore_index=True)

    return final_result


def map_columns(row: pd.Series) -> tuple:
    """
    Map the 'artisnl' and 'commrcl' columns to new 'mine_type' and 'label' columns.

    Parameters
    ----------
    row : pd.Series
        A row in the DataFrame.

    Returns
    -------
    tuple
        A tuple containing the values for 'mine_type' and 'label'.

    Example
    --------
    >>> map_columns(pd.Series({'artisnl': 1, 'commrcl': 0}))
    ('artisanal', 1)
    """
    if row["artisnl"] == 1 and row["commrcl"] == 1:
        return ("commercial", 1)
    elif row["artisnl"] == 1:
        return ("artisanal", 1)
    elif row["commrcl"] == 1:
        return ("commercial", 1)
    else:
        return ("negative", 0)


def assign_overlap_group_ids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Assign a unique group ID for overlapping polygons in a GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The input GeoDataFrame with polygons to check for overlaps.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with an additional column containing group IDs for overlapping polygons.

    Notes
    -----
    This function iterates over rows in the GeoDataFrame, which may be computationally
    expensive for large GeoDataFrames.
    """

    gdf["group_id"] = np.zeros(len(gdf), dtype=int)
    current_group_id = 1

    for idx, row in gdf.iterrows():
        # Extract geometry for the current row
        geom = row.geometry

        # Check if already assigned a group
        if gdf.at[idx, "group_id"] == 0:
            overlaps = gdf.geometry.intersects(geom)
            gdf.loc[overlaps, "group_id"] = current_group_id
            current_group_id += 1

    return gdf


def add_province_column(data, provinces):
    """
    Adds a 'province' column to the data DataFrame. Points are first checked
    if they are within a province. If not, the closest province is found.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing columns 'lon', 'lat', and others.
    provinces : pandas.DataFrame
        DataFrame containing a 'geometry' column with province shapes and 'NAME_1' for province names.

    Returns
    -------
    geopandas.GeoDataFrame
        Modified data DataFrame with a new 'province' column.
    """
    # Convert to GeoDataFrame
    gdf_grid_cells = gpd.GeoDataFrame(
        data,
        geometry=[Point(xy) for xy in zip(data.lon, data.lat)],
        crs=provinces.crs,
    )

    # Spatial join for points within provinces
    gdf_joined = gpd.sjoin(
        gdf_grid_cells,
        provinces[["NAME_1", "geometry"]],
        how="left",
        predicate="within",
    )
    gdf_joined["province"] = gdf_joined["NAME_1"]

    # Identify points not within any province
    no_province = gdf_joined["province"].isnull()
    gdf_no_province = gdf_joined[no_province]

    # Function to find closest province
    def find_closest_province(point, provinces_gdf):
        nearest = provinces_gdf.geometry.apply(
            lambda geom: geom.distance(point)
        ).idxmin()
        return provinces_gdf.iloc[nearest]["NAME_1"]

    # Find closest province for points not within any province
    closest_provinces = gdf_no_province.apply(
        lambda row: find_closest_province(row.geometry, provinces), axis=1
    )

    # Update the main DataFrame
    gdf_joined.loc[no_province, "province"] = closest_provinces

    return gdf_joined.drop(columns=["NAME_1", "index_right"])


def create_buffer_column(
    df: pd.DataFrame,
    label_column: str,
    positive_value,
    n_cells: int,
    cell_size: float,
) -> pd.DataFrame:
    """
    Create a new column in a DataFrame that flags all grid cells within a specified grid cell distance
    from any grid cell with a positive label. This considers a square buffer including diagonal neighbors.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the grid cell data with latitude, longitude, and label columns.
    label_column : str
        The name of the column containing the label data.
    positive_value : Any
        The value in the label_column that identifies a positive label.
    n_cells : int
        The number of cells to use to calculate the buffer size (Chebyshev distance).
    cell_size : float
        The size of each cell in degrees.

    Returns
    -------
    DataFrame
        The DataFrame with an additional boolean column indicating proximity to positive labels.
    """
    # Define buffer size as n_cells * cell_size (assuming each cell is 0.01 degrees)
    buffer_size = n_cells * cell_size

    # Create a GeoDataFrame from the DataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))

    # Extract positive cells
    positive_cells = gdf[gdf[label_column] == positive_value]
    positive_cells = positive_cells.loc[~(positive_cells.sample_type == "SNL")]

    # Create a square buffer around positive cells to include diagonal neighbors
    positive_cells.loc[:, "geometry"] = positive_cells.geometry.buffer(
        buffer_size, cap_style=3
    )

    # Prepare a spatial index for the positive cells
    positive_cells_sindex = positive_cells.sindex

    # Check if each cell is within the buffer of any positive cell
    def is_within_buffer(row):
        possible_matches_index = list(
            positive_cells_sindex.intersection(row.geometry.bounds)
        )
        possible_matches = positive_cells.iloc[possible_matches_index]
        return possible_matches.geometry.contains(row.geometry).any()

    # Apply the function to all rows
    df[f"{n_cells}_cell_buffer"] = gdf.apply(is_within_buffer, axis=1)

    return df


def filter_confidence(data, confidence=3, operator=ge):
    """
    Filters a pandas DataFrame containing data on confidence level.

    Parameters:
    -----------
    data: pd.DataFrame
        The input DataFrame to be filtered.
    confidence: int, optional (default=3)
        The confidence level above which data will be retained.
        Any data with a confidence level less than this will be removed.
    operator: function, optional (default=ge)
        The operator to be used for filtering the confidence level.
        Default is 'greater than or equal to' (ge), but can also be 'greater than' (gt).

    Returns:
    --------
    pd.DataFrame:
        A filtered DataFrame with the specified parameters.
    """
    # Filter data based on confidence level
    mask = operator(data["confidence"], confidence)
    data = data[mask]

    # Reset the index to remove any gaps created by removing rows
    return data.reset_index(drop=True)


def stratified_group_split(
    df, group_col, stratify_col, test_size=0.2, random_state=None
):
    """
    Split the data into train and test sets, while ensuring that values with the same group_id are
    split together and stratifying based on a given column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    group_col : str
        Name of the column that indicates the group.
    stratify_col : str
        Name of the column based on which the stratification needs to be done.
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split.
    random_state : int or None, optional
        Random state for reproducibility.

    Returns
    -------
    train : pandas.DataFrame
        Training subset of the input dataframe.
    test : pandas.DataFrame
        Testing subset of the input dataframe.
    """

    # Create a DataFrame with one row for each unique value of 'group_col'
    # and the mode of 'stratify_col' for that group.
    group_stratify = (
        df.groupby(group_col).apply(lambda x: x[stratify_col].mode()[0]).reset_index()
    )
    group_stratify.columns = [group_col, stratify_col]

    # Split unique group_ids into train and test sets, stratifying by 'stratify_col'
    train_groups, test_groups = train_test_split(
        group_stratify[group_col],
        stratify=group_stratify[stratify_col],
        test_size=test_size,
        random_state=random_state,
    )

    # Map the group_ids back to the original dataframe
    train = df[df[group_col].isin(train_groups)]
    test = df[df[group_col].isin(test_groups)]

    return train, test


def split_data(
    data: pd.DataFrame,
    img_feat_cols: list,
    geo_feat_cols: list,
    location_col: str = "country",
    stratify_col: str = "country",
    label_col: str = "label",
    sample_type_col: str = "sample_type",
    mine_type_col: str = "mine_type",
    test_location: str = None,
    random_state: int = None,
) -> Tuple:
    """
    Splits a dataset into train and test sets based on a grouping column.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input data to be split.
    img_feat_cols : list
        The list of column names containing the image features.
    geo_feat_cols : list
        The list of column names containing the geologic features.
    location_col : str, optional
        The name of the column used to group the data geographically. Default is 'country'.
    stratify_col : str, optional
        The name of the column used to stratify the data. Default is 'country'.
    label_col : str
        The name of the column containing the target labels.
    sample_type_col : str
        The name of the column containing the sample types.
    mine_type_col : str
        The name of the column containing the mining types.
    test_location : str, optional
        The name of the geographic group used as test data. If None, a random test set is created.
    random_state : int, optional
        The random seed used to generate the random test set.

    Returns:
    --------
    tuple
        A tuple of arrays containing the train and test data for image features, geologic features, target labels,
        geographic group, sample types, and mining types.
    """

    def extract_features(df):
        return (
            df[img_feat_cols],
            df[geo_feat_cols],
            df[label_col],
            df[location_col],
            df[sample_type_col],
            df[mine_type_col],
        )

    if test_location is None:
        train, test = train_test_split(
            data,
            stratify=data[stratify_col] if stratify_col is not None else None,
            test_size=0.2,
            random_state=random_state,
        )
    else:
        test = data[data[location_col] == test_location]
        train = data[data[location_col] != test_location]

    (
        X_train_img,
        X_train_geo,
        y_train,
        location_train,
        sample_train,
        mine_type_train,
    ) = extract_features(train)
    (
        X_test_img,
        X_test_geo,
        y_test,
        location_test,
        sample_test,
        mine_type_test,
    ) = extract_features(test)

    return (
        X_train_img,
        X_test_img,
        X_train_geo,
        X_test_geo,
        y_train,
        y_test,
        location_train,
        location_test,
        sample_train,
        sample_test,
        mine_type_train,
        mine_type_test,
    )


def filter_mining_sites(
    *arrays, mining_types, include_commercial_mines, return_mask=False
):
    """
    Filters mining sites from the input data based on the `include_commercial_mines` flag.

    Parameters:
    -----------
    *arrays (numpy.ndarray):
        An arbitrary number of arrays to be filtered in the same way.
    mining_types (numpy.ndarray):
        An array of mining site types
        (0 = None, 1 = commercial, 2 = artisanal).
    include_commercial_mines (bool):
        Whether or not to include mining sites categorized
        as "Commercial" in the filtered data.
    return_mask (bool, optional):
        Whether to return the boolean mask used for filtering. Default is False.

    Returns:
    --------
    tuple:
        A tuple of numpy arrays containing the filtered data, in the same order
        as they were passed in. If return_mask is True, the mask is also returned
        as the last element of the tuple.
    """
    if not arrays:
        return tuple()

    if not include_commercial_mines:
        commercial_mask = mining_types != "commercial"
        filtered_arrays = tuple(data[commercial_mask] for data in arrays)
    else:
        commercial_mask = mining_types == mining_types
        filtered_arrays = arrays

    if return_mask:
        return filtered_arrays + (commercial_mask,)
    else:
        return filtered_arrays


def prepend_keys(
    dict_to_update: dict, prepend_string: str, exception_key: str = "random_state"
) -> dict:
    """
    Prepends a given string to all keys in a dictionary, except for a specific key if provided.

    Parameters:
    -----------
    dict_to_update (dict):
        The dictionary to modify.
    prepend_string (str):
        The string to prepend to all keys except for `exception_key`.
    exception_key (str):
        The key to exclude from the prepend operation. If None, no key
        will be excluded. Default is 'random_state'.

    Returns
    -------
    A new dictionary with modified keys.
    """
    new_dict = {}
    for key, value in dict_to_update.items():
        if exception_key is not None and key == exception_key:
            new_dict[key] = value
        else:
            new_dict[prepend_string + key] = value
    return new_dict


def chunks(param_list, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(param_list), n):
        yield param_list[i : i + n]


def extract_features_fn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts specific features from the 'image_features' column in the dataframe
    and returns a new dataframe with the extracted features as new columns.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas dataframe containing a column named 'image_features'.

    Returns
    -------
    pd.DataFrame
        A new pandas dataframe with the extracted features as new columns,
        placed immediately after the 'image_features' column.

    """

    def process_filename(filename: str) -> Dict[str, str]:
        """
        Processes the given filename and extracts the required features.

        Parameters
        ----------
        filename : str
            The filename from which features are to be extracted.

        Returns
        -------
        dict
            A dictionary containing the extracted features as key-value pairs.

        """
        # Remove the '_.csv' suffix from the filename
        filename = filename.replace(".csv", "")

        # Split the filename into key-value pairs
        parts = filename.split("_")

        # Initialize an empty dictionary to store the features
        features = {}

        # Loop through each part to extract key-value pairs
        for part in parts:
            key_value = part.split("-")
            key = key_value[-1]
            value = "-".join(key_value[:-1])
            features[key] = value

        return features

    # Apply the process_filename function to each value in the 'image_features' column
    extracted_features = df["image_features"].apply(process_filename)

    # Convert the extracted features into a dataframe
    features_df = pd.DataFrame(list(extracted_features))

    # Get the index of the 'image_features' column in the original dataframe
    index = df.columns.get_loc("image_features")

    # Split the original dataframe into two parts: columns before and after 'image_features'
    before = df.iloc[:, : index + 1]
    after = df.iloc[:, index + 1 :]

    # Concatenate these parts along with the extracted features to form the final dataframe
    result_df = pd.concat([before, features_df, after], axis=1)

    return result_df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder the columns in the DataFrame to keep 'ridge_mean', 'rf_mean',
    'ens_mean', 'ridge_std', 'rf_std', 'ens_std' in place, then order the
    rest of the columns in the sequence _mean, _LOCO, _std.

    :param df: DataFrame containing the columns to reorder
    :return: DataFrame with reordered columns
    """
    # Separating the columns into categories
    main_means = ["ridge_mean", "rf_mean", "ens_mean"]
    other_means = [
        col for col in df.columns if "_mean" in col and col not in main_means
    ]
    loco_columns = [col for col in df.columns if "_LOCO" in col]
    main_stds = ["ridge_std", "rf_std", "ens_std"]
    other_stds = [col for col in df.columns if "_std" in col and col not in main_stds]
    other_columns = [
        col
        for col in df.columns
        if "_mean" not in col and "_std" not in col and "_LOCO" not in col
    ]

    # Concatenating the lists in the desired order
    reordered_columns = (
        other_columns + main_means + other_means + loco_columns + main_stds + other_stds
    )

    # Reordering the DataFrame using the reordered_columns list
    return df[reordered_columns]


def contains_nan(x: List[Union[float, int]]) -> bool:
    """
    Check if a list contains any 'nan' values.

    Parameters
    ----------
    x : List[Union[float, int]]
        The list to be checked.

    Returns
    -------
    bool
        True if any elements in the list are 'nan', otherwise False.

    Examples
    --------
    >>> contains_nan([1, 2, np.nan])
    True

    >>> contains_nan([1, 2, 3])
    False
    """

    return any(pd.isna(i) for i in x)


def convert_list_columns_to_float(
    df: pd.DataFrame,
    columns_to_convert: List[str] = [
        "mean",
        "std",
        "min",
        "max",
        "sum",
        "sum_of_squares",
    ],
) -> pd.DataFrame:
    """
    Converts the specified columns of a DataFrame from strings to lists of floats.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the columns to be converted.
    columns_to_convert : List[str], optional
        List of column names to be converted, by default ["mean", "std", "min", "max", "sum", "sum_of_squares"]

    Returns
    -------
    pd.DataFrame
        DataFrame with the specified columns converted to lists of floats.

    Examples
    --------
    >>> df = pd.DataFrame({'mean': ['[1.0, 2.0]'], 'std': ['[3.0, 4.0]']})
    >>> convert_columns_to_float(df, ['mean', 'std'])
    """
    for column in columns_to_convert:
        if df[column].dtype == "object":
            # Check if the first element is a string representation of a list
            first_element = df[column].iloc[0]
            if (
                isinstance(first_element, str)
                and first_element.startswith("[")
                and first_element.endswith("]")
            ):
                df[column] = df[column].apply(
                    lambda x: list(map(float, literal_eval(x)))
                )

    return df


class VisualBasemapDataset(Dataset):
    """
    A custom Dataset that loads and transforms .tif images from specified directories.

    Specifically, it:

    - Loads only the first three bands (RGB).
    - Normalizes the pixel values by dividing by 255.
    - Optionally resizes the images to a specified size.
    - Includes metadata: 'unique_id' and 'basemap_id'.

    Parameters
    ----------
    root_dir : str
        Root directory containing multiple 'global_quarterly_*_mosaic' subdirectories.
    transform : callable, optional
        Optional transform to be applied on a sample.
    resize : tuple of int, optional
        Desired output size as (height, width). If None, no resizing is applied.
    specific_dir : str, optional
        If provided, only load images from this directory under root_dir.
    verbosity : int, optional
        Level of verbosity. Default is 0 (silent). Set to 1 to enable print statements.

    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        resize: Optional[Tuple[int, int]] = None,
        specific_dir: Optional[str] = None,
        clipped: bool = False,
        verbosity: int = 0,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.resize = resize
        self.specific_dir = specific_dir
        self.verbosity = verbosity
        self.clipped = clipped
        self.image_paths = self._gather_image_paths()

    def _gather_image_paths(self) -> List[str]:
        """
        Collects paths to all .tif files.

        If `specific_dir` is provided, only loads images from that directory.
        Otherwise, traverses the root directory and collects paths to all .tif files within subdirectories.

        Returns
        -------
        List[str]
            List of file paths to .tif images.

        Raises
        ------
        ValueError
            If the specified directory does not exist.

        """
        image_paths = []

        if self.specific_dir:
            # Use specific_dir
            dir_to_search = os.path.join(self.root_dir, self.specific_dir)
            if os.path.isdir(dir_to_search):
                for file_name in os.listdir(dir_to_search):
                    if file_name.lower().endswith(".tif"):
                        full_path = os.path.join(dir_to_search, file_name)
                        image_paths.append(full_path)
            else:
                raise ValueError(
                    f"Specified directory '{dir_to_search}' does not exist"
                )
        else:
            # Search over subdirectories as before
            for mosaic_dir in os.listdir(self.root_dir):
                mosaic_path = os.path.join(self.root_dir, mosaic_dir)
                if (
                    os.path.isdir(mosaic_path)
                    and mosaic_dir.startswith("global_")
                    and mosaic_dir.endswith("_mosaic")
                ):
                    if not self.clipped:
                        image_dir = os.path.join(mosaic_path, "basemap_quads")
                    else:
                        image_dir = mosaic_path
                    for file_name in os.listdir(image_dir):
                        if file_name.lower().endswith(".tif"):
                            full_path = os.path.join(
                                image_dir, file_name
                            )
                            image_paths.append(full_path)

        if self.verbosity > 0:
            print(f"Total images found: {len(image_paths)}")

        return image_paths

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads and returns the transformed image and its metadata at the specified index.

        Parameters
        ----------
        idx : int
            Index of the image to retrieve.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
                - 'image': Normalized (and optionally resized) image tensor.
                - 'basemap_id': Name of the basemap directory.
                - 'unique_id': Image file name without the '.tif' extension.

        Raises
        ------
        IndexError
            If the index is out of bounds.

        """
        if idx < 0 or idx >= len(self.image_paths):
            raise IndexError("Index out of bounds")

        img_path = self.image_paths[idx]

        # **1. Extract Metadata**
        # Extract 'unique_id' from file name (without extension)
        file_name = os.path.basename(img_path)
        unique_id, _ = os.path.splitext(file_name)

        # Extract 'basemap_id' from the parent directory name
        basemap_id = os.path.basename(os.path.dirname(img_path))

        # **2. Read Only the First Three Bands (RGB)**
        with rasterio.open(img_path) as src:
            rgb_bands = src.read([1, 2, 3])  # Bands are 1-indexed in rasterio

            # **3. Convert to Float32 and Normalize**
            rgb_bands = rgb_bands.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # **4. Apply Optional Transformations**
        if self.transform:
            rgb_bands = self.transform(rgb_bands)

        # **5. Convert to PyTorch Tensor**
        # Current shape: (C, H, W) -> Desired shape: (C, H, W) as Tensor
        image_tensor = torch.from_numpy(rgb_bands)

        # **6. Apply Optional Resizing**
        if self.resize:
            # torch.nn.functional.interpolate expects input of shape (N, C, H, W)
            # Add batch dimension, perform interpolation, then remove batch dimension
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=self.resize,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # **7. Prepare the Sample Dictionary**
        sample = {
            "image": image_tensor,  # Shape: (3, H, W)
            "basemap_id": basemap_id,  # String
            "unique_id": unique_id,  # String
        }

        return sample
