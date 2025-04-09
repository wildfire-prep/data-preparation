"""
This module contains functions to handle the process of placing an order, polling for its success,
retrieving the results and downloading them. It also helps with image processing, normalization, 
and modeling using PyTorch.

Classes:
----------
    - PlanetAPI
        - __init__
        - authenticate_to_planet
        - get_mosaics
        - set_mosaic
        - get_mosaic_id
        - get_items
        - convert_items_to_geodataframe
        - place_order
        - poll_for_success
        - get_results
        - download_results
        - organize_files
    - PlanetB
    - RCF
        - __init__
        - forward
    - MOSAIKS
        - _normalize
        - __init__

Functions:
----------
    - filter_existing_quad_ids
    - chunk_list
    - process_image
    - process_image_args
    - process_images_from_dataframe
    - get_image_stats
    - calculate_image_level_stats
    - calculate_quad_statistics
    - calculate_country_statistics
    - parse_quad_id

Author: Cullen Molitor
Email: Cullen_Molitor@bren.ucsb.edu
Date: 2023-06-12
"""

import os
import re
import sys
import json
import time
import shutil
import pathlib
import logging
import requests
import rasterio
import rasterio.features
import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
from io import BytesIO
from pyhere import here
from pathlib import Path
from rasterio.mask import mask
from multiprocessing import Pool
from rasterio.merge import merge
from collections import defaultdict
from urllib3.exceptions import ProtocolError
from requests.exceptions import ChunkedEncodingError
from typing import Dict, List, Union, Tuple, Optional, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from shapely.geometry import Polygon, mapping, box, shape, Point, GeometryCollection

geodetic = "EPSG:4326"
mercator = "EPSG:3857"
mollweide = "ESRI:54009"

# Define API URLS
data_url = "https://api.planet.com/data/v1"
basemap_url = "https://api.planet.com/basemaps/v1/mosaics"
order_url = "https://api.planet.com/compute/ops/orders/v2"


class PlanetAPI:
    """
    A Python client for interacting with the Planet Labs API.

    This client supports the interaction, polling, and downloading of data orders
    from the Planet API. An instance of this class is initialized with a user's 
    API key and API URL, which are then used to authenticate all subsequent requests made 
    with this client.

    The client relies on the `requests` library to manage HTTP communication.

    Attributes:
    -----------
        api_key (str): The Planet API key to authenticate requests.
        api_url (str): The Planet API base URL to make requests.
        session (requests.Session): A Session object from the `requests` library 
                                    used for HTTP requests. The session handles 
                                    the HTTP protocol details and is configured 
                                    with the user's API key for authentication.

    Methods:
    --------
        __init__(api_key: str, api_url: str):
            Constructs a new `PlanetAPI` instance and configures the session for 
            HTTP communication with the Planet API.

        authenticate_to_planet(api_key: str) -> requests.Session:
            Creates a new `requests` session and configures it with the user's API key.

        get_mosaics() -> List[str]:
            Retrieves the names of available mosaics from the Planet API.

        set_mosaic(mosaic_name: str) -> None:
            Retrieves the ID of a specific mosaic and sets it as an instance variable.

        get_mosaic_id(mosaic_name: str) -> str:
            Retrieves the ID of a specific mosaic.

        get_items(bbox_aoi: List[float], mosaic_id: str = None) -> List[Dict[str, Union[str, List, Dict]]]:
            Retrieves quads/items within a bounding box.

        convert_items_to_geodataframe(all_items: List[Dict[str, Union[str, List, Dict]]]) -> gpd.GeoDataFrame:
            Converts a list of items (quads) into a GeoDataFrame.

        place_order(request: dict) -> str:
            Posts a request to place a data order and returns the order URL ID.

        poll_for_success(order_url_id: str, loop_time: int, num_loops: int) -> bool:
            Polls an order URL until the order state reaches 'success' or the maximum 
            number of loops is reached.

        get_results(order_url_id: str) -> list:
            Retrieves the results of an order.

        download_results(results: list, directory: str = 'data', overwrite: bool = False, 
                        show_progress: bool = False, max_retries: int = 5) -> None:
            Downloads the results of an order.

        organize_files(base_dir: str, mosaic_name: str = "planet_medres_normalized_analytic_2020-06_2020-08_mosaic",
                    overwrite: bool = False, verbose: bool = False) -> None:
            Organizes files in a given base directory after the order results are downloaded.
    """ 
    def __init__(self, api_key: str, api_url: str) -> None:
        """
        Initializes the PlanetAPI class with a specific API key and URL.

        Parameters:
        -----------
            api_key: Planet API key
            api_url: Planet API base URL
        """
        self.api_key = api_key
        self.session = self.authenticate_to_planet(self.api_key)
        self.api_url = api_url

    @staticmethod
    def authenticate_to_planet(api_key: str) -> requests.Session:
        """
        Authenticates to the Planet API.

        Parameters:
        -----------
            api_key: Planet API key

        Returns:
        --------
            Authenticated session object
        """
        session = requests.Session()
        session.auth = (api_key, "")
        return session
        
    def get_mosaics(self, search_str: str = None) -> List[str]:
        """
        Retrieves the names of available mosaics from the Planet API.

        Parameters:
        -----------
            search_str: str, optional
                The text string to search for in the mosaic names. If provided, 
                only mosaic names containing this string will be returned.
                Defaults to None, in which case all mosaic names are returned.

        Returns:
        --------
            A list of mosaic names.
            
        Note:
        -----
            The list contains names of all mosaics available under the provided API key, 
            potentially filtered by the `search_str` parameter. The names can then be used to 
            set a specific mosaic for further processing using the `set_mosaic` method.
        """
        session = self.authenticate_to_planet(self.api_key)
        parameters = {"_page_size": 200}
        res = session.get(self.api_url, params=parameters)
        res.raise_for_status()  # raise exception if the request failed
        mosaics = res.json()

        mosaic_names = [mosaic['name'] for mosaic in mosaics['mosaics']]

        if search_str:
            mosaic_names = [name for name in mosaic_names if search_str.lower() in name.lower()]

        return mosaic_names

    def set_mosaic(self, mosaic_name: str) -> None:
        """
        Retrieves the ID of a specific mosaic and sets it as an instance variable.

        Parameters:
        -----------
            mosaic_name: Name of the mosaic to retrieve

        Note: 
        -----
            The method does not return anything, it simply sets the ID of the specified 
            mosaic as an instance variable (self.mosaic_id) which can be accessed throughout 
            the lifecycle of the PlanetAPI object.
        """
        session = self.authenticate_to_planet(self.api_key)
        parameters = {"name__is": mosaic_name}
        res = session.get(self.api_url, params=parameters)
        res.raise_for_status()  # raise exception if the request failed
        mosaic = res.json()

        if mosaic["mosaics"]:
            self.mosaic_id = mosaic["mosaics"][0]["id"]
        else:
            raise ValueError(f"Mosaic not found with the name: {mosaic_name}")
        
    def get_mosaic_id(self, mosaic_name: str) -> str:
        """
        Retrieves the ID of a specific mosaic.

        Parameters:
        -----------
            mosaic_name: Name of the mosaic to retrieve

        Returns:
        --------
            The ID of the mosaic
        """
        parameters = {
            "name__is": mosaic_name,
        }
        res = self.session.get(self.api_url, params=parameters)
        res.raise_for_status()
        mosaic = res.json()
        return mosaic["mosaics"][0]["id"]

    def get_items(
        self, bbox_aoi: List[float], mosaic_id: str = None
    ) -> List[Dict[str, Union[str, List, Dict]]]:
        """
        Retrieves quads/items within a bounding box.

        Parameters:
        -----------
            bbox_aoi: Bounding box coordinates (min x, min y, max x, max y)
            mosaic_id: ID of the mosaic to retrieve quads/items from. If None, the mosaic_id set by set_mosaic is used. Defaults to None.

        Returns:
        --------
            A list of items (quads)
        """
        if mosaic_id is None:
            if self.mosaic_id is None:
                raise ValueError("No mosaic_id provided and no mosaic_id set by set_mosaic")
            mosaic_id = self.mosaic_id

        string_bbox = ",".join(map(str, bbox_aoi))
        search_parameters = {"bbox": string_bbox, "minimal": True}
        quads_url = f"{self.api_url}/{mosaic_id}/quads"
        
        unique_items = set()

        while quads_url is not None:
            res = self.session.get(quads_url, params=search_parameters, stream=True)
            res.raise_for_status()
            quads = res.json()

            for item in quads["items"]:
                item_json_str = json.dumps(item, sort_keys=True)
                unique_items.add(item_json_str)

            # Update the quads_url to the '_next' url if it exists, else set it to None
            quads_url = quads["_links"].get("_next", None)

        return [json.loads(item_str) for item_str in unique_items]

    @staticmethod
    def convert_items_to_geodataframe(
        all_items: List[Dict[str, Union[str, List, Dict]]]
    ) -> gpd.GeoDataFrame:
        """
        Converts a list of items (quads) into a GeoDataFrame.

        Parameters:
        -----------
            all_items: A list of items (quads)

        Returns:
        --------
            GeoDataFrame containing the items
        """
        geometry_list = [
            Polygon(
                [
                    (item.get("bbox")[0], item.get("bbox")[1]),
                    (item.get("bbox")[2], item.get("bbox")[1]),
                    (item.get("bbox")[2], item.get("bbox")[3]),
                    (item.get("bbox")[0], item.get("bbox")[3]),
                ]
            )
            for item in all_items
        ]
        gdf = gpd.GeoDataFrame(all_items, geometry=geometry_list)
        gdf = gdf.drop(columns=["_links", "bbox", "percent_covered"])
        gdf.crs = geodetic
        return gdf
    
    def place_order(self, request):
        """
        Posts a request to place an order.
        
        Parameters:
        -----------
            request (dict): The order request to be placed.
        
        Returns:
        --------
            str: The order URL ID.
        """
        response = self.session.post(
            order_url,
            data=json.dumps(request),
            headers={"content-type": "application/json"},
        )
        order_id = response.json()["id"]
        print(f"Order ID: {order_id}")
        order_url_id = order_url + "/" + order_id
        return order_url_id
    
    def poll_for_success(self, order_url_id, loop_time=10, num_loops=50):
        """
        Polls the order URL until the order reaches 'success' or the maximum number of loops is reached.
        
        Parameters:
        -----------
            order_url_id (str): The URL ID of the order to poll.
            loop_time (int, optional): The time in seconds to wait between polling attempts. Defaults to 10.
            num_loops (int, optional): The maximum number of polling attempts. Defaults to 50.
            
        Returns:
        --------
            bool: True if the order state is 'success', False otherwise.
        """
        count = 0
        while count < num_loops:
            count += 1
            response = self.session.get(order_url_id).json()
            state = response["state"]
            if state == "success":
                print(state)
                return True
            elif state in ["failed", "partial"]:
                continue
            time.sleep(loop_time)
        print("Reached maximum attempts without receiving 'success' state")
        return False

    def get_results(self, order_url_id):
        """
        Retrieves the results of the order.
        
        Parameters:
        -----------
            order_url_id (str): The URL ID of the order.
        
        Returns:
        --------
            list: The results of the order.
        """
        response = self.session.get(order_url_id).json()
        return response["_links"]["results"]

    def download_results(
        self, 
        results, 
        directory='data',
        overwrite=False,
        show_progress=False, 
        max_retries=5
        ):
        """
        Downloads the results of the order.
        
        Parameters:
        -----------
            results (list): The results to be downloaded.
            directory (str, optional): The location to download files. Defaults to 'data'.
            overwrite (bool, optional): If True, overwrites any existing files with the same name. Defaults to False.
            show_progress (bool, optional): If True, shows a progress bar instead of print statements. Defaults to False.
            max_retries (int, optional): Maximum number of retries before giving up on download. Defaults to 5.
            
        Returns:
        --------
            None: just downloads to the specified path. 
        """
        results_urls = [r["location"] for r in results]
        results_names = [r["name"] for r in results]
        pbar = None

        if show_progress:
            pbar = tqdm(total=len(results_urls), desc="Downloading items")

        for url, name in zip(results_urls, results_names):
            path = pathlib.Path(os.path.join(directory, name))

            if overwrite or not path.exists():
                retries = 0
                while retries <= max_retries:
                    try:
                        r = self.session.get(url, allow_redirects=True)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        open(path, "wb").write(r.content)
                        if show_progress:
                            pbar.update()
                        break
                    except (ChunkedEncodingError, ProtocolError):
                        print(f"Download failed for {url}.\nRetrying...", end="\n")
                        retries += 1
                        if retries > max_retries:
                            print(f"Failed to download {url}\nafter {max_retries} attempts. Skipping.", end="\n")
                            break

        if show_progress:
            pbar.close()

    @staticmethod
    def organize_files(
        base_dir: str,
        mosaic_name: str = "planet_medres_normalized_analytic_2020-06_2020-08_mosaic",
        overwrite: bool = False,
        verbose: bool = False,
    ):
        """
        Organizes files in given base directory.

        Parameters:
        -----------
            base_dir (str): Base directory where order_id directories exist.
            mosaic_name (str): The name of the mosaic downloaded
            overwrite (bool, optional): Whether to overwrite files in the destination directory.
                Defaults to False.
            verbose (bool, optional): Whether to print status messages. Defaults to False.
        """
        if not os.path.isdir(base_dir):
            if verbose:
                print(f"The provided directory does not exist: {base_dir}")
            return

        # define your new folders
        basemap_quad_dir =  here(
            base_dir, 
            mosaic_name,
            "basemap_quads",
            )
        basemap_metadata_dir =  here(
            base_dir, 
            mosaic_name,
            "basemap_metadata",
            )

        # create the directories if they do not exist
        os.makedirs(basemap_quad_dir, exist_ok=True)
        os.makedirs(basemap_metadata_dir, exist_ok=True)

        # iterate over each order_id directory
        for order_id in os.listdir(base_dir):
            order_dir = os.path.join(base_dir, order_id)

            if os.path.isdir(order_dir):
                subfolder_dir =  os.path.join(order_dir, mosaic_name)

                if os.path.exists(subfolder_dir):  # check if mosaic_name subfolder exists
                    # iterate over each file in the subfolder
                    for filename in os.listdir(subfolder_dir):
                        src_file =  os.path.join(subfolder_dir, filename)

                        if filename.endswith("_quad.tif"):
                            dest_dir = basemap_quad_dir
                        else:
                            dest_dir = basemap_metadata_dir

                        dest_file = os.path.join(dest_dir, filename)

                        # only move the file if it doesn't exist in the destination directory
                        if os.path.exists(dest_file):
                            if overwrite:
                                shutil.move(src_file, dest_file)
                            elif verbose:
                                print(
                                    f"File '{filename}' already exists in the destination directory."
                                )
                        else:
                            shutil.move(src_file, dest_file)


def filter_existing_quad_ids(quad_ids_list, existing_files):
    # Extract the quad_ids from the filenames
    existing_quad_ids = [
        re.search(r"(\d+-\d+)", filename).group()
        for filename in existing_files
        if re.search(r"(\d+-\d+)", filename)
    ]

    # Filter out the ids in your list that are already present in the quad_dir
    filtered_quad_ids_list = [
        quad_id for quad_id in quad_ids_list if quad_id not in existing_quad_ids
    ]

    return filtered_quad_ids_list


def chunk_list(input_list, chunk_size=500):
    # For item i in a range that is a length of input_list,
    # create an index range for input_list of chunk_size items:
    return [
        input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)
    ]


def process_image(
    image_dict: Dict, 
    input_dir: str, 
    output_dir: str, 
    verbose: int = 0,
    overwrite: bool = False,
    visual: bool = False
) -> None:
    """
    This function processes a single image. It opens all the quads that intersect
    with the polygon, merges the images, updates the metadata, masks the merged
    image, processes the image, and then saves the processed image.

    Parameters:
    -----------
        image_dict (Dict):
            The dictionary containing the image data. 
            It should include 'quad_id', 'geometry', and 'unique_id' keys.
        input_dir (str):
            The directory path for the base map quads. 
        output_dir (str):
            The directory path for the processed images. 
        verbose (int):
            An optional argument that determines whether or not to print 
            verbose messages. Defaults to 0 (off).
        overwrite (bool):
            An optional argument that determines whether or not to overwrite 
            existing files. Defaults to False.
        visual (bool):
            An optional argument that determines whether to process the image as visual. 
            If True, the image pixel values will be kept as is with original data type. 
            Defaults to False.

    Returns:
    --------
        None: The function does not return a value. It writes the processed
              image directly to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(image_dict, pd.DataFrame):
        if len(image_dict) != 1:
            raise ValueError(
                "If a DataFrame is provided, it should contain only one row."
            )
        image_dict = image_dict.iloc[0].to_dict()

    quad_ids = image_dict["quad_id"]
    output_path = os.path.join(output_dir, f"{image_dict['unique_id']}.tif")

    if os.path.exists(output_path) and not overwrite:
        if verbose > 0:
            print(f"File {output_path} already exists. Skipping processing.", flush=True)
        return
    if verbose > 0:
        print(f"Processing: {output_path}", flush=True)

    images = [
        rasterio.open(os.path.join(input_dir, f"{quad_id}_quad.tif"))
        for quad_id in quad_ids
    ]

    merged_images, merged_transform = merge(images)

    merged_meta = images[0].meta.copy()
    merged_meta.update(
        {
            "driver": "GTiff",
            "height": merged_images.shape[1],
            "width": merged_images.shape[2],
            "transform": merged_transform,
        }
    )

    with rasterio.open(BytesIO(), "w", **merged_meta) as merged_dst:
        merged_dst.write(merged_images)

        out_image, out_transform = mask(
            merged_dst, [mapping(image_dict["geometry"])], crop=True, all_touched=True
        )

    if visual:
        out_meta = merged_dst.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "count": out_image.shape[0],
            }
        )

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

        for img in images:
            img.close()

        return
    
    out_meta = merged_dst.meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "count": out_image.shape[0],
        }
    )

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

    for img in images:
        img.close()


def process_image_args(
    args: Tuple[dict, Union[str, Path], Union[str, Path], int, bool, bool]
) -> None:
    """
    Wrapper function to allow process_image to be called with a single tuple of arguments,
    facilitating use with multiprocessing.
    """
    return process_image(*args)


def process_images_from_dataframe(
    image_df: pd.DataFrame, 
    input_dir: str, 
    output_dir: str, 
    verbose: int = 0,
    num_workers: int = 4,
    overwrite: bool = False,
    visual: bool = False
) -> None:
    """
    This function processes multiple images based on a DataFrame of image information. 
    Each row in the DataFrame should correspond to one image to be processed.

    Parameters:
    -----------
        image_df (pd.DataFrame): 
            A DataFrame where each row contains the image data for a single image. 
            The DataFrame should include 'quad_id', 'geometry', and 'unique_id' columns.
        input_dir (str): 
            The directory path for the base map quads.
        output_dir (str): 
            The directory path for the processed images. 
        verbose (int):
            An optional argument that determines whether or not to print verbose messages. 
            Defaults to 0 (off).
        num_workers (int):
            The number of worker processes to use. If this is 0 then the current process will be used to compute the results. 
            If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. 
            For n > 1, (n-1) processes are being used. Defaults to 4.
        overwrite (bool):
            An optional argument that determines whether or not to overwrite existing files. 
            Defaults to False.
        visual (bool):
            An optional argument that determines whether to process the image as visual. 
            If True, the image pixel values will be kept as is with original data type. 
            Defaults to False.

    Returns:
    --------
        None: The function does not return a value. It writes the processed
              images directly to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    args_list = [
        (row.to_dict(), input_dir, output_dir, verbose, overwrite, visual)
        for _, row in image_df.iterrows()
    ]

    with Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                process_image_args, args_list
            ), total=len(args_list), desc="Processing images"
        ):
            pass


def get_image_stats(unique_id: str, directory: str, channels):
    """
    Get statistical information for the specified image and channels.

    Parameters
    ----------
    unique_id : str
        The unique ID of the image.
    directory : str
        The directory containing the image.
    channels : list of int, optional
        List of channels to consider for statistics. If None, all channels will be considered.

    Returns
    -------
    list or None
        A list containing unique ID, count, mean, standard deviation, minimum, maximum, sum,
        and sum of squares for the channels. Returns None if an exception occurs.

    Examples
    --------
    >>> get_image_stats("image1", "./images", [0, 2])
    ['image1', 1024, [0.5, 0.6], [0.1, 0.2], [0, 0], [1, 1], [512, 614], [256, 307]]

    """
    try:
        filepath = os.path.join(directory, f"{unique_id}.tif")
        with rasterio.open(filepath) as src:
            img = src.read()

            if channels is not None:
                img = img[channels]
        img = img.astype("float32")
        return [
            unique_id,
            np.prod(img.shape[1:]),
            list(np.mean(img, axis=(1, 2))),
            list(np.std(img, axis=(1, 2))),
            list(np.min(img, axis=(1, 2))),
            list(np.max(img, axis=(1, 2))),
            list(np.sum(img, axis=(1, 2))),
            list(np.sum(img**2, axis=(1, 2))),
        ]

    except Exception as e:
        print(f"Error processing {unique_id}: {e}")
        return None


def calculate_image_level_stats(
    df: pd.DataFrame,
    directory: str,
    channels: Optional[List[int]] = None,
    id_col: str = "unique_id",
    num_cores: Optional[int] = None,
    output_file: str = "results.csv"
) -> pd.DataFrame:
    """
    Calculate statistical information for multiple images and save/append results to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the unique IDs of the images to be processed.
    directory : str
        The directory containing the image files.
    channels : list of int, optional
        List of channels to consider for statistics. If None, all channels will be considered.
    id_col : str, optional
        Column name in `df` that contains the unique IDs of the images, default is "unique_id".
    num_cores : int, optional
        Number of cores to use for parallel processing. If None, all available cores will be used.
    output_file : str, optional
        Name of the output CSV file where results will be saved. Existing results will not be overwritten.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated image-level statistics.

    Examples
    --------
    >>> df = pd.DataFrame({"unique_id": ["img1", "img2"]})
    >>> calculate_image_level_stats(df, "./images", [0, 1])
    """
    # If the CSV file exists, load it and filter out already processed images
    if os.path.exists(output_file):
        processed_df = pd.read_csv(output_file)
        processed_ids = processed_df[id_col].values
        df = df[~df[id_col].isin(processed_ids)]

    unique_ids = df[id_col]

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(
            tqdm(
                executor.map(
                    get_image_stats,
                    unique_ids,
                    [directory] * len(unique_ids),
                    [channels] * len(unique_ids),
                ),
                total=len(unique_ids),
            )
        )

    # Remove None results from the error handling
    results = [result for result in results if result is not None]

    columns = [id_col, "n", "mean", "std", "min", "max", "sum", "sum_of_squares"]

    df_results = pd.DataFrame(results, columns=columns)

    # Append to the existing CSV file if it exists
    if os.path.exists(output_file):
        df_results.to_csv(output_file, mode="a", header=False, index=False)
        # Re-read the entire CSV to return the full dataset
        df_results = pd.read_csv(output_file)
    else:
        df_results.to_csv(output_file, index=False)

    return df_results


def calculate_quad_statistics(
    df: pd.DataFrame, groupby: str = "unique_id"
) -> pd.DataFrame:
    """
    Calculate statistical values for quadrants in a DataFrame.

    This function computes the mean, standard deviation, minimum, and maximum values for each unique ID in the dataset.
    The calculations are performed over four channels, so the resulting DataFrame contains lists of results for each statistic.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame, containing columns 'unique_id', 'mean', 'std', 'min', 'max', 'n', 'sum', and 'sum_of_squares'.
        Here, 'n' is the count, 'sum' is the sum of values, and 'sum_of_squares' is the sum of squared values for each channel.
    groupby : str, optional
        The name of the column to group by, default is 'unique_id'.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the calculated statistics for each unique ID, with columns ["id", "mean", "std", "min", "max"].

    Example
    --------
    >>> df = pd.DataFrame({...}) # Create a DataFrame with appropriate columns
    >>> result = calculate_quad_statistics(df)
    """

    # Initialize the results DataFrame
    results = pd.DataFrame(columns=["id", "mean", "std", "min", "max"])

    # Iterate through unique IDs
    for unique_id in df[groupby].unique():
        sub_df = df[df[groupby] == unique_id]

        # If there's only one row, the stats remain the same
        if sub_df.shape[0] == 1:
            result = (
                unique_id,
                sub_df["mean"].values[0],
                sub_df["std"].values[0],
                sub_df["min"].values[0],
                sub_df["max"].values[0],
            )
            results.loc[len(results)] = result
            continue

        # Calculate the aggregated stats for each of the 4 channels
        mean_result = []
        std_result = []
        min_result = []
        max_result = []

        for i in range(4):
            n = sub_df["n"].sum()
            total_sum_of_sums = sum(x[i] for x in sub_df["sum"])
            total_sum_of_squares = sum(x[i] for x in sub_df["sum_of_squares"])
            mean = total_sum_of_sums / n
            var = (total_sum_of_squares / n) - (mean**2)
            std = np.sqrt(var)
            min_val = min(x[i] for x in sub_df["min"])
            max_val = max(x[i] for x in sub_df["max"])

            mean_result.append(mean)
            std_result.append(std)
            min_result.append(min_val)
            max_result.append(max_val)

        result = (unique_id, mean_result, std_result, min_result, max_result)
        results.loc[len(results)] = result

    return results


def calculate_country_statistics(dataset: iter, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean, standard deviation, min, max, and 99th percentile max for each country and the overall dataset.

    Parameters
    ----------
    dataset : iterable
        An iterable collection of dictionaries, where each dictionary contains 'image' and 'image_name'.
        'image' is a tensor representing the image and 'image_name' is a string following the pattern 'unique_id'.
    
    labels_df : pd.DataFrame
        A DataFrame containing the label dataset with columns like 'unique_id', 'country', etc.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing computed statistics.

    Examples
    --------
    >>> dataset = [
    ...     {"image": np.array([[1, 2], [3, 4]]), "unique_id": 'lat_8--895__lon_-13--135'},
    ...     {"image": np.array([[5, 6], [7, 8]]), "unique_id": 'lat_8--235__lon_-12--975'},
    ...     ...
    ... ]
    >>> labels_df = pd.DataFrame({
    ...     'unique_id': ['lat_8--895__lon_-13--135', 'lat_8--235__lon_-12--975'],
    ...     'country': ['SLE', 'SLE']
    ... })
    >>> result = calculate_country_statistics(dataset, labels_df)
    """
    
    # Initialize statistics and max values dictionaries
    stats = defaultdict(
        lambda: {
            "sum": np.zeros(4),
            "sum_square": np.zeros(4),
            "count": 0,
            "min": np.inf * np.ones(4),
            "max": -np.inf * np.ones(4),
        }
    )
    max_values = defaultdict(list)
    stats["dataset"] = {
        "sum": np.zeros(4),
        "sum_square": np.zeros(4),
        "count": 0,
        "min": np.inf * np.ones(4),
        "max": -np.inf * np.ones(4),
    }
    max_values["dataset"] = []

    # Create a dictionary for faster lookup of country by unique_id
    id_to_country = dict(zip(labels_df['unique_id'], labels_df['country']))

    # Iterate through the dataset
    for inputs in tqdm(dataset):
        img = inputs["image"].cpu().numpy().reshape(4, -1)  # Convert to numpy and reshape
        unique_id = inputs["image_name"].split('.')[0]  # Assuming the image_name is 'unique_id.tif'

        # Look up country code for this unique_id
        country_code = id_to_country.get(unique_id, 'UNKNOWN')

        # Update statistics for country-specific and overall
        for key in ["dataset", country_code]:
            stats[key]["sum"] += img.sum(axis=1)
            stats[key]["sum_square"] += (img ** 2).sum(axis=1)
            stats[key]["count"] += img.shape[1]
            stats[key]["min"] = np.minimum(stats[key]["min"], img.min(axis=1))
            stats[key]["max"] = np.maximum(stats[key]["max"], img.max(axis=1))

        # Get the max values for each band
        img_max = img.max(axis=1)

        # Add to country-specific and overall max values
        max_values[country_code].extend(img_max.tolist())
        max_values["dataset"].extend(img_max.tolist())

    # Prepare the results
    results = []
    for code, stat in stats.items():
        mean = stat["sum"] / stat["count"]
        var = (stat["sum_square"] / stat["count"]) - (mean ** 2)
        std = np.sqrt(var)

        # Calculate 99th percentile of the max values for each band
        percentile_99th = [np.percentile(max_values[code][i::4], 99) for i in range(4)]

        results.append(
            {
                "id": code,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "min": stat["min"].tolist(),
                "max": stat["max"].tolist(),
                "99th percentile max": percentile_99th,
            }
        )

    # Create a DataFrame to organize the results
    df = pd.DataFrame(results)
    return df


def parse_quad_id(quad_id_str: str) -> list:
    """
    Parse a quadrilateral ID string by removing brackets and splitting by commas.

    This function takes a string representing quadrilateral IDs, removes brackets,
    replaces single quotes, and splits the string by commas to return a list of IDs.

    Parameters
    ----------
    quad_id_str : str
        A string containing quadrilateral IDs in the form "['id1', 'id2', ...]".

    Returns
    -------
    list
        A list of quadrilateral IDs as strings.

    Examples
    --------
    >>> quad_id_str = "['id1', 'id2', 'id3']"
    >>> parse_quad_id(quad_id_str)
    ['id1', 'id2', 'id3']
    """

    # Removing brackets and splitting by comma
    return quad_id_str.strip("[]").replace("'", "").split(", ")


class PlanetBasemapsAPI:
    """
    A Python client focused exclusively on Planet's Basemaps API for:
      - Listing mosaics
      - Retrieving quads
      - Downloading quads directly
    """

    PAGE_SIZE = 200

    def __init__(self, api_key: str, basemaps_url: str = basemap_url) -> None:
        """
        Initialize the Basemaps API client.

        Parameters
        ----------
        api_key : str
            Planet API key used for authentication.
        basemaps_url : str
            Base URL for the Planet Basemaps API.
        """
        self.api_key = api_key
        self.basemaps_url = basemaps_url
        self.session = self._authenticate_to_planet(api_key)
        self.mosaic_id = None

    @staticmethod
    def _authenticate_to_planet(api_key: str) -> requests.Session:
        """
        Creates an authenticated session.

        Parameters
        ----------
        api_key : str
            Planet API key

        Returns
        -------
        requests.Session
            Session authenticated with the given API key.
        """
        session = requests.Session()
        session.auth = (api_key, "")
        return session

    def check_authentication_status(self) -> str:
        """
        Check that we can access the basemaps endpoint (authentication test).

        Returns
        -------
        str
            A message indicating authentication status.
        """
        try:
            url = f"{self.basemaps_url}/mosaics"
            resp = self.session.get(url, params={"_page_size": 1})  # quick check
            resp.raise_for_status()
            return "Authentication successful."
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return "Authentication failed: Unauthorized."
            else:
                return f"Authentication check failed with status code: {e.response.status_code}"
        except Exception as e:
            return f"An error occurred: {e}"

    def get_mosaics(self, search_str: str = None) -> List[str]:
        """
        Retrieves the names of available mosaics from the Basemaps API.

        Parameters
        ----------
        search_str : str, optional
            If provided, return only mosaic names containing this substring (case-insensitive).

        Returns
        -------
        list of str
            Names of mosaics matching the (optional) search.
        """
        url = f"{self.basemaps_url}/mosaics"
        params = {"_page_size": self.PAGE_SIZE}
        mosaic_names = []

        # Paginate through all mosaics
        while url:
            resp = self.session.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            for mosaic in data.get("mosaics", []):
                mosaic_names.append(mosaic["name"])

            url = data["_links"].get("_next")

        if search_str:
            import re
            pattern = re.compile(search_str, re.IGNORECASE)
            mosaic_names = [name for name in mosaic_names if pattern.search(name)]
        return mosaic_names

    def get_mosaic_id(self, mosaic_name: str) -> str:
        """
        Retrieves the ID of a specific mosaic given its name.

        Parameters
        ----------
        mosaic_name : str
            Exact name of the mosaic to find.

        Returns
        -------
        str
            The mosaic ID.

        Raises
        ------
        ValueError
            If no mosaic with that name is found.
        """
        url = f"{self.basemaps_url}/mosaics"
        params = {
            "name__is": mosaic_name,
            "_page_size": self.PAGE_SIZE,
        }

        while url:
            resp = self.session.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            mosaics = data.get("mosaics", [])
            if mosaics:
                return mosaics[0]["id"]

            url = data["_links"].get("_next")

        raise ValueError(f"No mosaic found with name '{mosaic_name}'.")

    def set_mosaic(self, mosaic_name: str) -> None:
        """
        Sets the internal mosaic_id attribute based on provided mosaic name.

        Parameters
        ----------
        mosaic_name : str
            The mosaic name to retrieve and store as self.mosaic_id.

        Raises
        ------
        ValueError
            If no mosaic with that name is found.
        """
        self.mosaic_id = self.get_mosaic_id(mosaic_name)

    def get_items(
        self,
        bbox_aoi: List[float],
        mosaic_id: str = None,
    ) -> List[Dict[str, Union[str, List, Dict]]]:
        """
        Retrieves basemap quads/items within a bounding box.

        Parameters
        ----------
        bbox_aoi : list of float
            [minX, minY, maxX, maxY] bounding box (EPSG:4326).
        mosaic_id : str, optional
            The mosaic ID. If not provided, uses self.mosaic_id.

        Returns
        -------
        list of dict
            List of quad metadata dicts (each includes _links.download).
        """
        if mosaic_id is None:
            if not self.mosaic_id:
                raise ValueError(
                    "No mosaic_id provided or set. Call set_mosaic() first."
                )
            mosaic_id = self.mosaic_id

        quads_url = f"{self.basemaps_url}/mosaics/{mosaic_id}/quads"
        params = {
            "bbox": ",".join(map(str, bbox_aoi)),
            "minimal": True,
            "_page_size": self.PAGE_SIZE,
        }

        all_quads = []
        while quads_url:
            resp = self.session.get(quads_url, params=params)
            resp.raise_for_status()
            data = resp.json()

            items = data["items"]
            all_quads.extend(items)

            quads_url = data["_links"].get("_next")

        # Remove duplicates if they appear
        unique_quads = {q["id"]: q for q in all_quads}
        return list(unique_quads.values())

    @staticmethod
    def convert_items_to_geodataframe(
        all_items: List[Dict[str, Union[str, List, Dict]]],
    ) -> gpd.GeoDataFrame:
        """
        Converts a list of quad dicts into a GeoDataFrame.

        Parameters
        ----------
        all_items : list of dict
            The Basemap quads metadata.

        Returns
        -------
        geopandas.GeoDataFrame
            A GDF of quads, with geometry from 'bbox'.
        """
        geometry_list = []
        for item in all_items:
            # item['bbox'] = [minX, minY, maxX, maxY]
            minx, miny, maxx, maxy = item["bbox"]
            geometry_list.append(
                Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
            )

        gdf = gpd.GeoDataFrame(all_items, geometry=geometry_list, crs=geodetic)
        # Optionally drop unwanted columns
        drop_cols = ["_links", "bbox", "percent_covered"]
        for col in drop_cols:
            if col in gdf.columns:
                gdf.drop(columns=[col], inplace=True, errors="ignore")

        return gdf

    def download_quads(
        self,
        quads: List[Dict],
        directory: str = "data",
        overwrite: bool = False,
        log_interval: int = 100,
        logger=None,
    ) -> None:
        """
        Downloads the quads directly from the Basemaps API.
        Each quad dict should include "_links['download']" from get_items().

        Parameters
        ----------
        quads : list of dict
            Quad metadata (each item should have 'id' and '_links.download').
        directory : str, optional
            Directory where files are saved.
        overwrite : bool, optional
            If True, overwrite existing files.
        log_interval : int, optional
            Log progress after every N downloads.
        """
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        total_quads = len(quads)
        download_count = 0
        skip_count = 0

        for i, quad in enumerate(quads):
            quad_id = quad["id"]
            download_url = quad["_links"]["download"]  # direct TIF link

            # Output filename: e.g., <quad_id>.tif
            quad_name = f"{quad_id}_quad.tif"
            out_path = pathlib.Path(directory) / quad_name

            # Skip existing files if not overwriting
            if out_path.exists() and not overwrite:
                skip_count += 1
                continue

            # Create folder if needed
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Download the quad
            try:
                resp = self.session.get(download_url, stream=True)
                resp.raise_for_status()

                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

                download_count += 1

                # Log progress at specified intervals
                if (i + 1) % log_interval == 0 or i == total_quads - 1:
                    logger.info(
                        f"Progress: {i + 1}/{total_quads} quads processed ({download_count} downloaded, {skip_count} skipped)"
                    )

            except requests.exceptions.RequestException as e:
                logger.info(f"Failed to download {quad_id} -> {e}")
                continue

        logger.info(
            f"Download complete. Total: {download_count} downloaded, {skip_count} skipped"
        )

