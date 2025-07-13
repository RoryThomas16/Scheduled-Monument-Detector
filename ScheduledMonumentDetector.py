""" The Python Implementation of the Jupyter Notebook, this is to have one streamlined file for the Scheduled Monument Detector project.
The Notebook is more of a 'breadboard' for the code, where this file intends to be the streamlined implementation"""

# Read data from DataMapWales and Google Earth Engine
from owslib.wms import WebMapService
import ee
from PIL import Image
import requests
# Standard Imports
import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
import shapely.geometry
import matplotlib.pyplot as plt
import re, random
from matplotlib.path import Path

random_seed = 42
random.seed(random_seed) # Random seed for reproducibility
# This will get us the coordinates of the Scheduled Monuments in Wales
csv_url = 'https://datamap.gov.wales/geoserver/ows?service=WFS&version=1.0.0&request=GetFeature&typename=inspire-wg%3ACadw_SAM&outputFormat=csv&srs=EPSG%3A27700'
# This is to get the data from DataMapWales
bbox = (-3.05, 51.50, -2.90, 51.65)  # Example bounding box coordinates (min-x, min-y, max-x, max-y)
wms_url = "https://datamap.gov.wales/capabilities/map/scheduled-monuments-and-sssi/?ows_service=wms"

ee.Authenticate()  # If you are running on VSCode, prompt will be at top of screen. Other IDEs, I have no idea...
ee.Initialize()

class ScheduledMonumentDetector:
    def __init__(self, csv_url, bbox, wms_url):
        self.csv_url = csv_url
        self.bbox = bbox
        self.wms_url = wms_url
        self.csv_data = None
        self.wales_bbox = None
        
    def load_csv_data(self):
        """
        Load the CSV data from the provided URL and convert it to a GeoDataFrame.
        """
        def alternating_delim_string_to_tuple_strings(s):
            # Find all numbers (floats or ints) in the string
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
            # Group every two numbers and format as a tuple string
            return [(float(nums[i]), float(nums[i+1])) for i in range(0, len(nums), 2)]

        def get_axial_extremes(coords):
            # coords: list of (x, y) tuples
            # expansion scalar: a factor to expand the bounding box by
            # returns (min_x, max_x, min_y, max_y, var_x, var_y)
            xs = [c[1] for c in coords]
            ys = [c[0] for c in coords]
            # Calculate the bounding box elements with expansion
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            return min_x, max_x, min_y, max_y, max_x - min_x, max_y - min_y

        def en_list_to_latlon(coords, lon_offset=1E-5, lat_offset=-2E-5):  # Offset applied because it slightligy shifts the coordinates up and left, which I don't want
            # Create GeoDataFrame from the list of tuples
            gdf = gpd.GeoDataFrame(geometry=[shapely.geometry.Point(x, y) for x, y in coords], crs="EPSG:27700")
            # Convert to lat/lon
            gdf = gdf.to_crs(epsg=4326)
            # Extract as list of (lat, lon) tuples
            return [(pt.x + lon_offset, pt.y + lat_offset) for pt in gdf.geometry]

        def expand_bbox(bbox, scalar=1.2):
            """
            Expands a bounding box by a given scalar.
            bbox: (lat_min, lon_min, lat_max, lon_max)
            scalar: expansion factor (e.g., 1.5 expands by 10%)
            Returns: (lat_min, lon_min, lat_max, lon_max) expanded
            """
            lat_min, lon_min, lat_max, lon_max = bbox
            lat_c = (lat_min + lat_max) / 2
            lon_c = (lon_min + lon_max) / 2
            half_lat = (lat_max - lat_min) / 2 * scalar
            half_lon = (lon_max - lon_min) / 2 * scalar
            new_lat_min = lat_c - half_lat
            new_lat_max = lat_c + half_lat
            new_lon_min = lon_c - half_lon
            new_lon_max = lon_c + half_lon
            return (new_lat_min, new_lon_min, new_lat_max, new_lon_max)

        csv_data = pd.read_csv(self.csv_url)
        polygon_data = csv_data['geom']
        # Establish empty lists to store the extremes
        x_var = []
        y_var = []
        bbox_set = []
        # Convert the polygon data (read string) to tuples and calculate extremes
        for data in polygon_data:
            new_data = alternating_delim_string_to_tuple_strings(data)
            latlon_data = en_list_to_latlon(new_data)  # Convert to lat/lon
            # Get the extremes for the new data
            x_min_temp, x_max_temp, y_min_temp, y_max_temp, x_var_temp, y_var_temp = get_axial_extremes(latlon_data)
            x_var.append(x_var_temp)
            y_var.append(y_var_temp)
            bbox_set.append(expand_bbox((y_min_temp, x_min_temp, y_max_temp, x_max_temp)))  # Store the bounding box as a tuple
        # Add the extremes to the DataFrame
        csv_data['x_var'] = x_var
        csv_data['y_var'] = y_var
        csv_data['bbox'] = bbox_set
    
    def fetch_wms_data(wms_url, width, height, bbox_method = 'auto', srs="EPSG:4326", format="image/png", version="1.3.0"):
        """
        Fetches map data from a WMS endpoint.

        Parameters:
            wms_url (str): The URL of the WMS endpoint.
            width (int): The width of the requested image in pixels.
            height (int): The height of the requested image in pixels.
            bbox_method: The method to calculate the bounding box (default is 'auto'):
                'auto' (str): Use the bounding box defined by the layer
                (tuple): Use the provided bounding box coordinates (min-x, min-y, max-x, max-y).
            srs (str): The spatial reference system (default is "EPSG:4326").
            format (str): The format of the returned image (default is "image/png").
            version (str): The WMS version to use (default is "1.3.0").

        Returns:
            bytes: The raw image data returned by the WMS service.
        """
        try:
            def get_response(layer_name, bbox):
                # Fetch the map data
                print(f"Fetching layer: {layer_name}")
                response = wms.getmap(
                    layers=[layer_name],
                    srs=srs,
                    bbox=bbox,
                    size=(width, height),
                    format=format,
                    transparent=True
                )
                return response.read()
            
            # Initialize the WMS service
            wms = WebMapService(wms_url, version=version)

            layers = list(wms.contents)
            
            # Check contents of the WMS service
            print("Available Layers: ", layers)
            
            if len(layers) == 0:
                print("No layers available in the WMS service.")
                return None
            elif len(layers) == 1:
                layer = layers[0]
                bbox = wms[layer].boundingBoxWGS84 if bbox_method == 'auto' else bbox_method
                return get_response(layer, bbox)
            elif len(layers) > 1:
                response_list = []
                for layer in layers:
                    bbox = wms[layer].boundingBoxWGS84 if bbox_method == 'auto' else bbox_method
                    response_list.append(get_response(layer,bbox))
                return response_list
            else: 
                print("An unexpected error occurred: No layers found.")
                return None
            
        except Exception as e:
            print(f"An error occurred while fetching WMS data: {e}")
            return None

    def get_welsh_data(bbox, width = 2600, height = 2600):
        raw_output = self.fetch_wms_data(wms_url, width = width, height = height, bbox_method=bbox, srs="EPSG:4326", format="image/png", version="1.3.0")
        lidar_arr = cv2.imdecode(np.frombuffer(raw_output[0], np.uint8), cv2.IMREAD_UNCHANGED)
        monuments_arr = cv2.imdecode(np.frombuffer(raw_output[1], np.uint8), cv2.IMREAD_UNCHANGED)
        return lidar_arr, monuments_arr

    def get_random_false_locations(csv_data, n=5, max_attempts=100):
        """
        Generate n random square bounding boxes within the bounds of Wales that do not overlap with any existing monument bboxes.
        Each bbox will have width=height sampled from the mean and std of x_var/y_var in csv_data.
        Returns a list of (bbox, lidar_arr, monuments_arr) tuples.
        """

        # Get the bounds of all monument bboxes using min/max of all bbox coordinates
        all_bboxes = list(csv_data['bbox'])
        all_boxes = [shapely.geometry.box(b[1], b[0], b[3], b[2]) for b in all_bboxes]  # (minx, miny, maxx, maxy)

        # Find the farthest latitudes and longitudes
        min_lat = min(b[0] for b in all_bboxes)
        max_lat = max(b[2] for b in all_bboxes)
        min_lon = min(b[1] for b in all_bboxes)
        max_lon = max(b[3] for b in all_bboxes)

        # Get mean and std for bbox size (use average of x_var and y_var for square)
        mean_size = (csv_data['x_var'].mean() + csv_data['y_var'].mean()) / 2
        std_size = (csv_data['x_var'].std() + csv_data['y_var'].std()) / 2

        results = []
        attempts = 0
        while len(results) < n and attempts < max_attempts:
            # Randomly sample square size
            size = max(1e-4, random.gauss(mean_size, std_size))

            # Randomly select a center point within Wales bounds
            center_x = random.uniform(min_lon + size/2, max_lon - size/2)
            center_y = random.uniform(min_lat + size/2, max_lat - size/2)

            # Build square bbox (lat_min, lon_min, lat_max, lon_max)
            lat_min = center_y - size/2
            lat_max = center_y + size/2
            lon_min = center_x - size/2
            lon_max = center_x + size/2
            candidate_bbox = (lat_min, lon_min, lat_max, lon_max)
            candidate_box = shapely.geometry.box(lon_min, lat_min, lon_max, lat_max)

            # Check for overlap with any monument bbox
            overlaps = any(candidate_box.intersects(b) for b in all_boxes)
            if not overlaps:
                try:
                    lidar_arr, monuments_arr = self.get_welsh_data(candidate_bbox)
                    if np.sum(lidar_arr) == 0: # cheap solution to avoid selecting the water
                        continue
                    results.append((candidate_bbox, lidar_arr, monuments_arr))
                except Exception:
                    pass  # Skip if data fetch fails
            attempts += 1

        return results