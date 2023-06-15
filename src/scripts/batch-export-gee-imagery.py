
import geopandas as gpd
import pandas as pd
import ee
import folium
import shapely.geometry
from shapely.geometry import shape
from matplotlib import pyplot as plt
from IPython.display import display

ee.Initialize(project='gsl-monitoring')

gdf = gpd.read_file('../../data/definitions/preliminary-roi.geojson')
polygon_geojson = shapely.geometry.mapping(gdf.loc[0, 'geometry'])
roi = ee.Geometry.Polygon(polygon_geojson['coordinates'])

# Define the SAR image collection.
collection = ee.ImageCollection('COPERNICUS/S1_GRD')\
               .filterBounds(roi)\
               .filterDate('2016-09-03', '2016-11-30')\
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
               .filter(ee.Filter.eq('instrumentMode', 'IW'))\
               .select(['VV', 'angle'])

image_ids = [
    'COPERNICUS/S1_GRD/S1A_IW_GRDH_1SSV_20161118T133343_20161118T133408_013997_0168E2_0E34',
    'COPERNICUS/S1_GRD/S1A_IW_GRDH_1SSV_20161025T133344_20161025T133409_013647_015DFD_D6D4',
    'COPERNICUS/S1_GRD/S1A_IW_GRDH_1SSV_20161001T133344_20161001T133409_013297_015303_46D8',
    'COPERNICUS/S1_GRD/S1A_IW_GRDH_1SSV_20160914T012714_20160914T012739_013042_014AAB_8E21',
    'COPERNICUS/S1_GRD/S1A_IW_GRDH_1SSV_20160907T133338_20160907T133403_012947_0147A1_6662',
    'COPERNICUS/S1_GRD/S1A_IW_GRDH_1SSV_20160814T133337_20160814T133402_012597_013BFD_B0DF',
    ]

collection.size().getInfo()
collection.getInfo()
collection.toList(collection.size()).get(2).getInfo()

# ------------------------------------------------------------
# functions

def get_image_collection(image_ids):
    def get_images(image_id):
        return ee.Image(image_id)

    images = ee.ImageCollection.fromImages(list(map(get_images, image_ids)))
    return images

def lee_sigma(image, look_window, sigma_v):
    """Applies a Lee Sigma filter to an image."""
    image = image.toFloat()
    
    mean = image.reduceNeighborhood(
        ee.Reducer.mean(), ee.Kernel.square(look_window))

    variance = image.reduceNeighborhood(
        ee.Reducer.variance(), ee.Kernel.square(look_window))

    sigma = variance.sqrt().divide(mean)

    return image.where(sigma.gt(sigma_v), mean)

def apply_speckle_filter(images, size=7, sigma_v=0.025):
    return images.map(lambda img: lee_sigma(img, size, sigma_v))

def to_linear(img):
    # from db to linear
    linear_vv = ee.Image(10.0).pow(img.select(['VV']).divide(10.0))
    img = img.addBands(linear_vv, overwrite=True).select(['VV','angle'])
    # img = img.addBands(linear_vv, overwrite=True).select(['VV'])
    return img

def convert_to_linear_units(images):
    return images.map(to_linear)

def min_max_normalize(img, roi):
    stats = img.reduceRegion(reducer=ee.Reducer.minMax(), geometry=roi, maxPixels=1e12)
    min_value = ee.Number(stats.get('VV_min'))
    max_value = ee.Number(stats.get('VV_max'))
    return img.subtract(min_value).divide(max_value.subtract(min_value))

def normalize_collection(images, roi):
    return images.map(lambda img: min_max_normalize(img, roi))

from math import pi
def incidence_angle_correction(img):
    """A function to compute the incidence angle normalization."""
    # Convert the angle to radians.
    angle = img.select('angle').multiply(pi / 180)
    corr = img.select('VV').divide(angle.cos())
    return corr.copyProperties(img, img.propertyNames())



def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles = map_id_dict['tile_fetcher'].url_format,
        attr = "Map Data © Google Earth Engine",
        name = name,
        overlay = True,
        control = True
    ).add_to(self)

def show_map(image, roi, vis_params):
    folium.Map.add_ee_layer = add_ee_layer
    my_map = folium.Map(location=[roi.centroid().coordinates().get(1).getInfo(), 
                                  roi.centroid().coordinates().get(0).getInfo()], zoom_start=8)
    my_map.add_ee_layer(image, vis_params, 'Normalized SAR Image')
    my_map.add_child(folium.LayerControl())
    return my_map

# ------------------------------------------------------------
# process

collection = get_image_collection(image_ids)
linear_collection = convert_to_linear_units(collection)
filtered_collection = apply_speckle_filter(linear_collection)
# corrected_collection = filtered_collection.map(incidence_angle_correction)
normalized_collection = normalize_collection(filtered_collection, roi)

# ------------------------------------------------------------
# visualize

vis_params = {
    # 'min': -25,
    # 'max': -5,
    'min':  0,
    'max':  1,
    'palette': ['#1b63e0', 'white', '#e0981b']
}

image = ee.Image(normalized_collection.toList(normalized_collection.size()).get(2)).select('VV')
my_map = show_map(image, roi, vis_params)
display(my_map)

# ------------------------------------------------------------
# export

# for i in range(normalized_collection.size().getInfo()):
for i in range(3):
    image = ee.Image(normalized_collection.toList(normalized_collection.size()).get(i)).select('VV')
    id = image.id().getInfo()
    task = ee.batch.Export.image.toDrive(
        image = image,
        description = id,
        region = roi.getInfo()['coordinates'],
        maxPixels = 1e12,
        # scale = 10,
        dimensions = '15000',
        fileFormat='GeoTIFF',
        crs = 'EPSG:6341'
    )
    task.start()


task.status()

# list tasks
tasks = ee.batch.Task.list()

# how many tasks
print(len(tasks))

# cancel all tasks
for task in tasks:
    print(task.status())

image.getInfo()


