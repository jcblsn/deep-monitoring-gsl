
import geopandas as gpd
import pandas as pd
import ee
import folium
import shapely.geometry
from shapely.geometry import shape

# ee.Authenticate()
ee.Initialize(project='gsl-monitoring')

# ----------------------------- define region of interest -----------------------------

gdf = gpd.read_file('../../data/definitions/preliminary-roi.geojson')
polygon_geojson = shapely.geometry.mapping(gdf.loc[0, 'geometry'])
roi = ee.Geometry.Polygon(polygon_geojson['coordinates'])

# create image collection
collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
    .filterBounds(roi) \
    .filterDate('2014-01-01', '2023-05-30') \
    .select('VV')

# collection = ee.ImageCollection('COPERNICUS/S2')\
#     .filterDate('2014-01-01', '2023-05-30') \
#     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 3)) \
#     .filterBounds(roi)

collection.size().getInfo()

# ----------------------------- explore image collection -----------------------------

# save collection info

collection_info = collection.getInfo()
features = collection_info['features']
collection_info_df = pd.json_normalize(features)

with open('../../data/gee/gee-sentinel-1-vv-collection-info.txt', 'w') as f:
    f.write(str(collection_info))
collection_info_df.to_csv('../../data/gee/gee-sentinel-1-vv-collection-info.csv')

# get roi intersection with each image

collection_size = collection.size().getInfo()
image_list = collection.toList(collection.size())
metadata = []

# initialize
# with open('../data/gee/gee-sentinel-1-vv-metadata.csv', 'w') as f:
with open('../../data/gee/gee-sentinel-2-metadata.csv', 'w') as f:
    f.write('id,date,roi_coverage\n')

for i in range(collection_size): 

    image = ee.Image(image_list.get(i))
    
    id = image.id().getInfo()
    date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()

    image_geom = ee.Geometry(image.geometry()).getInfo()
    roi_geom = roi.getInfo()

    image_shape = shape(image_geom)
    roi_shape = shape(roi_geom)
    roi_coverage = image_shape.intersection(roi_shape).area / roi_shape.area
    
    metadata.append({
        'i': i,
        'id': id,
        'date': date,
        'roi_coverage': roi_coverage,
    })

    # Write to csv
    # with open('../data/gee/gee-sentinel-1-vv-metadata.csv', 'a') as f:
    with open('../../data/gee/gee-sentinel-2-metadata.csv', 'a') as f:
        f.write(f'{id},{date},{roi_coverage}\n')
