
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt

polygons = gpd.read_file('../../data/definitions/GreatSaltLake_Breaklines_Polygons.geojson')

dst_crs = 'EPSG:6341' # consider {'init': 'epsg:6341'}
polygons = polygons.to_crs(dst_crs)

imagery_path = '../../data/gee/imagery/'
imagery_name = 'S1A_IW_GRDH_1SSV_20161001T133344_20161001T133409_013297_015303_46D8'
imagery_filename = imagery_name + '.tif'

with rasterio.open(imagery_path + imagery_filename) as src:
    transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    
    with rasterio.open(imagery_path + imagery_name + "_reprojected" + ".tif", 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)

# Plot to confirm
with rasterio.open(imagery_path + imagery_name + "_reprojected" + ".tif") as src:
    fig, ax = plt.subplots()
    ax.imshow(src.read(1), cmap='pink')
    polygons.plot(ax=ax, facecolor='none', edgecolor='blue')
    plt.show()
