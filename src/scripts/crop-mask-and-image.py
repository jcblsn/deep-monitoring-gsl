import os
import numpy as np
import geopandas as gpd
import folium
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt

# -----------------------------
# paths

dir_model = '../../models/lidar-mask'
dir_mask = dir_model + '/data/ground-truth/'
dir_image = '../../data/gee/imagery/sentinel-1/'
image = 'S1A_IW_GRDH_1SSV_20161118T133343_20161118T133408_013997_0168E2_0E34'
path_image = dir_image + image + "_cropped.tif"
path_ref = '../../data/gee/imagery/sentinel-1/S1A_IW_GRDH_1SSV_20161001T133344_20161001T133409_013297_015303_46D8_cropped.tif'

# -----------------------------
# define polygon boundary

polygons = gpd.read_file('../../data/definitions/GreatSaltLake_Breaklines_Polygons.geojson')

dst_crs = 'EPSG:6341' # consider {'init': 'epsg:6341'}
polygons = polygons.to_crs(dst_crs)

combined = polygons.unary_union
simplified = combined.buffer(12500)

# -----------------------------
# read in image and ref

with rasterio.open(path_image) as src1:
    img1 = src1.read()
    transform1 = src1.transform
    crs1 = src1.crs

with rasterio.open(path_ref) as src2:
    img2 = src2.read()
    transform2 = src2.transform
    crs2 = src2.crs

# -----------------------------
# check crs

print(crs1, crs2)


# -----------------------------
# crop

# bounds1 = rasterio.transform.array_bounds(img1.shape[1], img1.shape[2], transform1)
# bounds2 = rasterio.transform.array_bounds(img2.shape[1], img2.shape[2], transform2)

# xmin = max(bounds1[0], bounds2[0])
# ymin = max(bounds1[1], bounds2[1])
# xmax = min(bounds1[2], bounds2[2])
# ymax = min(bounds1[3], bounds2[3])

# crop_window = [(xmin, ymin), (xmax, ymax)]

# cropped_img1, _ = mask(img1, [crop_window], crop=True)
# cropped_img2, _ = mask(img2, [crop_window], crop=True)

# -----------------------------

def crop_geotiff(geotiff_path, shapes, out_path):
    with rasterio.open(geotiff_path) as src:
        out_image, out_transform = mask(src, shapes, crop=True, nodata=1.01)
        out_meta = src.meta
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "nodata": 1.01
    })

    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_image)

def visualize_geotiff(path):
    with rasterio.open(path) as src:
        band1 = src.read(1)
        print(src.shape)

    plt.imshow(band1, cmap='pink')
    plt.show()

shapes = [simplified] 


# crop_geotiff(
#     dir_mask + "lidar-mask.tif",
#     shapes, 
#     dir_mask + "lidar-mask-cropped.tif"
# )

crop_geotiff(
    dir_image + image + ".tif", 
    shapes, 
    path_image)

visualize_geotiff(dir_mask + "lidar-mask-cropped.tif")
visualize_geotiff(dir_image + image + "_cropped.tif")
visualize_geotiff(path_ref)