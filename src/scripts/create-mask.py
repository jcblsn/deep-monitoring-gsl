import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt

polygons = gpd.read_file('../../data/definitions/GreatSaltLake_Breaklines_Polygons.geojson')

dst_crs = 'EPSG:6341' # consider {'init': 'epsg:6341'}
polygons = polygons.to_crs(dst_crs)

# --- Create mask ---

with rasterio.open("../../data/gee/imagery/sentinel-1/S1A_IW_GRDH_1SSV_20161001T133344_20161001T133409_013297_015303_46D8.tif") as src:
    
    out_meta = src.meta.copy()
    band = src.read(1)

    mask = np.zeros(src.shape, dtype='uint8')

    for _, row in polygons.iterrows():
        geom = row.geometry

        rasterized = features.rasterize([(geom, 1)],
                                 out_shape=src.shape,
                                 transform=src.transform,
                                 fill=0,
                                 all_touched=True,
                                 dtype='uint8')

        mask = np.logical_or(mask, rasterized)
        
    mask = mask.astype('uint8')

    out_meta.update({"driver": "GTiff",
                    "height": mask.shape[0],
                    "width": mask.shape[1],
                    "count": 1,
                    "dtype": rasterio.uint8})

    with rasterio.open("../../data/lidar-mask.tif", "w", **out_meta) as dest:
        dest.write(mask.astype(rasterio.uint8), 1)

with rasterio.open('../../models/lidar-mask/data/ground-truth/lidar-mask-cropped.tif') as src:
    fig, ax = plt.subplots()
    ax.imshow(src.read(1), cmap='pink')
    plt.show()

with rasterio.open('../../data/gee/imagery/sentinel-1/S1A_IW_GRDH_1SSV_20161001T133344_20161001T133409_013297_015303_46D8_cropped.tif') as src:
    fig, ax = plt.subplots()
    ax.imshow(src.read(1), cmap='pink')
    plt.show()
