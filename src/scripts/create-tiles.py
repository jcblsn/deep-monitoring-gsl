import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from rasterio.plot import show
from PIL import Image
import tensorflow as tf

# ---

dir_model = '../../models/lidar-mask'
dir_image = '../../data/gee/imagery/sentinel-1/'
dir_mask = dir_model + '/data/ground-truth/'
path_mask = dir_mask + 'lidar-mask-cropped.tif'
version = '/20161001'
# version = '/20161025'
# version = '/20161118'
dir_tiles = dir_model + '/data/input/tiles' + version
dir_partition = dir_tiles + '/partition'
dir_overlap = dir_tiles + '/overlap'
dir_sampled = dir_tiles + '/sampled'
image = 'S1A_IW_GRDH_1SSV_20161001T133344_20161001T133409_013297_015303_46D8'
# image = 'S1A_IW_GRDH_1SSV_20161025T133344_20161025T133409_013647_015DFD_D6D4'
# image = 'S1A_IW_GRDH_1SSV_20161118T133343_20161118T133408_013997_0168E2_0E34'
path_image = dir_image + image + "_cropped.tif"

if not os.path.exists(dir_tiles):
    os.makedirs(dir_tiles)
if not os.path.exists(dir_partition):
    os.makedirs(dir_partition)
if not os.path.exists(dir_overlap):
    os.makedirs(dir_overlap)
if not os.path.exists(dir_sampled):
    os.makedirs(dir_sampled)

# ---

def read_data_in_window(src, window):
    return src.read(window=window)

def save_image_tile(data, out_path, meta):
    with rasterio.open(out_path, 'w', **meta) as dest:
        dest.write(data)

def save_mask_tile(data, out_path):
    im = Image.fromarray(data.astype(np.uint8))
    im.save(out_path)

def check_out_of_bounds(data, out_of_bounds_value):
    return (data == out_of_bounds_value).any()
    # return (data.mean() == out_of_bounds_value)

# Additional functions for writing tiles to TFRecords
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_example(image_data, mask_data, x, y):
    feature = {
        'image': _bytes_feature(image_data),
        'mask': _bytes_feature(mask_data),
        'x': tf.train.Feature(int64_list=tf.train.Int64List(value=[x])),
        'y': tf.train.Feature(int64_list=tf.train.Int64List(value=[y]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_tfrecord(image_data, mask_data, out_path, x, y):
    with tf.io.TFRecordWriter(out_path) as writer:
        example = create_example(image_data, mask_data, x, y)
        writer.write(example.SerializeToString())


def create_tiles_partition(
        geotiff_path, 
        mask_path, 
        out_dir, 
        tile_size=256, 
        out_of_bounds_value=1.01, 
        offset = False
        ):
    with rasterio.open(geotiff_path) as src:
        with rasterio.open(mask_path) as src_mask:
            meta = src.meta.copy()
            meta.update({
                'driver': 'GTiff',
                'height': tile_size,
                'width': tile_size,
                'count': 1
            })
            if not offset:
                start_val = 0
            else:
                start_val = tile_size // 2
            for i in range(start_val, src.shape[0], tile_size):
                for j in range(start_val, src.shape[1], tile_size):
                    if i + tile_size <= src.shape[0] and j + tile_size <= src.shape[1]:
                        window = Window(j, i, tile_size, tile_size)
                        data = read_data_in_window(src, window)

                        data_mask = read_data_in_window(src_mask, window)
                        data = data.astype(np.float32)
    
                        # print(f"Max after conversion: {data_mask.max()}, Min after conversion: {data_mask.min()}")

                        out_of_bounds = check_out_of_bounds(data, out_of_bounds_value)
                        if not out_of_bounds:
                            out_path = os.path.join(out_dir, f"tile_{i}_{j}.tfrecord")
                            write_tfrecord(data.tobytes(), data_mask.tobytes(), out_path, j, i)  # i and j are top-left coordinates of the tile


def create_tiles_sampled(
        geotiff_path, 
        mask_path, 
        out_dir, 
        tile_size=256, 
        out_of_bounds_value=1.01,
        n_tiles=3150 # approximately 7 of 10 will be in bounds and saved as currently configured
        ):
    tile_log = pd.DataFrame(columns=['image', 'y', 'x', 'out_of_bounds'])
    with rasterio.open(geotiff_path) as src:
        with rasterio.open(mask_path) as src_mask:
            meta = src.meta.copy()
            meta.update({
                'driver': 'GTiff',
                'height': tile_size,
                'width': tile_size,
                'count': 1
            })

            for _ in range(n_tiles):
                upper_left_x = random.sample(range(0, src.shape[1] - tile_size), 1)[0]
                upper_left_y = random.sample(range(0, src.shape[0] - tile_size), 1)[0]
                window = Window(upper_left_x, upper_left_y, tile_size, tile_size)
                data = read_data_in_window(src, window)

                data_mask = read_data_in_window(src_mask, window)
                data = data.astype(np.float32)

                out_of_bounds = check_out_of_bounds(data, out_of_bounds_value)
                if not out_of_bounds:
                    out_path = os.path.join(out_dir, f"tile_{upper_left_y}_{upper_left_x}.tfrecord")
                    write_tfrecord(data.tobytes(), data_mask.tobytes(), out_path, upper_left_x, upper_left_y)
                    tile_log.loc[len(tile_log)] = [geotiff_path, upper_left_y, upper_left_x, out_of_bounds]
    return tile_log
    

# create_tiles_partition(path_image, path_mask, dir_partition)
# create_tiles_partition(path_image, path_mask, dir_overlap, offset=True)

log = create_tiles_sampled(path_image, path_mask, dir_sampled)
log.to_csv(os.path.join(dir_tiles, '_sampled_log.csv'), index=False)







# ------------------------------------
# visualize a random tile

# def _parse_record(example_proto):
#     feature_description = {
#         'image': tf.io.FixedLenFeature([], tf.string),
#         'mask': tf.io.FixedLenFeature([], tf.string),
#         'x': tf.io.FixedLenFeature([], tf.int64),
#         'y': tf.io.FixedLenFeature([], tf.int64)
#     }
#     example = tf.io.parse_single_example(example_proto, feature_description)
    
#     image = tf.io.decode_raw(example['image'], tf.float32)
#     mask = tf.io.decode_raw(example['mask'], tf.uint8)
 
#     image = tf.reshape(image, [256, 256, 1])
#     mask = tf.reshape(mask, [256, 256, 1])
    
#     image = tf.image.resize(image, [256, 256])
#     mask = tf.cast(mask, tf.float32)

#     x = example['x']
#     y = example['y']


#     return image, mask, x,y

# def parse_image_pair(tfrecord_file):
#     dataset = tf.data.TFRecordDataset([tfrecord_file])
#     image_mask_pair = list(dataset.map(_parse_record))[0]
#     return image_mask_pair#.numpy()

# def display_image_mask(image, mask, x, y):
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#     ax[0].imshow(image, cmap='viridis')
#     ax[0].set_title("Image (x: "+ str(x.numpy()) + ", y: " + str(y.numpy()) + ")")
#     ax[1].imshow(mask*255.0, cmap='viridis', vmin=0, vmax=255)  # or cmap='jet'
#     ax[1].set_title("Mask")
#     plt.show()

# tfrecord_files = list(filter(lambda x: x.endswith('.tfrecord'), os.listdir(dir_sampled)))

# random_tfrecord = os.path.join(dir_sampled, random.choice(tfrecord_files))
# img, mask, x, y = parse_image_pair(random_tfrecord)
# display_image_mask(img, mask, x, y)


