import os
import rasterio
import numpy as np
from rasterio.transform import from_origin
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess_tiles import process_data
import define_model_functions
from define_model_parameters import *


dir_out = dir_model + 'data/output/'
dir_out_tiles = dir_out + 'tiles/'
dir_out_full = dir_out + 'full/'
dt = '2023-06-14_155740'
dir_image = '../../data/gee/imagery/sentinel-1/'
path_image = dir_image + "S1A_IW_GRDH_1SSV_20161001T133344_20161001T133409_013297_015303_46D8_cropped.tif"

_, _, _, reconstruction_dataset, steps_per_epoch, validation_steps, test_steps = process_data(
    augment=AUGMENT, 
    training_split=TRAINING_SPLIT,
    batch_size=BATCH_SIZE,
    shuffle_buffer=SHUFFLE_BUFFER,
    tile_dir=dir_in
    )

focal_loss_fixed = define_model_functions.focal_loss(gamma=2., alpha=.75)

model = load_model(
    dir_weights + dt + '_lidar-unet.hdf5', 
    custom_objects={'focal_loss': define_model_functions.focal_loss, 
                    'focal_loss_fixed': focal_loss_fixed, 
                    'iou': define_model_functions.iou, 
                    'dice_loss': define_model_functions.dice_loss}
)

dataset = reconstruction_dataset

predictions_dir = dir_out_tiles + dt + '/'
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir, exist_ok=True)

with rasterio.open(path_image) as original_src:
    original_crs = original_src.crs
    original_transform = original_src.transform

for (image, _), (x, y) in dataset:
    image = tf.expand_dims(image, axis=0) 
    preds = model.predict(image)  
    pred = preds[0] 
    x_val = x.numpy()  
    y_val = y.numpy()
    metadata = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': pred.shape[1],
        'height': pred.shape[0],
        'count': 1,
        'crs': original_crs,
        'transform': from_origin(x_val, y_val, 1, 1)
    }

    with rasterio.open(f'{predictions_dir}/prediction_{x_val}_{y_val}.tif', 'w', **metadata) as dst:
        dst.write(pred.squeeze(), 1)

full_image_shape = (original_src.shape[0], original_src.shape[1])
prediction_image = np.zeros(full_image_shape)


for filename in os.listdir(predictions_dir):
    with rasterio.open(f'{predictions_dir}/{filename}') as src:
        pred = src.read(1)  
        _, i, j = filename[:-4].split('_')  
        x, y = int(i), int(j)  

        prediction_image[y:y+pred.shape[0], x:x+pred.shape[1]] = pred

metadata = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'width': prediction_image.shape[1],
    'height': prediction_image.shape[0],
    'count': 1,
    'crs': original_crs,
    'transform': original_transform
}

# Save the prediction image as a GeoTIFF
with rasterio.open(dir_out_full + dt + '_prediction_continuous.tif', 'w', **metadata) as dst:
    dst.write(prediction_image, 1)

with rasterio.open(dir_out_full + dt + '_prediction_binary.tif', 'w', **metadata) as dst:
    dst.write(prediction_image > 0.5, 1)




# import rasterio
# import numpy as np
# from rasterio.transform import from_origin
# from tensorflow.keras.models import load_model
# from preprocess_tiles import BATCH_SIZE, train_dataset, test_dataset, steps_per_epoch, validation_steps
# from define_model_functions import *

# focal_loss_fixed = focal_loss(gamma=2., alpha=.75)

# model = load_model('../../models/lidar-unet.hdf5', custom_objects={'focal_loss': focal_loss, 'focal_loss_fixed': focal_loss_fixed, 'iou': iou, 'dice_loss': dice_loss})

# dataset = train_dataset.concatenate(test_dataset)

# # Make predictions and store them with their coordinates
# predictions = []
# coordinates = []
# for image, _, x, y in dataset:
#     # image = np.expand_dims(image, axis=0)  # Expand dimension to fit the model input shape
#     preds = model.predict(image)  # Predict the masks
#     for pred, x_val, y_val in zip(preds, x, y):
#         predictions.append(pred.squeeze())
#         coordinates.append((x_val.numpy(), y_val.numpy()))

# # read original image
# with rasterio.open('../../models/data/lidar-mask/whole/S1A_IW_GRDH_1SSV_20161001T133344_20161001T133409_013297_015303_46D8_cropped.tif') as src:
#     # Create an empty 2D array with the same shape as the original image
#     full_image_shape = (src.shape[0], src.shape[1])
#     prediction_image = np.zeros(full_image_shape)

# tile_size = 256

# # Place each prediction tile in the correct position
# for pred, (x, y) in zip(predictions, coordinates):
#     prediction_image[y:y+tile_size, x:x+tile_size] = pred

# # Create GeoTIFF metadata
# metadata = {
# # need to complete
# }

# # Save the prediction image as a GeoTIFF
# with rasterio.open('../../models/data/lidar-mask/predictions/prediction.tif', 'w', **metadata) as dst:
#     dst.write(prediction_image, 1)

# with rasterio.open('../../models/data/lidar-mask/predictions/prediction.tif') as src:
#     fig, ax = plt.subplots()
#     ax.imshow(src.read(1), cmap='pink')
#     plt.show()


#     'driver': 'GTiff',
#     'height': full_image_shape[0],
#     'width': full_image_shape[1],
#     'count': 1,
#     'dtype': 'float32',
#     'crs': '+proj=latlong'
#     # 'transform': from_origin(xmin, ymin, x_pixel_size, y_pixel_size),