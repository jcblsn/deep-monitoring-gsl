
import os
import numpy as np
import rasterio
import tensorflow as tf
from sklearn.model_selection import train_test_split

def process_data(augment=False, training_split=0.7, validation_split=0.2, batch_size=32, shuffle_buffer=1000, tile_dir='../../models/lidar-mask/data/input/tiles'):
    
    # check training_split + validation_split + test_split = 1.0
    assert training_split + validation_split < 1.0, "training_split + validation_split must be less than 1.0"

    dir_subdirs = [os.path.join(tile_dir, d) for d in os.listdir(tile_dir) if os.path.isdir(os.path.join(tile_dir, d))]

    dir_partition = os.path.join(tile_dir, '20161001', 'partition')

    dir_subsubdirs = []
    for subdir in dir_subdirs:
        dir_subsubdirs.extend([os.path.join(subdir, d) for d in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, d))])

    tfrecord_files = []
    for subdir in dir_subsubdirs:
        # if directory other than dir_partition
        # if subdir != dir_partition:
        if subdir.endswith('sampled'):
            tfrecord_files.extend([os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith('.tfrecord')])

    tfrecord_files_partition = [os.path.join(dir_partition, f) for f in os.listdir(dir_partition) if f.endswith('.tfrecord')]

    assert len(tfrecord_files) > 0, "No TFRecord files found"

    def _parse_record_with_coords(example_proto):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'mask': tf.io.FixedLenFeature([], tf.string),
            'x': tf.io.FixedLenFeature([], tf.int64),
            'y': tf.io.FixedLenFeature([], tf.int64)
        }

        example = tf.io.parse_single_example(example_proto, feature_description)
        
        image = tf.io.decode_raw(example['image'], tf.float32)
        mask = tf.io.decode_raw(example['mask'], tf.uint8)
        image = tf.reshape(image, [256, 256, 1])
        mask = tf.reshape(mask, [256, 256, 1])
        
        x = example['x']
        y = example['y']

        return (image, mask), (x, y)
    
    def _parse_record(example_proto):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'mask': tf.io.FixedLenFeature([], tf.string),
        }

        example = tf.io.parse_single_example(example_proto, feature_description)
        
        image = tf.io.decode_raw(example['image'], tf.float32)
        mask = tf.io.decode_raw(example['mask'], tf.uint8)
        image = tf.reshape(image, [256, 256, 1])
        mask = tf.reshape(mask, [256, 256, 1])

        return image, mask

    def random_augmentation(image, mask):
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
            
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
            
        if tf.random.uniform(()) > 0.5:
            image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32))
            mask = tf.image.rot90(mask, k=tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32))
        
        return image, mask

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(_parse_record, num_parallel_calls=tf.data.AUTOTUNE)
    reconstruction_dataset = tf.data.TFRecordDataset(tfrecord_files_partition).map(_parse_record_with_coords)

    # Count total number of samples
    n_total = sum(1 for _ in dataset)
    n_train = int(n_total * training_split)
    n_validation = int(n_total * validation_split)
    n_test = int(n_total - n_train - n_validation)

    if augment:
        dataset = dataset.map(random_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(shuffle_buffer, seed=9)

    # split
    train_dataset = dataset.take(n_train)
    validation_dataset = dataset.skip(n_train).take(n_validation)
    test_dataset = dataset.skip(n_train + n_validation)

    # batch and repeat
    train_dataset = train_dataset.batch(batch_size).repeat()
    validation_dataset = validation_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    steps_per_epoch = n_train // batch_size
    validation_steps = n_validation // batch_size
    test_steps = n_test // batch_size

    return train_dataset, validation_dataset, test_dataset, reconstruction_dataset, steps_per_epoch, validation_steps, test_steps
