# ------------------------------

import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import backend as K
# from preprocess_tiles import BATCH_SIZE, AUGMENT, TRAINING_SPLIT, train_dataset, test_dataset, steps_per_epoch, validation_steps
from preprocess_tiles import process_data
import define_model_functions
from define_model_parameters import *
# from evaluate_model import *

# -----------------------------
# process data

train_dataset, validation_dataset, test_dataset, _, steps_per_epoch, validation_steps, test_steps = process_data(
    augment=AUGMENT, 
    training_split=TRAINING_SPLIT,
    validation_split=VALIDATION_SPLIT,
    batch_size=BATCH_SIZE,
    shuffle_buffer=SHUFFLE_BUFFER,
    tile_dir=dir_in
    )

# -----------------------------
# model

def unet(input_shape=INPUT_SHAPE, dropout_rate=DROPOUT_RATE, l1_coeff=L1_COEF, l2_coeff=L2_COEF):
    reg = l1_l2(l1=l1_coeff, l2=l2_coeff)
    
    inputs = Input(input_shape)

    #contracting path
    conv1 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv1 = Dropout(dropout_rate)(conv1)

    conv1 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg)(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    conv2 = Dropout(dropout_rate)(conv2)
    
    conv2 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    conv3 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg)(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    conv3 = Dropout(dropout_rate)(conv3)
    
    conv3 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    conv4 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg)(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    
    conv4 = Dropout(dropout_rate)(conv4)
    
    conv4 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    
    pool4 = MaxPooling2D((2, 2))(conv4)
    
    conv5 = Conv2D(512, (3, 3), padding='same', kernel_regularizer=reg)(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    
    conv5 = Dropout(dropout_rate)(conv5)
    
    conv5 = Conv2D(512, (3, 3), padding='same', kernel_regularizer=reg)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    
    #  expnding path
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    
    conv6 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg)(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    conv6 = Dropout(dropout_rate)(conv6)
    
    conv6 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    
    conv8 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg)(up7)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    
    conv8 = Dropout(dropout_rate)(conv8)
    
    conv8 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    
    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv2], axis=3)
    
    conv10 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg)(up9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    
    conv10 = Dropout(dropout_rate)(conv10)
    
    conv10 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg)(conv10)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    
    up11 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv10), conv1], axis=3)
    
    conv12 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg)(up11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)
    
    conv12 = Dropout(dropout_rate)(conv12)
    
    conv12 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg)(conv12)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)
    
    # output
    output = Conv2D(1, (1, 1), activation='sigmoid')(conv12)
    return Model(inputs=inputs, outputs=output)

model = unet()

model.compile(
    optimizer=Adam(learning_rate=LR), 
    loss=LOSS_FN,
    metrics=['accuracy', define_model_functions.iou]
    )

model_checkpoint = ModelCheckpoint(
    dir_weights+DT+'_lidar-unet.hdf5',
    monitor='val_iou',
    mode = 'max', # for IOU only
    verbose=1,
    save_best_only=False
    )

start_time = time.time()
history = model.fit(
    train_dataset,
    epochs=EPOCHS, 
    steps_per_epoch=steps_per_epoch, 
    validation_data=validation_dataset,
    validation_steps=validation_steps, 
    callbacks=[model_checkpoint, lr_scheduler, early_stopping]  
)
end_time = time.time()
training_time = end_time - start_time

define_model_functions.plot_history(history, dir_results+DT)

model.load_weights(dir_weights+DT+'_lidar-unet.hdf5')
score = model.evaluate(test_dataset, steps=test_steps, verbose=1)
# save score
with open(dir_results+DT+'_score.txt', 'w') as f:
    f.write(str(score))

model_info = define_model_functions.evaluate_model(
    desc_str='unet-with-overlap-training-data-and-focal-loss',
    test_dataset=test_dataset, 
    model=model,
    training_time=training_time,
    validation_steps=test_steps,
    steps_per_epoch=steps_per_epoch,
    params=PARAMS
)
df = pd.DataFrame(model_info)
df.to_csv(dir_results+DT+'_results.csv', index=False)


# visualize sample predictions
define_model_functions.visualize_prediction(train_dataset, model)  
define_model_functions.visualize_prediction(test_dataset, model) 