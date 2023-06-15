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
from preprocess_tiles import process_data
import define_model_functions

dir_model = '../../models/lidar-mask/'
dir_weights = dir_model + 'weights/'
dir_results = dir_model + 'results/'
dir_in = '../../models/lidar-mask/data/input/tiles/'

DT = datetime.now().strftime("%Y-%m-%d_%H%M%S")
AUGMENT = True
TRAINING_SPLIT = 0.7
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
BATCH_SIZE = 32
LR = 2e-5
FOCAL_LOSS_GAMMA = 2.
FOCAL_LOSS_ALPHA = .66
# FOCAL_LOSS_ALPHA = .5
# LOSS_FN = define_model_functions.focal_loss(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA)
LOSS_FN = define_model_functions.dice_loss
SHUFFLED = True
if SHUFFLED:
    SHUFFLE_BUFFER = 10000
EPOCHS = 50
INPUT_SHAPE = (256, 256, 1)
DROPOUT_RATE = 0.5
L1_COEF = 1e-3
L2_COEF = 1e-3

LR_SCH_PATIENCE = 2
LR_SCH_FACTOR = 0.5
LR_SCH_MIN = 1e-7

EARLY_STOP_PATIENCE = 5

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=LR_SCH_FACTOR, 
    patience=LR_SCH_PATIENCE, 
    min_lr=LR_SCH_MIN, 
    verbose=1
    )

LR_SCH_DECAY_RATE = 0.005

def lr_time_based_decay(epoch, lr = LR):
    return lr * 1 / (1 + LR_SCH_DECAY_RATE * epoch)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_time_based_decay, verbose=1)

early_stopping = EarlyStopping(
    monitor='val_loss', 
    min_delta=0, 
    patience=EARLY_STOP_PATIENCE, 
    verbose=1, 
    mode='auto', 
    restore_best_weights=True
    )

PARAMS = {
    'DATA_PATH': dir_in,
    'DT': DT,
    'INPUT_SHAPE': INPUT_SHAPE,
    'EPOCHS': EPOCHS,
    'BATCH_SIZE': BATCH_SIZE,
    'TRAINING_SPLIT': TRAINING_SPLIT,
    'VALIDATION_SPLIT': VALIDATION_SPLIT,
    'TEST_SPLIT': TEST_SPLIT,
    'AUGMENT': AUGMENT,
    'LR': LR,
    'LOSS_FN': LOSS_FN,
    'SHUFFLED': SHUFFLED,
    'SHUFFLE_BUFFER': SHUFFLE_BUFFER,
    'L1_COEF': L1_COEF,
    'L2_COEF': L2_COEF,
    'DROPOUT_RATE': DROPOUT_RATE,
    # 'FOCAL_LOSS_GAMMA': FOCAL_LOSS_GAMMA,
    # 'FOCAL_LOSS_ALPHA': FOCAL_LOSS_ALPHA,
    # 'LR_SCH_PATIENCE': LR_SCH_PATIENCE,
    # 'LR_SCH_FACTOR': LR_SCH_FACTOR,
    # 'LR_SCH_MIN': LR_SCH_MIN,
    'LR_SCH_DECAY_RATE': LR_SCH_DECAY_RATE,
    'EARLY_STOP_PATIENCE': EARLY_STOP_PATIENCE
}