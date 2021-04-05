from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflowUtils as tfu
import sys
import scipy.io as sio 


# FROM : https://github.com/krasserm/super-resolution/blob/master/model/srgan.py
def res_block(x_in,num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization(momentum=0.8)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Add()([x_in, x])
    return x

# FROM : https://github.com/krasserm/super-resolution/blob/master/model/srgan.py
def upsample(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    return PReLU(shared_axes=[1, 2])(x)

def make_SRNN_SC_model(Case):

    num_filters = Case['numFilters']
    numBlocks = Case['numBlocks']

    h_LR = Case['h_LR']
    w_LR = Case['w_LR']
    c = Case['c_LR']

    h_HR = Case['h_HR']
    w_HR = Case['w_HR']

    x_in = Input(shape=(h_LR,w_LR,c))
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)

    for _ in range(numBlocks):
        x = res_block(x, num_filters)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    x = upsample(x,int((h_HR//h_LR)**2*c))

    x = Reshape((h_HR, w_HR, c))(x)

    return Model(x_in, x)

def makeModel(Case):
    SRNN = make_SRNN_SC_model(Case)
    SRNN.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse'])
    SRNN.summary()
    return SRNN


def train(SRNN,Case):
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    EPOCHS = 100
    BATCH_SIZE=min(256 ,Case['nSnapTrain'])   
    Case['dsTrain'] = Case['dsTrain'].shuffle(100).repeat(EPOCHS).batch(BATCH_SIZE)
    Case['dsTest'] = Case['dsTest'].batch(BATCH_SIZE)
    WeightFolder = 'weights/WeightsSC_filt_'+str(Case['numFilters'])+'_blocks_'+str(Case['numBlocks'])
    path = WeightFolder.split('/')
    for i in range(len(path)):
         directory = os.path.join(*path[:i+1])
         os.makedirs(directory,exist_ok=True)

    mc = tf.keras.callbacks.ModelCheckpoint(WeightFolder+'/weights{epoch:08d}.h5', period=100)
    csv_logger = tf.keras.callbacks.CSVLogger(WeightFolder +'/training.log')
    

    history = SRNN.fit(
        Case['dsTrain'],
        steps_per_epoch=Case['nSnapTrain'] // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=Case['dsTest'],
        validation_steps=Case['nSnapTest'] // BATCH_SIZE,
        callbacks = [mc,csv_logger]
    )
    
    sio.savemat( WeightFolder + '/training_results.mat', {'train':history.history['loss'], 'valid':history.history['val_loss']})


