import itertools
import time
import h5py
import sys
import os
import scipy.special
import numpy as np
sys.path.append('partools')
sys.path.append('scitools')
sys.path.append('util')
import parallel as par
from filters import boxFilter2D, upSample2D
import tensorflow as tf
import tensorflowUtils as tfu
from tensorflow.keras.models import load_model

par.printRoot('GENERATE TF RECORD WITH SUBFILTER SQUARED')

# Filenames to read
filenameTrain = 'DataWind/SFexampleData.tfrecord'
filenameTest = 'DataWind/SFexampleData.tfrecord'
model = load_model('pretrainedWeights/SF_mom1/WeightsSC_filt_32_blocks_16/weights00005000.h5')


# Initialize the tf dataset to read
dsTrain = tf.data.TFRecordDataset(filenameTrain)
dsTrain = dsTrain.map(tfu._parse_image_wind_SFfunction)  # parse the record
dsTest = tf.data.TFRecordDataset(filenameTest)
dsTest = dsTest.map(tfu._parse_image_wind_SFfunction)  # parse the record

# Filename to write
dataPath = filenameTrain.split('/')
dataPath[-1] = 'SQ' + dataPath[-1]
filenameToWriteTrain = os.path.join(*dataPath)

dataPath = filenameTest.split('/')
dataPath[-1] = 'SQ' + dataPath[-1]
filenameToWriteTest = os.path.join(*dataPath)



with tf.io.TFRecordWriter(filenameToWriteTrain) as writer:
    counter=0
    for image_LR, image_SF in dsTrain:
        # ~~~~ Prepare the data
        LR_snapshot = np.squeeze(image_LR.numpy())
        SF_snapshot = np.squeeze(image_SF.numpy())
        w_LR, h_LR, c  = LR_snapshot.shape     
        w_HR, h_HR, c  = SF_snapshot.shape     
        # Create the subfilter field
        A = np.squeeze(model.predict(np.reshape(LR_snapshot,(1,w_LR,h_LR,c))))
        subfiltFieldSq = (SF_snapshot - A)**2      

        # ~~~~ Write the data
        tf_example = tfu.SF_image_example(counter,h_LR,w_LR,h_HR,w_HR,c,bytes(LR_snapshot),bytes(subfiltFieldSq))
        writer.write(tf_example.SerializeToString())

        counter += 1

with tf.io.TFRecordWriter(filenameToWriteTest) as writer:
    counter=0
    for image_LR, image_SF in dsTest:
 
        # ~~~~ Prepare the data
        LR_snapshot = np.squeeze(image_LR.numpy())
        SF_snapshot = np.squeeze(image_SF.numpy())
        w_LR, h_LR, c  = LR_snapshot.shape     
        w_HR, h_HR, c  = SF_snapshot.shape     
    
        # Create the subfilter field
        A = np.squeeze(model.predict(np.reshape(LR_snapshot,(1,w_LR,h_LR,c))))
        subfiltFieldSq = (SF_snapshot - A)**2      
        
        # ~~~~ Write the data
        tf_example = tfu.SF_image_example(counter,h_LR,w_LR,h_HR,w_HR,c,bytes(LR_snapshot),bytes(subfiltFieldSq))
        writer.write(tf_example.SerializeToString())

        counter += 1
