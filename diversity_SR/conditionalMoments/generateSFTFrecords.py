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

par.printRoot('GENERATE TF RECORD WITH SUBFILTER')

# Filenames to read
filenameTrain = 'DataWind/exampleData.tfrecord'
filenameTest = 'DataWind/exampleData.tfrecord'

# Initialize the tf dataset to read
dsTrain = tf.data.TFRecordDataset(filenameTrain)
dsTrain = dsTrain.map(tfu._parse_image_wind_function)  # parse the record
dsTest = tf.data.TFRecordDataset(filenameTest)
dsTest = dsTest.map(tfu._parse_image_wind_function)  # parse the record

# Filename to write
dataPath = filenameTrain.split('/')
dataPath[-1] = 'SF' + dataPath[-1]
filenameToWriteTrain = os.path.join(*dataPath)

dataPath = filenameTest.split('/')
dataPath[-1] = 'SF' + dataPath[-1]
filenameToWriteTest = os.path.join(*dataPath)


with tf.io.TFRecordWriter(filenameToWriteTrain) as writer:
    counter=0
    for image_HR, image_LR in dsTrain:
 
        # ~~~~ Prepare the data
        LR_snapshot = np.squeeze(image_LR.numpy())
        HR_snapshot = np.squeeze(image_HR.numpy())
        w_LR, h_LR, c  = LR_snapshot.shape     
        w_HR, h_HR, c  = HR_snapshot.shape     
        # Create the subfilter field
        filtField, subfiltField, _ = upSample2D(HR_snapshot,LR_snapshot)
        
        # ~~~~ Write the data
        tf_example = tfu.SF_image_example(counter,h_LR,w_LR,h_HR,w_HR,c,bytes(LR_snapshot),bytes(subfiltField))
        writer.write(tf_example.SerializeToString())

        counter += 1

with tf.io.TFRecordWriter(filenameToWriteTest) as writer:
    counter=0
    for image_HR, image_LR in dsTest:
 
        # ~~~~ Prepare the data
        LR_snapshot = np.squeeze(image_LR.numpy())
        HR_snapshot = np.squeeze(image_HR.numpy())
        w_LR, h_LR, c  = LR_snapshot.shape     
        w_HR, h_HR, c  = HR_snapshot.shape     
    
        # Create the subfilter field
        filtField, subfiltField, _ = upSample2D(HR_snapshot,LR_snapshot)
        
        # ~~~~ Write the data
        tf_example = tfu.SF_image_example(counter,h_LR,w_LR,h_HR,w_HR,c,bytes(LR_snapshot),bytes(subfiltField))
        writer.write(tf_example.SerializeToString())

        counter += 1
