from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("util")
sys.path.append("scitools")
sys.path.append("partools")
import parallel as par
from filters import boxFilter2D, upSample2D
from plotsUtil import *
import tensorflowUtils as tfu
from tensorflow.keras.models import load_model
from myProgressBar import printProgressBar

def writeTfRecords(Case):
    h_HR = Case['h_HR']    
    w_HR = Case['w_HR']    
    c_HR = Case['c_HR']    
    h_LR = Case['h_LR']    
    w_LR = Case['w_LR']    
    c_LR = Case['c_LR']    
    nSnapTrain = Case['nSnapTrain']    
    nSnapTest = Case['nSnapTest']    

    dataPath = Case['dataFilenameTrain'].split('/')
    indexPrefix = dataPath[-1].index('.tfrecord')
    dataPath[-1] = 'modelNN' + '_' + dataPath[-1][:indexPrefix] + "_diversity.tfrecord"
    filenameToWrite = os.path.join(*dataPath)
   

    # ~~~~ Write TF RECORD for training data
    if par.irank == par.iroot:
        printProgressBar(0, nSnapTrain, prefix = 'Output snapshot Train ' + str(0) + ' / ' +str(nSnapTrain),suffix = 'Complete', length = 50) 
        with tf.io.TFRecordWriter(filenameToWrite) as writer:
            counter=0
            for image_LR, image_SF in Case['dsTrain']:
    
                # ~~~~ Log advancement
                printProgressBar(counter+1, nSnapTrain, prefix = 'Output snapshot Train ' + str(counter+1) + ' / ' +str(nSnapTrain),suffix = 'Complete', length = 50) 
                #par.printRoot(str(counter) + '/' + str(nSnapTrain))

                # ~~~~ Prepare the data
                LR_snapshot = np.squeeze(image_LR.numpy())
                SF_snapshot = np.squeeze(image_SF.numpy())
                w_LR, h_LR, c  = LR_snapshot.shape
                w_HR, h_HR, c  = SF_snapshot.shape
                # upsample the LR
                LR_upsampled, _, _ = upSample2D(SF_snapshot,LR_snapshot)
                # Recreate HR
                HR_snapshot = np.squeeze(image_SF.numpy()) + LR_upsampled
                # Create moments
                Mean = np.float64(np.squeeze(Case['modelFirstMoment'].predict(np.reshape(LR_snapshot,(1,w_LR,h_LR,c)))))
                Std = np.squeeze(np.sqrt(np.clip(Case['modelSecondMoment'].predict(np.reshape(LR_snapshot,(1,w_LR,h_LR,c))),0,100000)))
                

                # ~~~~ Write the data
                index = counter
                data_LR = np.reshape(LR_snapshot,(1,w_LR,h_LR,c))
                data_HR = np.reshape(HR_snapshot,(1,w_HR,h_HR,c))
                mean = np.reshape(Mean,(1,w_HR,h_HR,c))
                std = np.reshape(Std,(1,w_HR,h_HR,c))
                tf_example = tfu.diversity_image_example(counter,bytes(data_LR),h_LR,w_LR,bytes(data_HR),h_HR,w_HR,c,bytes(mean),bytes(std))
                writer.write(tf_example.SerializeToString())
    
                counter=counter+1

    dataPath = Case['dataFilenameTest'].split('/')
    indexPrefix = dataPath[-1].index('.tfrecord')
    dataPath[-1] = 'modelNN' + '_' + dataPath[-1][:indexPrefix] + "_diversity.tfrecord"
    filenameToWrite = os.path.join(*dataPath)
   

    # ~~~~ Write TF RECORD for training data
    if par.irank == par.iroot:
        printProgressBar(0, nSnapTest, prefix = 'Output snapshot Test ' + str(0) + ' / ' +str(nSnapTest),suffix = 'Complete', length = 50) 
        with tf.io.TFRecordWriter(filenameToWrite) as writer:
            counter=0
            for image_LR, image_SF in Case['dsTest']:
    
                # ~~~~ Log advancement
                printProgressBar(counter+1, nSnapTest, prefix = 'Output snapshot Test ' + str(counter+1) + ' / ' +str(nSnapTest),suffix = 'Complete', length = 50) 
                #par.printRoot(str(counter) + '/' + str(nSnapTest))

                # ~~~~ Prepare the data
                LR_snapshot = np.squeeze(image_LR.numpy())
                SF_snapshot = np.squeeze(image_SF.numpy())
                w_LR, h_LR, c  = LR_snapshot.shape
                w_HR, h_HR, c  = SF_snapshot.shape
                # upsample the LR
                LR_upsampled, _, _ = upSample2D(SF_snapshot,LR_snapshot)
                # Recreate HR
                HR_snapshot = np.squeeze(image_SF.numpy()) + LR_upsampled
                # Create moments
                Mean = np.float64(np.squeeze(Case['modelFirstMoment'].predict(np.reshape(LR_snapshot,(1,w_LR,h_LR,c)))))
                Std = np.squeeze(np.sqrt(np.clip(Case['modelSecondMoment'].predict(np.reshape(LR_snapshot,(1,w_LR,h_LR,c))),0,100000)))
                

                # ~~~~ Write the data
                index = counter
                data_LR = np.reshape(LR_snapshot,(1,w_LR,h_LR,c))
                data_HR = np.reshape(HR_snapshot,(1,w_HR,h_HR,c))
                mean = np.reshape(Mean,(1,w_HR,h_HR,c))
                std = np.reshape(Std,(1,w_HR,h_HR,c))
                tf_example = tfu.diversity_image_example(counter,bytes(data_LR),h_LR,w_LR,bytes(data_HR),h_HR,w_HR,c,bytes(mean),bytes(std))
                writer.write(tf_example.SerializeToString())
    
                counter=counter+1
