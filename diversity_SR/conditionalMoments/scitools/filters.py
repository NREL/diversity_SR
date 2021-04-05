import tensorflow as tf
import numpy as np
from scipy import ndimage
from parallel import printRoot


def verifyNxVal(Nx):
    if Nx<4:
        printRoot('WARNING: Nx found to be:',Nx)
    return       

def boxFilter2D(field,boxSizeH, boxSizeW):
    fieldShape = field.shape
    h_HR = fieldShape[0]
    w_HR = fieldShape[1]
    c = fieldShape[2]
    verifyNxVal(h_HR)
    nDim = len(fieldShape)
    KH = boxSizeH
    KW = boxSizeW
  
    if nDim==3:
        fieldFilt = np.zeros((h_HR,w_HR,c))
        # ~~~~ Filter and coarsen 
        weight = tf.constant(1.0/(KH*KW), shape=[KH, KW, 1, 1], dtype=tf.float32)
        fieldFiltCoarseU = tf.nn.conv2d(np.reshape(field[:,:,0],(1,fieldShape[0],fieldShape[1],1)), weight, strides=[1, KH, KW, 1], padding='SAME') 
        fieldFiltCoarseV = tf.nn.conv2d(np.reshape(field[:,:,1],(1,fieldShape[0],fieldShape[1],1)), weight, strides=[1, KH, KW, 1], padding='SAME') 
        # ~~~~ Upsample
        weightUp = tf.constant(1.0, shape=[KH, KW, 1, 1], dtype=tf.float32)
        fieldFilt[:,:,0] = np.squeeze(tf.nn.conv2d_transpose(fieldFiltCoarseU, weightUp, strides=[1, KH, KW, 1],  output_shape=[1, h_HR, w_HR, 1], padding='SAME'))
        fieldFilt[:,:,1] = np.squeeze(tf.nn.conv2d_transpose(fieldFiltCoarseV, weightUp, strides=[1, KH, KW, 1],  output_shape=[1, h_HR, w_HR, 1], padding='SAME'))
  
        # Get the subFilter 
        fieldSubFilt = field - fieldFilt

        return fieldFilt, fieldSubFilt, np.squeeze(fieldFiltCoarseU[0,:,:,0])

    else:
        printRoot('boxFilter2D exclusively dedicated to 2D windData')
        return

def upSample2D(field,fieldFilt):
    fieldShape = field.shape
    h_HR = fieldShape[0]
    w_HR = fieldShape[1]
    c = fieldShape[2]
    fieldFiltShape = fieldFilt.shape
    h_LR = fieldFiltShape[0]
    w_LR = fieldFiltShape[1]
    c = fieldFiltShape[2]
    fieldFilt = np.reshape(fieldFilt,(1,h_LR,w_LR,c))
    fieldFiltUpsampled = np.zeros((h_HR,w_HR,c))
    verifyNxVal(h_HR)
    nDim = len(fieldShape)
    boxSizeH = int((h_HR//h_LR))
    boxSizeW = int((w_HR//w_LR))
    KH = boxSizeH
    KW = boxSizeW
    if nDim==3:
        # ~~~~ Upsample
        weightUp = tf.constant(1.0, shape=[KH, KW, 1, 1], dtype=tf.float32)
        fieldFiltUpsampled[:,:,0] = np.squeeze(tf.nn.conv2d_transpose(np.reshape(fieldFilt[:,:,:,0],(1,h_LR,w_LR,1)), weightUp, strides=[KH, KW],  output_shape=[1, h_HR, w_HR, 1], padding='SAME'))
        fieldFiltUpsampled[:,:,1] = np.squeeze(tf.nn.conv2d_transpose(np.reshape(fieldFilt[:,:,:,1],(1,h_LR,w_LR,1)), weightUp, strides=[KH, KW],  output_shape=[1, h_HR, w_HR, 1], padding='SAME'))
          

        # Get the subFilter 
        fieldSubFilt = field - fieldFiltUpsampled

        return fieldFiltUpsampled, fieldSubFilt, np.squeeze(fieldFilt[0,:,:,0])

    else:
        printRoot('boxFilter2D exclusively dedicated to 2D windData')
        return


