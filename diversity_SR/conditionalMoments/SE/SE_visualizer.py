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
import tensorflow as tf
import tensorflowUtils as tfu
from plotsUtil import *
from myProgressBar import printProgressBar

def plotMoments(Case):
   
    # Only root processor does something   
    if not par.irank == par.iroot:
        return 

    # Filename 
    dataPath = Case['dataFilename'].split('/')
    indexPrefix = dataPath[-1].index('.tfrecord')
    dataPath[-1] = 'model' + str(Case['SEmodel']) + '_' + dataPath[-1][:indexPrefix] + "_diversity.tfrecord"
    filenameToRead = os.path.join(*dataPath)

    # Initialize the tf dataset
    dsM = tf.data.TFRecordDataset(filenameToRead)
    dsM = dsM.map(tfu._parse_image_wind_diversityFull)  # parse the record

    # Size of dataset
    h_HR = Case['h_HR']    
    w_HR = Case['w_HR']    
    c_HR = Case['c_HR']    
    h_LR = Case['h_LR']    
    w_LR = Case['w_LR']    
    c_LR = Case['c_LR']    
    nSnap = Case['nSnap']    
  
    # ~~~~ Read dataset
    counter = 0
    printProgressBar(0, nSnap, prefix = 'Movie snapshot ' + str(0) + ' / ' +str(nSnap),suffix = 'Complete', length = 50)
    movieDir  = 'MovieTmp'
    os.makedirs(movieDir,exist_ok=True)
    for image_HR, image_LR, mean, std in dsM:
         index = counter
         printProgressBar(counter+1, nSnap, prefix = 'Movie snapshot ' + str(counter+1) + ' / ' +str(nSnap),suffix = 'Complete', length = 50)
         #par.printRoot(str(counter) + '/' + str(nSnap))
         # ~~~~ Prepare the data
         LR_snapshot = np.squeeze(image_LR.numpy())
         HR_snapshot = np.squeeze(image_HR.numpy())
         mean = np.squeeze(mean.numpy())
         std = np.squeeze(std.numpy())

         # ~~~~ Define grid
         x = np.linspace(0,Case['w_HR']-1,Case['w_HR'])
         y = np.linspace(0,Case['h_HR']-1,Case['h_HR'])

         plotNIm(field=[HR_snapshot[:,:,0],mean[:,:,0],std[:,:,0]], x=[x,x,x], y=[y,y,y], title=[r'$\xi_{HR}$',r'$E(\xi_{HR}|\xi_{LR})$',r'$\sigma(\xi_{HR}|\xi_{LR})$']) 
         plt.savefig(movieDir+'/im_'+str(index)+'.png')
         plt.close()
         
         counter += 1
    makeMovie(nSnap,movieDir,'conditionalMoments_model'+str(Case['SEmodel'])+'.gif',fps=4)
