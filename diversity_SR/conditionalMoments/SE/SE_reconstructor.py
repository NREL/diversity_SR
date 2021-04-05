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
import SE_models_wind as SE_models
import tensorflow as tf
import tensorflowUtils as tfu
from myProgressBar import printProgressBar

def reconstruct(A,Coeff,filtField,n_x,n_y,n_comp,Case):
    ResolvedVar = SE_models.createResolvedVar(filtField,Case) 
    fieldReconst = A.copy()
    for iref, ref in enumerate(Case['SETermsRef']):
        tmp1 = np.ones((n_x,n_y))
        tmp2 = np.ones((n_x,n_y))
        tmp3 = np.ones((n_x,n_y))
        tmp4 = np.ones((n_x,n_y))
        if not ref[0] == -1:
            tmp1 = ResolvedVar[ref[0]]
        if not ref[1] == -1:
            tmp2 = ResolvedVar[ref[1]]
        if not ref[2] == -1:
            tmp3 = ResolvedVar[ref[2]]
        if not ref[3] == -1:
            tmp4 = ResolvedVar[ref[3]]
        for icomp in range(n_comp):
            fieldReconst[:,:,icomp] = fieldReconst[:,:,icomp] +Coeff[:,:,icomp,iref]*(tmp1*tmp2*tmp3*tmp4)
    return fieldReconst

def writeTfRecords(Case):
    # Init the model details
    ResolvedVarNames, SETermsRef, SETermsNames = SE_models.initSEModel(Case)
    Case['SETermsNames'] = SETermsNames
    Case['SETermsRef'] = SETermsRef

    h_HR = Case['h_HR']    
    w_HR = Case['w_HR']    
    c_HR = Case['c_HR']    
    h_LR = Case['h_LR']    
    w_LR = Case['w_LR']    
    c_LR = Case['c_LR']    
    nSnap = Case['nSnap']    
    coeffFolder = Case['coeffFolder']
    coeffFile = Case['coeffFile']

    n_x = Case['w_HR']
    n_y = Case['h_HR']
    n_comp = Case['c_HR'] 
    nTerms = len(SETermsNames)
   
    # ~~~~ Read Coeffs  
    Coeff = np.zeros((n_x,n_y,n_comp,nTerms))
    CoeffSq = np.zeros((n_x,n_y,n_comp,nTerms))
    A = np.zeros((n_x,n_y,n_comp))
    ASq = np.zeros((n_x,n_y,n_comp))
    fr = h5py.File(coeffFolder+'/'+coeffFile, 'r')
    names = list(fr.keys())
    for i in range(n_x):
        for j in range(n_y):
            for comp in range(n_comp):
                Coeff[i,j,comp,:] = fr['Coeff_i'+str(i)+'_j'+str(j)+'_c'+str(comp)][:]
                CoeffSq[i,j,comp,:] = fr['CoeffSq_i'+str(i)+'_j'+str(j)+'_c'+str(comp)][:]
                A[i,j,comp] = fr['A_i'+str(i)+'_j'+str(j)+'_c'+str(comp)][:]
                ASq[i,j,comp] = fr['ASq_i'+str(i)+'_j'+str(j)+'_c'+str(comp)][:]
    fr.close()
    
    
    
    if (not n_x==h_HR) or (not n_y==w_HR) or (not n_comp==c_HR):
        par.printAll('only full data is outputed')
        sys.exit()
    
    
    HRSnap = np.zeros((1,h_HR,w_HR,c_HR))
    LRSnap = np.zeros((1,h_LR,w_LR,c_LR))
    stdField = np.zeros((1,h_HR,w_HR,c_HR))
    meanField = np.zeros((1,h_HR,w_HR,c_HR))
    
    
    dataPath = Case['dataFilename'].split('/')
    indexPrefix = dataPath[-1].index('.tfrecord')
    dataPath[-1] = 'model' + str(Case['SEmodel']) + '_' + dataPath[-1][:indexPrefix] + "_diversity.tfrecord"
    filenameToWrite = os.path.join(*dataPath)
   
    # ~~~~ Write TF RECORD with training data
    
    if par.irank == par.iroot:
    
        printProgressBar(0, nSnap, prefix = 'Output snapshot ' + str(0) + ' / ' +str(nSnap),suffix = 'Complete', length = 50)
        with tf.io.TFRecordWriter(filenameToWrite) as writer:
            counter=0
            for image_HR, image_LR in Case['ds']:
    
                # ~~~~ Log advancement
                printProgressBar(counter+1, nSnap, prefix = 'Output snapshot ' + str(counter+1) + ' / ' +str(nSnap),suffix = 'Complete', length = 50)
                #par.printRoot(str(counter) + '/' + str(nSnap))

                # ~~~~ Prepare the data
                LR_snapshot = np.squeeze(image_LR.numpy())
            
                # Create the subfilter field
                if Case['prescribeFW']:
                    filtField, subfiltField, _ = boxFilter2D(HR_snapshot,Case['boxSizeH'],Case['boxSizeW'])
                else:
                    HR_snapshot = np.squeeze(image_HR.numpy())
                    filtField, subfiltField, _ = upSample2D(HR_snapshot,LR_snapshot)
                
                HRSnap[0,:,:,:] = HR_snapshot
                LRSnap[0,:,:,:] = LR_snapshot
                stdField[0,:,:,:] = np.sqrt(np.clip(reconstruct(ASq,CoeffSq,filtField,n_x,n_y,n_comp,Case) - reconstruct(A,Coeff,filtField,n_x,n_y,n_comp,Case)**2,0,1000000))
                meanField[0,:,:,:] = reconstruct(A,Coeff,filtField,n_x,n_y,n_comp,Case)
            
    
                # ~~~~ Write the data
                index = counter
                data_LR = LRSnap
                data_HR = HRSnap
                mean = meanField
                std = stdField
                tf_example = tfu.diversity_image_example(index,bytes(data_LR),h_LR,w_LR,bytes(data_HR),h_HR,w_HR,c_HR,bytes(mean),bytes(std))
                writer.write(tf_example.SerializeToString())
    
                counter=counter+1
