import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append("util")
sys.path.append("partools")
import parallel as par
import tensorflowUtils as tfu
from tensorflow.keras.models import load_model

def readDataSE(filename,Case):
    # Initialize the tf dataset
    ds = tf.data.TFRecordDataset(filename)
    ds = ds.map(tfu._parse_image_wind_function)  # parse the record
      
    Case['ds'] = ds

    # Determine size of dataset
    nSnap = 0
    h_HR=0
    w_HR=0
    c_HR=0
    h_LR=0
    w_LR=0
    c_LR=0
    if par.irank == par.iroot:
        for image_HR, image_LR in ds:
            nSnap = nSnap+1
            if nSnap==1:
                HR_snapshot = np.squeeze(image_HR.numpy())
                LR_snapshot = np.squeeze(image_LR.numpy())
                h_HR = HR_snapshot.shape[0]
                w_HR = HR_snapshot.shape[1]
                c_HR = HR_snapshot.shape[2]
                h_LR = LR_snapshot.shape[0]
                w_LR = LR_snapshot.shape[1]
                c_LR = LR_snapshot.shape[2]
   
    # Every rank knows dataset size
    Case['nSnap'] = par.bcast(nSnap)
    Case['h_HR'] = par.bcast(h_HR)
    Case['w_HR'] = par.bcast(w_HR)
    Case['c_HR'] = par.bcast(c_HR)
    Case['h_LR'] = par.bcast(h_LR)
    Case['w_LR'] = par.bcast(w_LR)
    Case['c_LR'] = par.bcast(c_LR)

def readDataNN(Case):
    # Initialize the tf dataset
    dsTrain = tf.data.TFRecordDataset(Case['dataFilenameTrain'])
    dsTrain = dsTrain.map(tfu._parse_image_wind_SFfunction)  # parse the record
      
    Case['dsTrain'] = dsTrain

    dsTest = tf.data.TFRecordDataset(Case['dataFilenameTest'])
    dsTest = dsTest.map(tfu._parse_image_wind_SFfunction)  # parse the record
      
    Case['dsTest'] = dsTest

    # Determine size of dataset
    nSnapTest = 0
    nSnapTrain = 0
    h_HR=0
    w_HR=0
    c_HR=0
    h_LR=0
    w_LR=0
    c_LR=0
    for image_LR, image_HR in dsTrain:
        nSnapTrain = nSnapTrain+1
        if nSnapTrain==1:
            HR_snapshot = np.squeeze(image_HR.numpy())
            LR_snapshot = np.squeeze(image_LR.numpy())
            h_HR = HR_snapshot.shape[0]
            w_HR = HR_snapshot.shape[1]
            c_HR = HR_snapshot.shape[2]
            h_LR = LR_snapshot.shape[0]
            w_LR = LR_snapshot.shape[1]
            c_LR = LR_snapshot.shape[2]
   
    # Every rank knows dataset size
    Case['nSnapTrain'] = nSnapTrain
    Case['h_HR'] = h_HR
    Case['w_HR'] = w_HR
    Case['c_HR'] = c_HR
    Case['h_LR'] = h_LR
    Case['w_LR'] = w_LR
    Case['c_LR'] = c_LR

    for _, _ in dsTest:
        nSnapTest = nSnapTest+1
   
    # Every rank knows dataset size
    Case['nSnapTest'] = nSnapTest

def setUpSE(inpt):
    Case = {}
    
    # What to do 
    Case['estimateConditionalMoments'] = (inpt['estimateConditionalMoments']=='True')
    Case['outputConditionalMoments'] = (inpt['outputConditionalMoments']=='True')
    Case['plotConditionalMoments'] = (inpt['plotConditionalMoments']=='True')
    
    # Only linear terms?
    try:
        Case['LinearSE'] = (inpt['LinearSE']=='True')
    except KeyError:
        Case['LinearSE'] = False

    if Case['LinearSE']:
        # What SE model to use [1-5]
        Case['SEmodel'] = int(inpt['SEmodel'])
        # How many neighbours level
        if Case['SEmodel'] <=1:
            Case['neighbourLevels']=0
        if Case['SEmodel'] >1 and Case['SEmodel'] <=3:
            Case['neighbourLevels']=1
        if Case['SEmodel'] >3:
            Case['neighbourLevels']=2
        

    else:
        # What SE model to use [1-14]
        Case['SEmodel'] = int(inpt['SEmodel'])
        # How many neighbours level
        if Case['SEmodel'] <=3:
            Case['neighbourLevels']=0
        if Case['SEmodel'] >3 and Case['SEmodel'] <=7:
            Case['neighbourLevels']=1
        if Case['SEmodel'] >7:
            Case['neighbourLevels']=2

    # Name of the TF record that contains the data
    Case['dataFilename'] = inpt['dataFilename'] 

    # Read dataset
    readDataSE(Case['dataFilename'],Case)
    # Partition files across processors
    Case['nSnap_'], Case['startSnap_'] = par.partitionFiles(Case['nSnap'])
 
    # Filter width
    Case['prescribeFW'] = (inpt['prescribeFW']=='True') 
    if Case['prescribeFW']:
       Case['K_cutoffH'] = float(inpt['K_cutoffH'])
       Case['K_cutoffW'] = float(inpt['K_cutoffW'])

    # Size of stencil
    if Case['prescribeFW']:
       Case['boxSizeH']=int((Case['h_HR']//2)//Case['K_cutoffH'])
       Case['boxSizeW']=int((Case['w_HR']//2)//Case['K_cutoffW'])
    else:
       Case['boxSizeH']=int(Case['h_HR']//Case['h_LR'])
       Case['boxSizeW']=int(Case['w_HR']//Case['w_LR'])
    
    # Block that describes what part of the domain to treat
    Case['minIndX'] = int(inpt['minIndX'])
    Case['maxIndX'] = int(inpt['maxIndX'])
    Case['minIndY'] = int(inpt['minIndY'])
    Case['maxIndY'] = int(inpt['maxIndY'])

    # Do we restart a run? If yes, clean
    Case['WipePreviousCoeff'] = False
    if Case['estimateConditionalMoments']: 
        if Case['minIndX'] == 0 and  Case['maxIndX'] == Case['w_HR']-1 and Case['minIndY'] == 0 and  Case['maxIndY'] == Case['h_HR']-1:
           par.printRoot('ASSUMED NEW COMPUTATION FROM SCRATCH: CLEANED PREVIOUS DATA') 
           Case['WipePreviousCoeff'] = True
        else:
           par.printRoot('ASSUMED CONTINUED COMPUTATION: APPEND TO PREVIOUS DATA') 
    
    # No parallelization accross channels
    Case['minIndComp'] = 0
    Case['maxIndComp'] = 1

    
    # Input or Output folder that contains the polynomial coeff
    Case['coeffFolder'] = inpt['coeffFolder']
    # Set up output folder
    os.makedirs(Case['coeffFolder'],exist_ok=True)
    Case['coeffFile'] = "coeffs_model"+str(Case['SEmodel'])+".hdf5"
    # If necessary, remove the coeff file previously written
    if Case['WipePreviousCoeff']:
        if par.irank==par.iroot:
            try:
                os.remove(Case['coeffFolder']+"/"+Case['coeffFile'])
            except OSError as error:
                par.printRoot(error)
    par.comm.Barrier()
 
    return Case 

def setUpNN(inpt):
    Case = {}
 
    if par.nProc>1:
        printAll('ERROR: NN assisted estimation only meant for serial execution')
        sys.exit()

    # What to do 
    Case['estimateConditionalMoments'] = (inpt['estimateConditionalMoments']=='True')
    Case['outputConditionalMoments'] = (inpt['outputConditionalMoments']=='True')
    Case['plotConditionalMoments'] = (inpt['plotConditionalMoments']=='True')
    

    # Name of the TF record that contains the data
    Case['dataFilenameTrain'] = inpt['dataFilenameTrain'] 
    Case['dataFilenameTest'] = inpt['dataFilenameTest'] 
 
    # Read dataset
    readDataNN(Case)

    # ResBlock Architecture
    Case['numFilters'] = int(inpt['numFilters'])
    Case['numBlocks'] = int(inpt['numBlocks'])

    # Get model for moments
    if Case['outputConditionalMoments']:
        Case['modelFirstMoment'] = load_model(inpt['modelFirstMoment'])
        Case['modelSecondMoment'] = load_model(inpt['modelSecondMoment'])

    return Case 
