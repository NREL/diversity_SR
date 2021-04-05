import itertools
import time
import h5py
import sys
import numpy as np
sys.path.append('util')
sys.path.append('partools')
sys.path.append('scitools')
import parallel as par
from filters import boxFilter2D, upSample2D
import SE_models_wind as SE_models
from myProgressBar import printProgressBar

def estimate(Case):
    # Init the model details
    ResolvedVarNames, SETermsRef, SETermsNames = SE_models.initSEModel(Case)
    Case['SETermsNames'] = SETermsNames    

    # Size of block of variables
    nIndX = Case['maxIndX']-Case['minIndX']+1
    nIndY = Case['maxIndY']-Case['minIndY']+1
    nIndComp = Case['maxIndComp']-Case['minIndComp']+1
    startSnap_ = Case['startSnap_']
    nSnap_ = Case['nSnap_']    
    nSnap = Case['nSnap']    
    nTerms = len(SETermsNames)
    minIndX = Case['minIndX']
    maxIndX = Case['maxIndX']
    minIndY = Case['minIndY']
    maxIndY = Case['maxIndY']
    minIndComp = Case['minIndComp']
    maxIndComp = Case['maxIndComp']
    
    # Unresolved variables
    Xu_ = np.zeros((nSnap_,nIndX,nIndY,nIndComp))
    XuSum_ = np.zeros((1,nIndX,nIndY,nIndComp))     
    XuSq_ = np.zeros((nSnap_,nIndX,nIndY,nIndComp))
    XuSqSum_ = np.zeros((1,nIndX,nIndY,nIndComp))
    # Resolved variables
    Xr_ = np.zeros((nSnap_,nIndX,nIndY,len(ResolvedVarNames)))
    
    # Ensemble of matrices to invert (nIndX*nIndY) matrices of size (nTerms*nTerms)
    matA = np.zeros((nIndX,nIndY,nTerms,nTerms))
    # RHS for 1st moment
    Res = np.zeros((nIndX,nIndY,nIndComp,nTerms))
    # RHS for 2nd moment
    ResSq = np.zeros((nIndX,nIndY,nIndComp,nTerms))
    
    # Polynomial terms
    SETerms_ = np.zeros((nSnap_,nIndX,nIndY,nTerms))
    SETermsSum_ = np.zeros((1,nIndX,nIndY,nTerms))
   
 
    # Loop over snapshots, fill resolved arrays, unresolved arrays, and SE terms 
    counter = 0
    printProgressBar(0, nSnap_, prefix = 'Gather data for snapshots ' + str(0) + ' / ' +str(nSnap_),suffix = 'Complete', length = 50)
    for image_HR, image_LR in Case['ds']:
        # Each processor selects a particular chunk of snapshots
        if counter>=startSnap_ and counter<startSnap_+nSnap_: 
           
            # Log progress
            printProgressBar(counter+1, nSnap_, prefix = 'Gather data for snapshots ' + str(counter+1) + ' / ' +str(nSnap_),suffix = 'Complete', length = 50)
            #par.printRoot(str(counter) + '/' + str(nSnap_))
            LR_snapshot = np.squeeze(image_LR.numpy())
    
            # Create the subfilter field
            if Case['prescribeFW']:
                filtField, subfiltField, _ = boxFilter2D(HR_snapshot,boxSizeH,boxSizeW)
            else:
                HR_snapshot = np.squeeze(image_HR.numpy())
                filtField, subfiltField, _ = upSample2D(HR_snapshot,LR_snapshot)
    
            # Create list of resolved var
            ResolvedVar = SE_models.createResolvedVar(filtField,Case)
    
            # Fill Resolved/Unresolved array
            Xu_[counter-startSnap_,:,:,:] = subfiltField[minIndX:maxIndX+1,minIndY:maxIndY+1,minIndComp:maxIndComp+1] 
            XuSq_[counter-startSnap_,:,:,:] = subfiltField[minIndX:maxIndX+1,minIndY:maxIndY+1,minIndComp:maxIndComp+1]**2
            for i in range(len(ResolvedVar)):
                Xr_[counter-startSnap_,:,:,i] = ResolvedVar[i][minIndX:maxIndX+1,minIndY:maxIndY+1] 
    
            # Fill SETerms Array
            for iref, ref in enumerate(SETermsRef):
                tmp1 = np.ones((nIndX,nIndY))
                tmp2 = np.ones((nIndX,nIndY))
                tmp3 = np.ones((nIndX,nIndY))
                tmp4 = np.ones((nIndX,nIndY))
    
                if not ref[0] == -1:
                    tmp1 = Xr_[counter-startSnap_,:,:,ref[0]]
                if not ref[1] == -1:
                    tmp2 = Xr_[counter-startSnap_,:,:,ref[1]]
                if not ref[2] == -1:
                    tmp3 = Xr_[counter-startSnap_,:,:,ref[2]]
                if not ref[3] == -1:
                    tmp4 = Xr_[counter-startSnap_,:,:,ref[3]]
    
                SETerms_[counter-startSnap_,:,:,iref] = tmp1*tmp2*tmp3*tmp4
    
        elif counter==startSnap_+nSnap_:
            break
        counter=counter+1
    
    # Get XuMean and SEMean: necessary for the constant term in the stochastic estimation procedure
    XuSum_ = np.sum(Xu_,axis=0,keepdims=True) 
    XuSqSum_ = np.sum(XuSq_,axis=0,keepdims=True) 
    SETermsSum_ = np.sum(SETerms_,axis=0,keepdims=True) 
    XuMean = par.allsumMultiDArrays(XuSum_)/nSnap
    XuSqMean = par.allsumMultiDArrays(XuSqSum_)/nSnap
    SEMean = par.allsumMultiDArrays(SETermsSum_)/nSnap
    
    # Rescale by mean
    Xu_ = Xu_ - XuMean
    XuSq_ = XuSq_ - XuSqMean
    SETerms_ = SETerms_ - SEMean
    
    
    # Assemble matrices
    printProgressBar(0, nTerms*nTerms, prefix = 'Compute SE terms ' + str(0) + ' / ' +str(nTerms*nTerms),suffix = 'Complete', length = 50)
    for i in range(nTerms):
        for j in range(nTerms):
            printProgressBar(i*nTerms + j + 1, nTerms*nTerms, prefix = 'Compute SE terms ' + str(i*nTerms + j+1) + ' / ' +str(nTerms*nTerms),suffix = 'Complete', length = 50)
            #par.printRoot('i ' + str(i) + ' j ' + str(j))
            matA[:,:,i,j] = par.allsumMultiDArrays(np.sum(SETerms_[:,:,:,i]*SETerms_[:,:,:,j],axis=0))/nSnap
    # Assemble RHS
    for i in range(nTerms):
        for i_comp in range(nIndComp):
            Res[:,:,i_comp,i] = par.allsumMultiDArrays(np.sum(SETerms_[:,:,:,i]*Xu_[:,:,:,i_comp],axis=0))/nSnap
            ResSq[:,:,i_comp,i] = par.allsumMultiDArrays(np.sum(SETerms_[:,:,:,i]*XuSq_[:,:,:,i_comp],axis=0))/nSnap
    
    # Root processor does the matrix inversion
    # Polynomial coefficients
    Coeff = None
    CoeffSq = None
    ACoeff = None
    ASqCoeff = None
    if par.irank==par.iroot:
        Coeff = np.zeros((nIndX,nIndY,nIndComp,nTerms)) 
        CoeffSq = np.zeros((nIndX,nIndY,nIndComp,nTerms)) 
        ACoeff = np.zeros((nIndX,nIndY,nIndComp,1)) 
        ASqCoeff = np.zeros((nIndX,nIndY,nIndComp,1)) 
        for i_SFS in range(nIndX):
            for j_SFS in range(nIndY):
                mat = np.squeeze(matA[i_SFS,j_SFS,:,:])
                SEm = np.array(SEMean[0,i_SFS,j_SFS,:])
                for comp_SFS in range(nIndComp):
                    R = np.squeeze(Res[i_SFS,j_SFS,comp_SFS,:])
                    RSq = np.squeeze(ResSq[i_SFS,j_SFS,comp_SFS,:])
                    Xum = XuMean[0,i_SFS,j_SFS,comp_SFS]
                    Xusqm = XuSqMean[0,i_SFS,j_SFS,comp_SFS]
                    Coeff[i_SFS,j_SFS,comp_SFS] = np.linalg.lstsq(mat,R,rcond=None)[0]
                    CoeffSq[i_SFS,j_SFS,comp_SFS] = np.linalg.lstsq(mat,RSq,rcond=None)[0]
                    ACoeff[i_SFS,j_SFS,comp_SFS] = Xum - np.dot(Coeff[i_SFS,j_SFS,comp_SFS],SEm)
                    ASqCoeff[i_SFS,j_SFS,comp_SFS] = Xusqm - np.dot(CoeffSq[i_SFS,j_SFS,comp_SFS],SEm)
    
    return Coeff, CoeffSq, ACoeff, ASqCoeff 


def writeEstimate(Coeff, CoeffSq, ACoeff, ASqCoeff,Case):  
    # Root processor outputs
    if par.irank==par.iroot:
        # Size of block of variables
        minIndX = Case['minIndX']
        maxIndX = Case['maxIndX']
        minIndY = Case['minIndY']
        maxIndY = Case['maxIndY']
        minIndComp = Case['minIndComp']
        maxIndComp = Case['maxIndComp']
        nIndX = Case['maxIndX']-Case['minIndX']+1
        nIndY = Case['maxIndY']-Case['minIndY']+1
        nIndComp = Case['maxIndComp']-Case['minIndComp']+1 
        nTerms = len(Case['SETermsNames'])
        printProgressBar(0, nIndX*nIndY*nIndComp, prefix = 'Output SE coeff ' + str(0) + ' / ' +str(nIndX*nIndY*nIndComp),suffix = 'Complete', length = 50) 
        counter = 0
        for i_SFS in range(nIndX):
            for j_SFS in range(nIndY):
                for comp_SFS in range(nIndComp):   
                    i_SFS_glob = i_SFS + minIndX
                    j_SFS_glob = j_SFS + minIndY
                    comp_SFS_glob = comp_SFS + minIndComp
                    printProgressBar(counter + 1, nIndX*nIndY*nIndComp, prefix = 'Output SE coeff ' + str(counter+1) + ' / ' +str(nIndX*nIndY*nIndComp),suffix = 'Complete', length = 50)
                    f1 = h5py.File(Case['coeffFolder']+"/"+Case['coeffFile'], "a")
                    dataName = 'Coeff_i'+str(i_SFS_glob)+'_j'+str(j_SFS_glob)+'_c'+str(comp_SFS_glob)
                    dset1 = f1.create_dataset(dataName, (nTerms,) , dtype='double', data=Coeff[i_SFS,j_SFS,comp_SFS])
                    dataName = 'CoeffSq_i'+str(i_SFS_glob)+'_j'+str(j_SFS_glob)+'_c'+str(comp_SFS_glob)
                    dset2 = f1.create_dataset(dataName, (nTerms,) , dtype='double', data=CoeffSq[i_SFS,j_SFS,comp_SFS])
                    dataName = 'A_i'+str(i_SFS_glob)+'_j'+str(j_SFS_glob)+'_c'+str(comp_SFS_glob)
                    dset3 = f1.create_dataset(dataName, (1,) , dtype='double', data=ACoeff[i_SFS,j_SFS,comp_SFS])
                    dataName = 'ASq_i'+str(i_SFS_glob)+'_j'+str(j_SFS_glob)+'_c'+str(comp_SFS_glob)
                    dset4 = f1.create_dataset(dataName, (1,) , dtype='double', data=ASqCoeff[i_SFS,j_SFS,comp_SFS])
                    f1.close()   
                    counter += 1
     
                    # Write the names of the polynomes once
                    if i_SFS_glob==0 and j_SFS_glob==0 and comp_SFS_glob==0:
                        with open(Case['coeffFolder']+'/Names_model'+str(Case['SEmodel']), 'w+') as f:
                            for terms in Case['SETermsNames']:
                                f.write("%s\n" % terms)
