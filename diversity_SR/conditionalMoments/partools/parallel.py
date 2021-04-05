from mpi4py import MPI
import numpy as np
import sys

# MPI Init
comm = MPI.COMM_WORLD
irank = comm.Get_rank()+1
iroot = 1
nProc = comm.Get_size()
status = MPI.Status()

# ~~~~ Print functions
def printRoot(description, item=None):
    if irank == iroot:
        if type(item).__name__ == 'NoneType':
            print(description)
        elif (not type(item).__name__ == 'ndarray'):
            print(description + ': ',item)
        else:
            print(description + ': ',item.tolist())
        sys.stdout.flush()
    return

def printAll(description, item=None):
    if type(item).__name__ == 'NoneType':
        print('[' + str(irank) + '] ' + description)
    elif (not type(item).__name__ == 'ndarray'):
        print('[' + str(irank) + '] ' + description + ': ',item)
    else:
        print('[' + str(irank) + '] ' + description + ': ', item.tolist())
    sys.stdout.flush()
    return

def partitionFiles(nSnap):
    # ~~~~ Partition the files with MPI
    # Simple parallelization across snapshots
    NSnapGlob = nSnap
    tmp1=0
    tmp2=0
    for iproc in range(nProc):
        tmp2 = tmp2 + tmp1
        tmp1 = int(NSnapGlob/(nProc-iproc))
        if irank == (iproc+1):
            nSnap_ = tmp1
            startSnap_ = tmp2
        NSnapGlob = NSnapGlob - tmp1
    return nSnap_, startSnap_

def gather1DList(list_,rootId,N):
    list_=np.array(list_,dtype='double')
    sendbuf = list_
    recvbuf = np.empty(N,dtype='double')
    # Collect local array sizes:
    sendcounts = comm.gather(len(sendbuf), root=rootId)
    comm.Gatherv(sendbuf, recvbuf=(recvbuf, sendcounts), root=rootId)
    return recvbuf

def gather2DList(list_,rootId,N1Loc, N1Glob, N2):
    # ~~~ The parallelization is across the axis 0
    # ~~~ This will not work if the parallelization is across axis 1
    list_=np.array(list_,dtype='double')
    # Reshape the local data matrices:
    nElements_ = N1Loc * N2
    sendbuf = np.reshape(list_, nElements_, order='C')
    # Collect local array sizes:
    sendcounts = comm.gather(len(sendbuf), root=rootId)
    # Gather the data matrix:
    recvbuf = np.empty(N1Glob*N2, dtype='double')
    comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=0)
    recvbuf = np.reshape(recvbuf, (N1Glob, N2), order='C')
    return recvbuf

def allgather1DList(list_,N):
    list_=np.array(list_,dtype='double')
    # Collect local array sizes:
    recvbuf = np.empty(N,dtype='double')
    comm.Allgatherv(list_, recvbuf)
    return recvbuf

def allsum1DArrays(A):
    buf = np.zeros(len(A),dtype='double') 
    comm.Allreduce(A, buf, op=MPI.SUM)
    return buf

def allsumMultiDArrays(A):
    # Takes a 3D array as input
    # Returns 3D array
    shapeDim = A.shape
    nTotDim = int(np.prod(shapeDim))
    buf = np.zeros(nTotDim,dtype='double') 
    comm.Allreduce(np.reshape(A,nTotDim), buf, op=MPI.SUM)
    return np.reshape(buf,shapeDim)


def allsumScalar(A):
    result = comm.allreduce(A, op=MPI.SUM)
    return result

def bcast(A):
    A = comm.bcast(A, root=0)
    return A
