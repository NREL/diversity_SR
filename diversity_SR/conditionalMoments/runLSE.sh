# Estimate linear SE Coefficients with model 5 over the whole domain
# In reality, doing it over the whole domain will blow up the memory
# To adress this, one can reduce the size of the blocks in the input file. 
# In the paper, this was done 25 blocks (of size 20 by 20 each)
# If needed, parallelize by calling mpiexec -np XX python main_SE.py INPUTS/inputLSE1. In the paper this was done with 144 cores
# In CoeffTestWind folder, one can find a hdf5 file with the coefficients of the SE, and a list of the polynomial terms
mpiexec -np 4 python main_SE.py INPUTS/inputLSE1

# Write the conditional moments to a tfrecord for using it with the CGAN
# This will output a file of the form FOLDERX/model5FILEYdiversity.tfrecord
python main_SE.py INPUTS/inputLSE2


# This will generate a movie which will show the moments generated: conditionalMoments_model5.gif
python main_SE.py INPUTS/inputLSE3








