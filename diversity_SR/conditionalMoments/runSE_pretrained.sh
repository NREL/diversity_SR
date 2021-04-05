# Estimate SE Coefficients with model 7 over the whole domain
# In reality, doing it over the whole domain will blow up the memory
# To adress this, one can reduce the size of the blocks in the input file. 
# In the paper, this was done 25 blocks (of size 20 by 20 each)
# If needed, parallelize by calling mpiexec -np XX python main_SE.py INPUTS/inputSE1. In the paper this was done with 144 cores
#python main_SE.py INPUTS/inputSE1

# Write the conditional moments to a tfrecord for using it with the CGAN
# This will output a file of the form FOLDERX/model7FILEYdiversity.tfrecord
python main_SE.py INPUTS/inputSE2PreTrained


# This will generate a movie which will show the moments generated: conditionalMoments_model7.gif
python main_SE.py INPUTS/inputSE3








