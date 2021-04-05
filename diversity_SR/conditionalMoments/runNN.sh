# Generate a tfrecord were the first label are the low resolution files and the second are the subfilter (SF) files
# Note that this needs to be done for both the testing and training tfrecords. For now, these are assigned the same name in the script 
python generateSFTFrecords.py

# Estimate the conditional first moment with 32 filters and 16 blocks
# This step can be done on GPU
python main_NN.py INPUTS/inputNN1

# Generate a tfrecord were the first label are the low resolution files and the second are the subfilter (SF) files minus estimated mean, squared
# The estimated mean is given with a NN that was trained before. In the script, we provide pretrained weights
# Note that this needs to be done for both the testing and training tfrecords. For now, these are assigned the same name in the script 
python generateSFSQTFrecords.py

# Estimate the conditional second moment with 16 filters and 8 blocks
# This step can be done on GPU
python main_NN.py INPUTS/inputNN2

# Write the conditional moments to a tfrecord for using it with the CGAN
# This will output a file of the form FOLDERX/modelNNFILEYdiversity.tfrecord
# We provide pretrained models for that
python main_NN.py INPUTS/inputNN3

# This will generate a movie which will show the moments generated: conditionalMoments_modelNN.gif
python main_NN.py INPUTS/inputNN4








