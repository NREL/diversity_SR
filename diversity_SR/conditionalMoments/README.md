### Requirements
- Python v3.7.6
- TensorFlow v2.3.0
- numpy v1.19.1
- matplotlib v3.2.2

### Purpose
Estimate conditional moments of a distribution with stochastic estimation and neural nets

### Data
Example low-resolution (LR) and high-resolution (HR) data (obtained from NREL's WIND Toolkit) can be found in `DataWind/`. This data does not contain conditional moments. The goal here is to create tfrecords that contain these moments. Two methods are available for the computation of the moments: stochastic estimation and neural network assisted estimation.

## Stochastic estimation (SE)
The relevant codes are found under `SE/`. The stochastic estimation procedure requires computing ensemble averages. These averages are computed by MPI-parallelization across snapshots. Since the variables are treated independently, one can run the algorithm on a subset of variables at a time (this is called a block in the code). At every run, the coefficients are appended to a file. At the moment, the MPI-parallelization is not implemented to run on multiple blocks at once. Instead, one should run a separate simulation for each block. 
For the second moment, ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbb%7BE%7D%28%5Cxi_%7BHR%7D%5E2%7C%5Cxi_%7BLR%7D%29) is estimated

## Neural network assisted estimation (NN)
The relevant codes are found under `NN/`. The NN assisted estimation should NOT be run with MPI-parallelization. Before using the method, one should generate appropriate tfrecords. The scripts are provided to this end.
For the second moment, ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbb%7BE%7D%5Cleft%5B%20%28%5Cxi_%7BHR%7D-%5Cmathbb%7BE%7D%28%5Cxi_%7BHR%7D%7C%5Cxi_%7BLR%7D%29%29%5E2%7C%5Cxi_%7BLR%7D%20%5Cright%5D) is estimated. This requires first estimating the first moment and then using it for preparing the dataset of the second moment.

### Running
* Perform neural network assisted estimation with pretrained weights on the TRAINING data: `bash runNN_pretrained.sh`
* Perform neural network assisted estimation on the EXAMPLE data (for computational tractability): `bash runNN.sh`
* Perform stochastic estimation with model7 using pretrained coefficients on the TRAINING data: `bash runSE_pretrained.sh`
* Perform stochastic estimation with model 7 on the EXAMPLE data (for computational tractability): `bash runSE.sh`


#### Acknowledgments
This work was authored by the National Laboratory of the Rockies (NLR), operated by Alliance for Energy Innovation, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. This work was supported by the Laboratory Directed Research and Development (LDRD) Program at NLR. The research was performed using computational resources sponsored by the Department of Energy's Office of Critical Minerals and Energy Innovation (CMEI) and located at the National Laboratory of the Rockies. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.

