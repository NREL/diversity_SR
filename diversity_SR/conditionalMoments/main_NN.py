import numpy as np
import sys
sys.path.append("util")
sys.path.append("NN")
import myparser as myparser
import case as case
import NN_estimator as NN_estimator
import NN_reconstructor as NN_reconstructor
import NN_visualizer as NN_visualizer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Parse input and store useful variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inpt = myparser.parseInputFile()
Case = case.setUpNN(inpt)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Main algorithm
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Stochastic estimation procedure
if Case['estimateConditionalMoments']:
    model = NN_estimator.makeModel(Case)
    NN_estimator.train(model,Case)

# Write modified TF records
if Case['outputConditionalMoments']:
    NN_reconstructor.writeTfRecords(Case)

# Plot moments from TF records
if Case['plotConditionalMoments']:
    NN_visualizer.plotMoments(Case)
