import numpy as np
import sys
sys.path.append("util")
sys.path.append("SE")
import myparser as myparser
import case as case
import SE_estimator as SE_estimator
import SE_reconstructor as SE_reconstructor
import SE_visualizer as SE_visualizer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Parse input and store useful variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inpt = myparser.parseInputFile()
Case = case.setUpSE(inpt)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Main algorithm
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Stochastic estimation procedure
if Case['estimateConditionalMoments']:
    Coeff, CoeffSq, ACoeff, ASqCoeff = SE_estimator.estimate(Case)
    SE_estimator.writeEstimate(Coeff, CoeffSq, ACoeff, ASqCoeff, Case)

# Write modified TF records
if Case['outputConditionalMoments']:
    SE_reconstructor.writeTfRecords(Case)

# Plot moments from TF records
if Case['plotConditionalMoments']:
    SE_visualizer.plotMoments(Case)
