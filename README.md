Copyright (c) 2020 Alliance for Sustainable Energy, LLC

# NREL Super-resolution (SR) Repo

This repository contains deep learning tools for enhancing the spatial resolution of wind data. The software is developed in Python using the TensorFlow deep learning package. Models for diversity super-resolution is provided. Included in the package are pretrained models with example code/data to perform the super-resolution as well as tools of training models for different enhancement- or data-types.


## Diversity Super-resolution

M. Hassanaly, A. Glaws, K. Stengel, and R. N. King. <em>Adversarial sampling of unknown and high-dimensional conditional distributions</em>. In review.

The super-resolution is an inherently ill-conditioned problem, with multiple high-resolution fields plausibly mapping to the same coarse field. Considitional GANs provide a framework for generating a distribution of high-resolution realizations from a given low-resolution input. Stochastic estimation is used to inform the network of the expected degree and location of sub-grid diversity. The package includes a pretrained network to generate distributions of 10x-enhanced fields of wind data.


