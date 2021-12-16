

# Diverse Super-resolution (SR) Repo

This repository contains deep learning tools for enhancing the spatial resolution of wind data. The software is developed in Python using the TensorFlow deep learning package. Models for diversity super-resolution is provided. Included in the package are pretrained models with example code/data to perform the super-resolution as well as tools of training models for different enhancement- or data-types.

The super-resolution is an inherently ill-conditioned problem, with multiple high-resolution fields plausibly mapping to the same coarse field. Considitional GANs provide a framework for generating a distribution of high-resolution realizations from a given low-resolution input. Stochastic estimation is used to inform the network of the expected degree and location of sub-grid diversity. The package includes a pretrained network to generate distributions of 10x-enhanced fields of wind data.





## Reference

[Published paper](https://www.sciencedirect.com/science/article/pii/S0021999121007488)


[Preprint version (open access)](https://arxiv.org/abs/2111.05962)


```

@article{hassanaly2022adversarial,
  title={Adversarial sampling of unknown and high-dimensional conditional distributions},
  author={Hassanaly, Malik and Glaws, Andrew and Stengel, Karen and King, Ryan N.},
  journal={Journal of Computational Physics},
  pages={110853},
  volume={450},
  year={2022},
  publisher={Elsevier}
}

```

