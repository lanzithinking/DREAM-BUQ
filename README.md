# Scaling Up Bayesian Uncertainty Quantification for Inverse Problems using Deep Neural Networks

## Bayesian UQ with CNN and AutoEncoder (in preparation)

### software preparation
* [**FEniCS**](https://fenicsproject.org) Go to this [webpage](https://fenicsproject.org/download/) for installation.

* [**TensorFlow**](https://www.tensorflow.org) Go to this [webpage](https://www.tensorflow.org/install/pip) for installation.

* It is strongly **recommended** to use *Conda* to install FEniCS and *pip* to install TensorFlow. Load *FEniCS* environment in terminal session and include *TensorFlow* in PYTHONPATH for that session.

### package structure
* *elliptic_inverse* contains files for elliptic inverse problem.
	* `run_elliptic_geoinfMC.py` to collect samples using original MCMC algorithms.
	* `run_elliptic_EnK.py` to collect ensembles.
	* `prep_traindata.py` to prepare training samples.
	* `train_XX` to train different neural networks.
	* `run_elliptic_DREAM.py` to collect samples using DREAM algorithms.
* *nn* contains classes defining different neural networks.
	* `ae.py`: AutoEncoder
	* `cae.py`: Convolutional AutoEncoder
	* `cnn.py`: Convolutional Neural Network
	* `dnn.py`: Densely connected Neural Network
	* `vae.py`: Variational AutoEncoder
* *optimizer* contains ensemble Kalman algorithms as optimization (EKI) or approximate sampling (EKS) methods.
	* `EnK.py`: Ensemble Kalman algorithms
* *sampler* contains different MCMC algorithms
	* `AEinfGMC_dolfin.py`: AutoEncoder infinite-dimensional Geometric Monte Carlo (no emulator)
	* `CAEinfGMC_dolfin.py`: Convolutional AutoEncoder infinite-dimensional Geometric Monte Carlo (no emulator)
	* `DREAM_dolfin.py`: Dimension-Reduced Emulative AutoEncoder Monte Carlo
	* `einfGMC_dolfin.py`: Emulative infinite Geometric Monte Carlo (no autoencoder)
	* `geoinfMC_dolfin.py`: infinite-dimensional Geometric Monte Carlo (original)
* *util* contains utility functions supplementary to dolfin package in *FEniCS*.
* *vb* contains variational Bayes algorithms.
	* `vb.py`: Variational Bayes