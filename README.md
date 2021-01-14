# Scaling Up Bayesian Uncertainty Quantification for Inverse Problems using Deep Neural Networks

## Bayesian UQ with <u>D</u>imension <u>R</u>educed <u>E</u>mulative <u>A</u>utoEncoder <u>M</u>onte Carlo (DREAM) using CNN and AutoEncoder 

### software preparation
* [**FEniCS**](https://fenicsproject.org) Go to this [webpage](https://fenicsproject.org/download/) for installation.

* [**hIPPYlib**](https://hippylib.github.io) Go to this [webpage](https://hippylib.readthedocs.io/en/3.0.0/installation.html) for installation.

* [**TensorFlow**](https://www.tensorflow.org) Go to this [webpage](https://www.tensorflow.org/install/pip) for installation.

* It is strongly **recommended** to use *Conda* to install `FEniCS` and *pip* to install `hIPPYlib` and `TensorFlow`. 
Load `FEniCS` environment in terminal session and include `hIPPYlib` and `TensorFlow` in PYTHONPATH for that session.

### package structure
* **ad_diff** contains files for advection-diffusion inverse problem.
	* `run_advdiff_geoinfMC.py` to collect samples using original MCMC algorithms.
	* `run_advdiff_EnK.py` to collect ensembles.
	* `prep_traindata.py` to prepare training samples.
	* `train_XX` to train different neural networks.
	* `run_advdiff_einfGMC.py` to collect samples using emulative algorithms (no autoencoder).
	* `run_advdiff_DREAM.py` to collect samples using DREAM algorithms.
* **elliptic_inverse** contains files for elliptic inverse problem.
	* `run_elliptic_geoinfMC.py` to collect samples using original MCMC algorithms.
	* `run_elliptic_EnK.py` to collect ensembles.
	* `prep_traindata.py` to prepare training samples.
	* `train_XX` to train different neural networks.
	* `run_elliptic_einfGMC.py` to collect samples using emulative algorithms (no autoencoder).
	* `run_elliptic_DREAM.py` to collect samples using DREAM algorithms.
* **gp** contains class definition of Gaussian process emulator.
	* `multiGP.py`: Multi-output Gaussian process (using [**GPflow**](https://gpflow.readthedocs.io/en/master/))
* **nn** contains classes defining different neural networks.
	* `ae.py`: AutoEncoder
	* `cae.py`: Convolutional AutoEncoder
	* `cnn.py`: Convolutional Neural Network
	* `dnn.py`: Densely connected Neural Network
	* `vae.py`: Variational AutoEncoder
* **optimizer** contains ensemble Kalman algorithms as optimization (EKI) or approximate sampling (EKS) methods.
	* `EnK.py`: Ensemble Kalman algorithms
* **sampler** contains different MCMC algorithms
	* `geoinfMC_dolfin.py`: infinite-dimensional Geometric Monte Carlo (original)
	* `einfGMC_dolfin.py`: Emulative infinite Geometric Monte Carlo (no autoencoder)
	* `DREAM_dolfin.py`: Dimension Reduced Emulative AutoEncoder Monte Carlo (DREAM)
* **util** contains utility functions supplementary to dolfin package in `FEniCS`.