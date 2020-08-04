#!/usr/bin/env python
"""
Multi-output Gaussian Process
Shiwei Lan @ASU, 2020
-----------------------------
Standard GP model in GPflow 2
-----------------------------
Created July 29, 2020
"""
__author__ = "Shuyi Li; Shiwei Lan"
__copyright__ = "Copyright 2020"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com"

import numpy as np
import tensorflow as tf
import gpflow as gpf
from gpflow.ci_utils import ci_niter
from scipy.cluster.vq import kmeans
gpf.config.set_default_float(np.float64)

class multiGP:
    def __init__(self, input_dim, output_dim, latent_dim, **kwargs):
        """
        Multi-output Gaussian Process [Sparse Variational Gaussian Process (SVGP)]
        -------------------------------------------------------------------------------------
        refer to: https://gpflow.readthedocs.io/en/master/notebooks/advanced/multioutput.html
                  https://gpflow.readthedocs.io/en/master/notebooks/intro_to_gpflow2.html
        -------------------------------------------------------------------------------------
        input_dim: the dimension of the input space
        output_dim: the dimension of the output space
        latent_dim: the dimension of the latent space
        induce_num: the number of the inducing points
        kernel: the specification of GP kernel(s)
        shared_kernel: indicator whether kernel is shared among multiple outputs
        shared_induce: indicator whether inducing locations are shared among multiple outputs
        """
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.latent_dim=latent_dim
        self.induce_num=kwargs.pop('induce_num',np.ceil(.1*self.input_dim).astype('int'))
        self.kernel=kwargs.pop('kernel',gpf.kernels.SquaredExponential() + gpf.kernels.Linear())
        self.shared_kernel=kwargs.pop('shared_kernel',False)
        self.shared_induce=kwargs.pop('shared_induce',True)
        # build Gaussian Process Model
        self.kwargs=kwargs
        if 'x_train' in self.kwargs:
            self.build(input=self.kwargs.pop('x_train'),**self.kwargs)
    
    def _set_kernels(self):
        """
        Set kernels for multiple outputs
        """
        if self.shared_kernel:
            kernel=gpf.kernels.SharedIndependent(self.kernel,output_dim=self.output_dim)
        else:
            kern_list=[self.kernel,]*self.latent_dim
            if self.latent_dim==self.output_dim:
                kernel=gpf.kernels.SeparateIndependent(kern_list)
            else: # coregionalization model
                kernel=gpf.kernels.LinearCoregionalization(kern_list,W=np.random.randn(self.output_dim,self.latent_dim))
        return kernel
    
    def _set_inducing(self,input):
        """
        Set random inducing input locations
        """
        # initialization of inducing input locations
        Z=kmeans(input, self.induce_num)[0]
        if self.shared_induce:
            iv=gpf.inducing_variables.SharedIndependentInducingVariables(gpf.inducing_variables.InducingPoints(Z))
        else:
            iv=gpf.inducing_variables.SeparateIndependentInducingVariables([gpf.inducing_variables.InducingPoints(Zi) for Zi in [Z.copy(),]*self.output_dim])
        return iv
    
    def build(self,input,**kwargs):
        """
        Set up the network structure and compile the model with optimizer, loss and metrics.
        """
        kernel=self._set_kernels()
        inducing_variable=self._set_inducing(input)
        likelihood=kwargs.pop('likelihood',gpf.likelihoods.Gaussian())
        self.model=gpf.models.SVGP(kernel=kernel, likelihood=likelihood, inducing_variable=inducing_variable, num_latent_gps=self.latent_dim)
    
    def _optimize_model_with_scipy(self,train_data,**kwargs):
        """
        Optimize model using the Scipy optimizer in a single call
        """
        method=kwargs.pop('method',"l-bfgs-b")
        disp=kwargs.pop("disp",True)
        maxiter=kwargs.pop( "maxiter",ci_niter(200))
        optimizer = gpf.optimizers.Scipy()
        optimizer.minimize(
            self.model.training_loss_closure(train_data),
            variables=self.model.trainable_variables,
            method=method,
            options={"disp": disp, "maxiter": maxiter},
            **kwargs
        )
    
    def _optimize_model_with_tensorflow(self,train_data,**kwargs):
        """
        Optimize model using the Tensorflow builtin optimizer
        """
        batch_size = kwargs.pop('batch_size',32)
        batched_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)
        training_loss = self.model.training_loss_closure(iter(batched_dataset))
        
        optimizer = kwargs.pop('optimizer',tf.optimizers.Adam())
        epochs = kwargs.pop('epochs',100)
        logging_epoch_freq = kwargs.pop('logging_epoch_freq',1)
        test_data = kwargs.pop('test_data',None)
        for epoch in range(epochs):
            optimizer.minimize(training_loss, self.model.trainable_variables)
            
            epoch_id = epoch + 1
            if epoch_id % logging_epoch_freq == 0:
                if test_data is None:
                    tf.print(f"Epoch {epoch_id}: ELBO (train) {self.model.elbo(train_data)}")
                else:
                    tf.print(f"Epoch {epoch_id}: ELBO (train) {self.model.elbo(train_data)}; ELBO (test) {self.model.elbo(test_data)}")
    
    def _optimize_model_with_gradienttape(self,train_data,**kwargs):
        """
        Optimize model using the Tensorflow GradientTape with batch optimization
        """
        # obtain train_dataset and batches
        num_train_data = train_data[0].shape[0]
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        batch_size = kwargs.pop('batch_size',32)
        prefetch_size = tf.data.experimental.AUTOTUNE
        shuffle_buffer_size = num_train_data // 2
        num_batches_per_epoch = num_train_data // batch_size
        train_dataset = (
            train_dataset.repeat()
            .prefetch(prefetch_size)
            .shuffle(buffer_size=shuffle_buffer_size)
            .batch(batch_size)
        )
        batches = iter(train_dataset)
        
        optimizer = kwargs.pop('optimizer',tf.optimizers.Adam())
        epochs = kwargs.pop('epochs',100)
        logging_epoch_freq = kwargs.pop('logging_epoch_freq',1)
        test_data = kwargs.pop('test_data',None)
        for epoch in range(epochs):
            for _ in range(ci_niter(num_batches_per_epoch)):
                grads=self.stochastic_gradient(next(batches))
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            epoch_id = epoch + 1
            if epoch_id % logging_epoch_freq == 0:
                if test_data is None:
                    tf.print(f"Epoch {epoch_id}: ELBO (train) {self.model.elbo(train_data)}")
                else:
                    tf.print(f"Epoch {epoch_id}: ELBO (train) {self.model.elbo(train_data)}; ELBO (test) {self.model.elbo(test_data)}")
    
    def train(self, x_train, y_train, x_test=None, y_test=None, **kwargs):
        """
        Train the model with data
        """
        if any([i is None for i in (x_test, y_test)]):
            num_samp=x_train.shape[0]
            tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
            te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
            x_test, y_test = x_train[te_idx], y_train[te_idx]
            x_train, y_train = x_train[tr_idx], y_train[tr_idx]
        if not hasattr(self,'model'):
            self.build(input=x_train,**self.kwargs)
        
        if 'batch_size' in kwargs:
#             self._optimize_model_with_tensorflow(train_data=(x_train,y_train),test_data=(x_test,y_test),**kwargs)
            self._optimize_model_with_gradienttape(train_data=(x_train,y_train),test_data=(x_test,y_test),**kwargs)
        else:
            self._optimize_model_with_scipy(train_data=(x_train,y_train),**kwargs)
    
    def evaluate(self, input):
        """
        Output model prediction
        """
        assert input.shape[1]==self.input_dim, 'Wrong input dimension!'
        return self.model.predict_f(input)[0]
    
    def gradient(self, input, objf=None):
        """
        Obtain gradient of objective function wrt input
        """
        if objf is None: objf = lambda x: self.model.training_loss(x)
        x = tf.Variable(input, trainable=True)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            obj = objf(x)
        grad = tape.gradient(obj,x).numpy()
        return np.squeeze(grad)
    
    def stochastic_gradient(self, batch):
        """
        Obtain stochastic gradient of training loss wrt trainable variables
        """
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.model.training_loss(batch)
        grads = tape.gradient(loss,self.model.trainable_variables)
        return grads
    
    def jacobian(self, input):
        """
        Obtain Jacobian matrix of output wrt input
        """
#         x = tf.constant(input)
        x = tf.Variable(input, trainable=True, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            y = self.model(x)
        jac = g.jacobian(y,x).numpy()
        return np.squeeze(jac)
    
    def batch_jacobian(self, input=None):
        """
        Obtain Jacobian matrix of output wrt input
        ------------------------------------------
        Note: when using model input, it has to run with eager execution disabled in TF v2.2.0
        """
        if input is None:
            x = self.model.input
        else:
#             x = tf.constant(input)
            x = tf.Variable(input, trainable=True, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            y = self.model(x)
        jac = g.batch_jacobian(y,x)
        return jac if input is None else np.squeeze(jac.numpy())
    
    def save(self, save_dir='./result'):
        """
        Save the trained model for future use
        """
        frozen_model = gpf.utilities.freeze(self.model) #TODO: bugfix AttributeError: 'Softplus' object has no attribute '_from_x'
        module_to_save = tf.Module()
        predict_fn = tf.function(
            frozen_model.predict_f, input_signature=[tf.TensorSpec(shape=[None, self.input_dim], dtype=tf.float64)]
        )
        module_to_save.predict = predict_fn
        tf.saved_model.save(module_to_save,save_dir)
        # load model
#         loaded_model = tf.saved_model.load(save_dir)
        
#     def save_params(self,save_dir='./result',save_fname='gp_model'):
#         """
#         Save the trained model parameters for future use
#         """
#         import pickle
#         params = gpf.utilities.parameter_dict(self.model)
#         with open(os.path.join(save_dir,save_fname+'.pckl'),'wb') as f:
#             pickle.dump(params,f) # some parameters cannot be dumped
#     
#     def load_params(self,filepath='./result',filename='gp_model'):
#         """
#         Load saved model parameters for reuse
#         """
#         import pickle
#         with open(os.path.join(filepath,filename+'.pckl'),'rb') as f:
#             params = pickle.load(f)
#         gpf.utilities.multiple_assign(self.model, params)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # set random seed
    np.random.seed(2020)
    
    # Generate synthetic data
    N = 100  # number of points
    D = 1  # number of input dimensions
    M = 15  # number of inducing points
    L = 2  # number of latent GPs
    P = 3  # number of observations = output dimensions
    
    def generate_data(N=100):
        X = np.random.rand(N)[:, None] * 10 - 5  # Inputs = N x D
        G = np.hstack((0.5 * np.sin(3 * X) + X, 3.0 * np.cos(X) - X))  # G = N x L
        W = np.array([[0.5, -0.3, 1.5], [-0.4, 0.43, 0.0]])  # L x P
        F = np.matmul(G, W)  # N x P
        Y = F + np.random.randn(*F.shape) * [0.2, 0.2, 0.2]
        
        return X, Y
    
    X, Y = data = generate_data(N)
    Zinit = np.linspace(-5, 5, M)[:, None]
    
    # split train/test
    num_samp=X.shape[0]
    n_tr=np.int(num_samp*.75)
    x_train,y_train=X[:n_tr],Y[:n_tr]
    x_test,y_test=X[n_tr:],Y[n_tr:]
    
    # define GP model
    gp = multiGP(D,P,L,induce_num=M)
    try:
        gp.model=tf.saved_model.load('./result')
        print('GP model has been loaded!')
    except Exception as err:
        print(err)
        print('Train GP model...\n')
        epochs=100
        batch_size=64
        kwargs={'maxiter':epochs}
#         kwargs={'epochs':epochs,'batch_size':batch_size}
        import timeit,os
        t_start=timeit.default_timer()
        gp.train(x_train,y_train,x_test=x_test,y_test=y_test,**kwargs)
        t_used=timeit.default_timer()-t_start
        print('\nTime used for training GP: {}'.format(t_used))
        # save GP
#         save_dir='./result/GP'
#         if not os.path.exists(save_dir): os.makedirs(save_dir)
#         gp.save(save_dir)
    
    # print summary
    from gpflow.utilities import print_summary
    print_summary(gp.model)
    
    def plot_model(m, lower=-8.0, upper=8.0):
        pX = np.linspace(lower, upper, 100)[:, None]
        pY, pYv = m.predict_y(pX)
        if pY.ndim == 3:
            pY = pY[:, 0, :]
        plt.plot(X, Y, "x")
        plt.gca().set_prop_cycle(None)
        plt.plot(pX, pY)
        for i in range(pY.shape[1]):
            top = pY[:, i] + 2.0 * pYv[:, i] ** 0.5
            bot = pY[:, i] - 2.0 * pYv[:, i] ** 0.5
            plt.fill_between(pX[:, 0], top, bot, alpha=0.3)
        plt.xlabel("X")
        plt.ylabel("f")
        plt.title(f"ELBO: {m.elbo(data):.3}")
#         plt.plot(Z, Z * 0.0, "o")
    
    # plot model
    plot_model(gp.model)
    plt.show()
    