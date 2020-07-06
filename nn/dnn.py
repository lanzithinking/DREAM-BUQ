#!/usr/bin/env python
"""
Deep Neural Network
Shiwei Lan @ASU, 2020
------------------------------
Standard DNN in TensorFlow 2.2
--------------------
Created June 17, 2020
"""
__author__ = "Shuyi Li; Shiwei Lan"
__copyright__ = "Copyright 2020"
__license__ = "GPL"
__version__ = "0.5"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Dropout
from tensorflow.keras.models import Model
# from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping


class DNN:
    def __init__(self, input_dim, output_dim, depth=3, **kwargs):
        """
        Deep Neural Network
        --------------------------------------------------------------------------------
        input_dim: the dimension of the input space
        output_dim: the dimension of the output space
        depth: the depth of the network
        node_sizes: sizes of the nodes of the network, which can override depth
        droprate: the rate of Dropout
        activations: specification of activation functions, can be a list of strings or Keras activation layers
        kernel_initializers: kernel_initializers corresponding to activations
        """
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.depth = depth
        self.node_sizes = kwargs.pop('node_sizes',None)
        if self.node_sizes is None:
            self.node_sizes = np.linspace(self.input_dim,self.output_dim,self.depth+1,dtype=np.int)
        else:
            self.depth=np.size(self.node_sizes)-1
        if self.node_sizes[0]!=self.input_dim or self.node_sizes[-1]!=self.output_dim:
            raise ValueError('Node sizes not matching input/output dimensions!')
        self.droprate = kwargs.pop('droprate',0)
        self.activations = kwargs.pop('activations',{'hidden':'relu','output':'linear'})
        self.kernel_initializers=kwargs.pop('kernel_initializers',{'hidden':'glorot_uniform','output':'glorot_uniform'})
        # build neural network
        self.build(**kwargs)
    
    def _set_layers(self, input):
        """
        Set network layers
        """
        output=input
        for i in range(self.depth):
            layer_name = 'output' if i==self.depth-1 else 'hidden_layer{}'.format(i)
            activation = self.activations['output'] if i==self.depth-1 else self.activations['hidden']
            ker_ini_hidden = self.kernel_initializers['hidden'](output.shape[1]*30**(i==0)) if callable(self.kernel_initializers['hidden']) else self.kernel_initializers['hidden']
            ker_ini_output = self.kernel_initializers['output'](output.shape[1]) if callable(self.kernel_initializers['output']) else self.kernel_initializers['output']
            ker_ini = ker_ini_output if i==self.depth-1 else ker_ini_hidden
            if callable(activation):
                output=Dense(units=self.node_sizes[i+1], kernel_initializer=ker_ini, name=layer_name)(output)
                output=activation(output)
            else:
                output=Dense(units=self.node_sizes[i+1], activation=activation, kernel_initializer=ker_ini, name=layer_name)(output)
            if self.droprate>0: output=Dropout(rate=self.droprate)(output)#, training=self.droprate>0) # this bug causes discrepancy between self.model(input).numpy() and self.evaluate(input)!
        return output
    
    def _custom_loss(self,loss_f):
        """
        Wrapper to customize loss function (on latent space)
        """
        def loss(y_true, y_pred):
#             L=tf.keras.losses.MSE(y_true, y_pred)
            L=loss_f(y_true,y_pred)[0] # diff in potential
#             L+=tf.math.reduce_sum(tf.math.reduce_sum(self.batch_jacobian()*loss_f(y_true,y_pred)[1][:,:,None],axis=1)**2,axis=1) # diff in gradient potential
            return L
        return loss
    
    def build(self,**kwargs):
        """
        Set up the network structure and compile the model with optimizer, loss and metrics.
        """
        # initialize model
        input = Input(shape=self.input_dim, name='input')
        # set model layers
        output = self._set_layers(input)
        self.model = Model(input, output, name='dnn')
        # compile model
        optimizer = kwargs.pop('optimizer','adam')
        loss = kwargs.pop('loss','mse')
        metrics = kwargs.pop('metrics',['mae'])
        self.model.compile(optimizer=optimizer, loss=self._custom_loss(loss) if callable(loss) else loss, metrics=metrics, **kwargs)
    
    def train(self, x_train, y_train, x_test=None, y_test=None, epochs=100, batch_size=32, verbose=0, **kwargs):
        """
        Train the model with data
        """
        num_samp=x_train.shape[0]
        if any([i is None for i in (x_test, y_test)]):
            tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
            te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
            x_test, y_test = x_train[te_idx], y_train[te_idx]
            x_train, y_train = x_train[tr_idx], y_train[tr_idx]
        patience = kwargs.pop('patience',0)
        es = EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=patience)
        self.history = self.model.fit(x_train, y_train,
                                      validation_data=(x_test, y_test),
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      callbacks=[es],
                                      verbose=verbose, **kwargs)
    
    def save(self, savepath='./',filename='dnn_model'):
        """
        Save the trained model for future use
        """
        import os
        self.model.save(os.path.join(savepath,filename+'.h5'))
    
    def evaluate(self, input):
        """
        Output model prediction
        """
        assert input.shape[1]==self.input_dim, 'Wrong input dimension!'
        return self.model.predict(input)
    
    def gradient(self, input, objf=None):
        """
        Obtain gradient of objective function wrt input
        """
        if not objf:
            objf = lambda x: tf.keras.losses.MeanSquaredError(self.y_train,self.model(x))
        x = tf.Variable(input, trainable=True)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            obj = objf(x)
        grad = tape.gradient(obj,x).numpy()
        return np.squeeze(grad)
    
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

if __name__ == '__main__':
    import dolfin as df
    import sys
    sys.path.append( "../" )
    from elliptic_inverse.Elliptic import Elliptic
    # set random seed
    np.random.seed(2020)
    
    # define the inverse problem
    nx=40; ny=40
    SNR=50
    elliptic = Elliptic(nx=nx,ny=ny,SNR=SNR)
    # algorithms
    algs=['EKI','EKS']
    num_algs=len(algs)
    alg_no=0
    
    # load data
    ensbl_sz = 100
    folder = '../elliptic_inverse/analysis_f_SNR'+str(SNR)
    loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training.npz'))
    X=loaded['X']
    Y=loaded['Y']
    # pre-processing: scale X to 0-1
#     X-=np.nanmin(X,axis=1,keepdims=True)
#     X/=np.nanmax(X,axis=1,keepdims=True)
    # split train/test
    num_samp=X.shape[0]
    n_tr=np.int(num_samp*.75)
    x_train,y_train=X[:n_tr],Y[:n_tr]
    x_test,y_test=X[n_tr:],Y[n_tr:]
    
    # define DNN
    depth=2
    # activation='linear'
    activation=tf.keras.layers.LeakyReLU(alpha=0.1)
#     activations={'hidden':tf.math.sin,'output':'linear'}
    droprate=.5
    sin_init=lambda n:tf.random_uniform_initializer(minval=-tf.math.sqrt(6/n), maxval=tf.math.sqrt(6/n))
    kernel_initializers={'hidden':sin_init,'output':'glorot_uniform'}
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    dnn=DNN(x_train.shape[1], y_train.shape[1], depth=depth, droprate=droprate,
            activation=activation, kernel_initializers=kernel_initializers, optimizer=optimizer)
    try:
        dnn.model=load_model('./result/dnn_'+algs[alg_no]+'.h5')
#         dnn.model.load_weights('./result/dnn_'+algs[alg_no]+'.h5')
        print('dnn_'+algs[alg_no]+'.h5'+' has been loaded!')
    except Exception as err:
        print(err)
        print('Train DNN...\n')
        epochs=100
        import timeit
        t_start=timeit.default_timer()
        dnn.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1)
        t_used=timeit.default_timer()-t_start
        print('\nTime used for training DNN: {}'.format(t_used))
        # save DNN
#         dnn.model.save('./result/dnn_model.h5')
        dnn.save('./result','dnn_'+algs[alg_no])
#         dnn.model.save_weights('./result/dnn_'+algs[alg_no]+'.h5')
    
    # some more test
    loglik = lambda x: -0.5*elliptic.misfit.prec*tf.math.reduce_sum((dnn.model(x)-elliptic.misfit.obs)**2,axis=1)
    import timeit
    t_used = np.zeros((1,2))
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,6), facecolor='white')
    plt.ion()
    plt.show(block=True)
    u_f = df.Function(elliptic.pde.V)
    for n in range(10):
        u=elliptic.prior.sample()
        # calculate gradient
        t_start=timeit.default_timer()
        dll_xact = elliptic.get_geom(u,[0,1])[1]
        t_used[0] += timeit.default_timer()-t_start
        # emulate gradient
        t_start=timeit.default_timer()
        dll_emul = dnn.gradient(u.get_local()[None,:], logLik)
        t_used[1] += timeit.default_timer()-t_start
        # test difference
        dif = dll_xact.get_local() - dll_emul
        print('Difference between the calculated and emulated gradients: min ({}), med ({}), max ({})'.format(dif.min(),np.median(dif),dif.max()))
        
#         # check the gradient extracted from emulation
#         v=elliptic.prior.sample()
#         h=1e-4
#         dll_emul_fd_v=(loglik(u.get_local()[None,:]+h*v.get_local()[None,:])-loglik(u.get_local()[None,:]))/h
#         reldif = abs(dll_emul_fd_v - dll_emul.flatten().dot(v.get_local()))/v.norm('l2')
#         print('Relative difference between finite difference and extracted results: {}'.format(reldif))
        
        # plot
        plt.subplot(121)
        u_f.vector().set_local(dll_xact)
        df.plot(u_f)
        plt.title('Calculated')
        plt.subplot(122)
        u_f.vector().set_local(dll_emul)
        df.plot(u_f)
        plt.title('Emulated')
        plt.draw()
        plt.pause(1.0/30.0)
        
    print('Time used to calculate vs emulate gradients: {} vs {}'.format(t_used))
    