#!/usr/bin/env python
"""
Convolutional Neural Network
Shiwei Lan @ASU, 2020
--------------------------------------
Standard AutoEncoder in TensorFlow 2.2
--------------------
Created June 4, 2020
"""
__author__ = "Shiwei Lan; Shuyi Li"
__copyright__ = "Copyright 2020"
__license__ = "GPL"
__version__ = "0.4"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from tensorflow.keras.models import Sequential
# from tensorflow.keras.models import Model
# from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping


class CNN:
    def __init__(self, input_shape, output_dim, num_filters=[16,32], kernel_size=(3,3), pool_size=(2,2), strides=(1,1), **kwargs):
        """
        Convolutional Neural Network
        --------------------------------------------------------------------------------
        input_shape: input shape (im_sz, chnl)
        output_dim: the dimension of the output space
        num_filters: list of filter sizes of Conv2D
        kernel_size: kernel size of Conv2D
        pool_size: pool size of MaxPooling2D
        strides: strides of Conv2D/MaxPooling2D
        activations: specification of activation functions, can be a list of strings or Keras activation layers
        latent_dim: the dimension of the latent space
        padding: padding of Conv2D/MaxPooling2D
        droprate: the rate of Dropout
        """
        self.input_shape=input_shape
        self.output_dim=output_dim
        self.num_filters = num_filters
        self.conv_depth = len(self.num_filters)
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.strides = strides
        self.activations = kwargs.pop('activations',{'conv':'relu','latent':'linear','output':'softmax'})
        self.latent_dim = kwargs.pop('latent_dim',self.input_shape[0]+self.output_dim)
        self.padding = kwargs.pop('padding','valid')
        self.droprate = kwargs.pop('droprate',0)
        # build neural network
        self.build(**kwargs)
    
    def _set_layers(self, model):
        """
        Set network layers
        """
        for i in range(self.conv_depth):
            model.add(Conv2D(filters=self.num_filters[i], kernel_size=self.kernel_size, strides=self.strides, 
                             activation=self.activations['conv'], name='conv_layer{}'.format(i)))
            model.add(MaxPooling2D(pool_size=self.pool_size, padding=self.padding,name='pool_layer{}'.format(i)))
        model.add(Flatten())
        if callable(self.activations['latent']):
            model.add(Dense(units=self.latent_dim,name='latent'))
            model.add(self.activations['latent'])
        else:
            model.add(Dense(units=self.latent_dim,activation=self.activations['latent'],name='latent'))
        if self.droprate:
            model.add(Dropout(rate=self.droprate))
        model.add(Dense(units=self.output_dim,activation=self.activations['output'],name='output'))
        return model
    
#     def _set_layers(self, input):
#         """
#         Set network layers
#         """
#         output=input
#         for i in range(self.conv_depth):
#             output=Conv2D(filters=self.num_filters[i], kernel_size=self.kernel_size, strides=self.strides, 
#                           activation=self.activations['conv'], name='conv_layer{}'.format(i))(output)
#             output=MaxPooling2D(pool_size=self.pool_size, padding=self.padding,name='pool_layer{}'.format(i))(output)
#         output=Flatten()(output)
#         if callable(self.activations['latent']):
#             output=Dense(units=self.latent_dim,name='latent')(output)
#             output=self.activations['latent'](output)
#         else:
#             output=Dense(units=self.latent_dim,activation=self.activations['latent'],name='latent')(output)
#         if self.droprate:
#             output=Dropout(rate=self.droprate)(output)
#         output=Dense(units=self.output_dim,activation=self.activations['output'],name='output')(output)
#         return output
    
    def _custom_loss(self,loss_f):
        """
        Wrapper to customize loss function (on latent space)
        """
        def loss(y_true, y_pred):
#             L=tf.keras.losses.MSE(y_true, y_pred)
            L=loss_f(y_true,y_pred)[0] # diff in potential
            L+=tf.math.reduce_sum(tf.math.reduce_sum(self.batch_jacobian()*loss_f(y_true,y_pred)[1][:,:,None,None,None],axis=1)**2,axis=[1,2,3]) # diff in gradient potential
            return L
        return loss
    
    def build(self,**kwargs):
        """
        Set up the network structure and compile the model with optimizer, loss and metrics.
        """
        # initialize model
        input = Input(shape=self.input_shape, name='image_input')
        model = Sequential([input])
        # set model layers
        self.model = self._set_layers(model)
#         output = self._set_layers(input)
#         self.model = Model(input, output, name='cnn')
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
    
    def save(self, savepath='./',filename='cnn_model'):
        """
        Save the trained model for future use
        """
        import os
        self.model.save(os.path.join(savepath,filename+'.h5'))
    
    def evaluate(self, input):
        """
        Output model prediction
        """
        assert input.shape[1:]==self.input_shape, 'Wrong image shape!'
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
    from util.dolfin_gadget import vec2fun,fun2img,img2fun
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
    X-=np.nanmin(X,axis=np.arange(X.ndim)[1:])
    X/=np.nanmax(X,axis=np.arange(X.ndim)[1:])
    X=X[:,:,:,None]
    # split train/test
    num_samp=X.shape[0]
    n_tr=np.int(num_samp*.75)
    x_train,y_train=X[:n_tr],Y[:n_tr]
    x_test,y_test=X[n_tr:],Y[n_tr:]
    
    # define CNN
    num_filters=[16,32]
    activations={'conv':'relu','latent':tf.keras.layers.PReLU(),'output':'linear'}
    latent_dim=128
    droprate=.25
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    cnn=CNN(x_train.shape[1:], y_train.shape[1], num_filters=num_filters,
            latent_dim=latent_dim, activations=activations, droprate=droprate, optimizer=optimizer)
    try:
        cnn.model=load_model('./result/cnn_'+algs[alg_no]+'.h5')
        print('cnn_'+algs[alg_no]+'.h5'+' has been loaded!')
    except Exception as err:
        print(err)
        print('Train CNN...\n')
        epochs=100
        import timeit
        t_start=timeit.default_timer()
        cnn.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1)
        t_used=timeit.default_timer()-t_start
        print('\nTime used for training CNN: {}'.format(t_used))
        # save CNN
#         cnn.model.save('./result/cnn_model.h5')
        cnn.save('./result','cnn_'+algs[alg_no])
        # how to laod model
#         from tensorflow.keras.models import load_model
#         reconstructed_model=load_model('XX_model.h5')
    
    # some more test
    loglik = lambda x: 0.5*elliptic.misfit.prec*tf.math.reduce_sum((cnn.model(x)-elliptic.misfit.obs)**2,axis=1)
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
        u_img=fun2img(vec2fun(u,elliptic.pde.V))
        dll_emul = cnn.gradient(u_img[None,:,:,None], loglik)
        t_used[1] += timeit.default_timer()-t_start
        # test difference
        dif = dll_xact - dll_emul.img2fun(dll_emul,elliptic.pde.V).vector()
        print('Difference between the calculated and emulated gradients: min ({}), med ({}), max ({})'.format(dif.min(),np.median(dif.get_local()),dif.max()))
        
#         # check the gradient extracted from emulation
#         v=elliptic.prior.sample()
#         v_img=fun2img(vec2fun(v,elliptic.pde.V))
#         h=1e-4
#         dll_emul_fd_v=(loglik(u_img[None,:,:,None]+h*v_img[None,:,:,None])-loglik(u_img[None,:,:,None]))/h
#         reldif = abs(dll_emul_fd_v - dll_emul.flatten().dot(v_img.flatten()))/v.norm('l2')
#         print('Relative difference between finite difference and extracted results: {}'.format(reldif))
        
        # plot
        plt.subplot(121)
        u_f.vector().set_local(dll_xact)
        df.plot(u_f)
        plt.title('Calculated')
        plt.subplot(122)
        u_f=img2fun(dll_emul)
        df.plot(u_f)
        plt.title('Emulated')
        plt.draw()
        plt.pause(1.0/30.0)
        
    print('Time used to calculate vs emulate gradients: {} vs {}'.format(t_used))
    