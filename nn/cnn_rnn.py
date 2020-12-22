#!/usr/bin/env python
"""
Convolutional (input) - Recurrent (output) Neural Network (input)
Shiwei Lan @ASU, 2020
--------------------------
CNN-RNN in TensorFlow 2.2
-------------------------
Created December 21, 2020
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2020"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Reshape,SimpleRNN,GRU,LSTM
from tensorflow.keras.models import Model
# from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping


class CNN_RNN:
    def __init__(self, input_shape, output_shape, num_filters=[16,32], kernel_size=(3,3), pool_size=(2,2), strides=(1,1), **kwargs):
        """
        Convolutional Neural Network
        --------------------------------------------------------------------------------
        input_shape: input shape (im_sz, chnl)
        output_shape: output shape (temp_dim, spat_dim)
        num_filters: list of filter sizes of Conv2D
        kernel_size: kernel size of Conv2D
        pool_size: pool size of MaxPooling2D
        strides: strides of Conv2D/MaxPooling2D
        padding: padding of Conv2D/MaxPooling2D
        latent_dim: the dimension of the latent space
        droprate: the rate of Dropout
        activations: specification of activation functions, can be a list of strings or Keras activation layers
        kernel_initializers: kernel_initializers corresponding to activations
        """
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.num_filters = num_filters
        self.conv_depth = len(self.num_filters)
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.strides = strides
        self.padding = kwargs.pop('padding','same')
        self.latent_dim = kwargs.pop('latent_dim',self.output_shape[1])
        self.droprate = kwargs.pop('droprate',0)
        self.activations = kwargs.pop('activations',{'conv':'relu','latent':'linear','output':'sigmoid','lstm':'tanh'})
        self.kernel_initializers=kwargs.pop('kernel_initializers',{'conv':'glorot_uniform','latent':'glorot_uniform','output':'glorot_uniform'})
        # build neural network
        self.build(**kwargs)
        
    
    def _set_layers(self, input):
        """
        Set network layers
        """
        output=input
        for i in range(self.conv_depth):
            ker_ini = self.kernel_initializers['conv'](output.shape[1]*30**(i==0)) if callable(self.kernel_initializers['conv']) else self.kernel_initializers['conv']
            output=Conv2D(filters=self.num_filters[i], kernel_size=self.kernel_size, strides=self.strides, 
                          activation=self.activations['conv'], kernel_initializer=ker_ini, name='conv_layer{}'.format(i))(output)
            output=MaxPooling2D(pool_size=self.pool_size, padding=self.padding,name='pool_layer{}'.format(i))(output)
        output=Flatten()(output)
        ker_ini = self.kernel_initializers['latent'](output.shape[1]) if callable(self.kernel_initializers['latent']) else self.kernel_initializers['latent']
        if callable(self.activations['latent']):
            output=Dense(units=self.output_shape[0]*self.latent_dim, kernel_initializer=ker_ini, name='latent')(output)
            output=self.activations['latent'](output)
        else:
            output=Dense(units=self.output_shape[0]*self.latent_dim, activation=self.activations['latent'], kernel_initializer=ker_ini, name='latent')(output)
        if self.droprate:
            output=Dropout(rate=self.droprate)(output)
        ker_ini = self.kernel_initializers['output'](output.shape[1]) if callable(self.kernel_initializers['output']) else self.kernel_initializers['output']
        # transform original output(n_batch,temp_dim*spat_dim) to RNN input(n_batch,temp_dim,spat_dim)
        output = Reshape((self.output_shape[0],self.latent_dim))(output)
        recurr = {'output':SimpleRNN,'gru':GRU,'lstm':LSTM}[list(self.activations.keys())[-1]]
        if len(self.activations)==3:
            output = recurr(units=self.output_shape[1], return_sequences=True,  activation=self.activations['output'],
                            kernel_initializer=ker_ini, name='recur')(output)
        else:
            output = recurr(units=self.output_shape[1], return_sequences=True,  activation=list(self.activations.values())[-1], recurrent_activation=self.activations['output'],
                            kernel_initializer=ker_ini, name='recur')(output)
        return output

    
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
        # set model layers
        output = self._set_layers(input)
        self.model = Model(input, output, name='cnn')
        # compile model
        optimizer = kwargs.pop('optimizer','adam')
        loss = kwargs.pop('loss','mse')
        metrics = kwargs.pop('metrics',['mae'])
        self.model.compile(optimizer=optimizer, loss=self._custom_loss(loss) if callable(loss) else loss, metrics=metrics, **kwargs)
    
    def train(self, x_train, y_train, x_test=None, y_test=None, batch_size=32, epochs=100, verbose=0, **kwargs):
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
                                      shuffle=False,
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
            #where do we define self.y_train
            objf = lambda x: tf.keras.losses.MeanSquaredError(self.y_train,self.model(x))
        x = tf.Variable(input, trainable=True, dtype=tf.float32)
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
    import sys,os
    sys.path.append( "../" )
    from advdiff import advdiff
    from util.dolfin_gadget import *
    # set random seed
    seed=2020
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # define the inverse problem
    meshsz = (61,61)
    eldeg = 1
    gamma = 2.; delta = 10.
    rel_noise = .5
    nref = 1
    adif = advdiff(mesh=meshsz, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed)
    adif.prior.V=adif.prior.Vh
    adif.misfit.obs=np.array([dat.get_local() for dat in adif.misfit.d.data])
    temp_dim,spat_dim=adif.misfit.obs.shape
    # algorithms
    algs=['EKI','EKS']
    num_algs=len(algs)
    alg_no=1
    
    # load data
    ensbl_sz = 500
    folder = './train_NN_eldeg'+str(eldeg)
    loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XimgY.npz'))
    X=loaded['X']
    Y=loaded['Y']
    Y=Y.reshape((-1,temp_dim,spat_dim))
    # pre-processing: scale X to 0-1
    #X-=np.nanmin(X,axis=np.arange(X.ndim)[1:])
    #X/=np.nanmax(X,axis=np.arange(X.ndim)[1:])
    X=X[:,:,:,None]
    # split train/test
    num_samp=X.shape[0]
    n_tr=np.int(num_samp*.75)
    x_train,y_train=X[:n_tr],Y[:n_tr]
    x_test,y_test=X[n_tr:],Y[n_tr:]
    
    # define CNN-RNN
    num_filters=[16,32]
    #activations={'conv':'relu','latent':tf.keras.layers.PReLU(),'output':'linear','lstm':'tanh'}
    activations={'conv':'softplus','latent':'softmax','output':'linear','lstm':'tanh'}
#     activations={'conv':tf.math.sin,'latent':tf.math.sin,'output':'linear','lstm':'tanh'}
    latent_dim=256
    droprate=.25
    sin_init=lambda n:tf.random_uniform_initializer(minval=-tf.math.sqrt(6/n), maxval=tf.math.sqrt(6/n))
    #kernel_initializers={'conv':sin_init,'latent':sin_init,'output':'glorot_uniform'}
    kernel_initializers={'conv':'he_uniform','latent':sin_init,'output':'glorot_uniform'}
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    cnnrnn=CNN_RNN(x_train.shape[1:], y_train.shape[1:], num_filters=num_filters,latent_dim=latent_dim, droprate=droprate,
                   activations=activations, kernel_initializers=kernel_initializers, optimizer=optimizer)
    try:
#         cnnrnn.model=load_model('./result/cnnrnn_'+algs[alg_no]+'.h5')
        cnnrnn.model.load_weights('./result/cnnrnn_'+algs[alg_no]+'.h5')
        print('cnnrnn_'+algs[alg_no]+'.h5'+' has been loaded!')
    except Exception as err:
        print(err)
        print('Train CNN-RNN...\n')
        epochs=100
        import timeit
        t_start=timeit.default_timer()
        cnnrnn.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1)
        t_used=timeit.default_timer()-t_start
        print('\nTime used for training CNN-RNN: {}'.format(t_used))
        # save CNN-RNN
#         cnnrnn.model.save('./result/cnnrnn_model.h5')
#         cnnrnn.save('./result','cnnrnn_'+algs[alg_no])
        cnnrnn.model.save_weights('./result','cnnrnn_'+algs[alg_no]+'.h5')
    
    # some more test
    loglik = lambda x: -0.5*tf.math.reduce_sum((cnnrnn.model(x)-adif.misfit.obs)**2/adif.misfit.noise_variance,axis=[1,2])
    import timeit
    t_used = np.zeros((1,2))
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,6), facecolor='white')
    plt.ion()
    plt.show(block=True)
    u_f = df.Function(adif.prior.V)
    eldeg = adif.prior.V.ufl_element().degree()
    if eldeg>1:
        V_P1 = df.FunctionSpace(adif.mesh,'Lagrange',1)
        d2v = df.dof_to_vertex_map(V_P1)
        u_f1 = df.Function(V_P1)
    else:
        u_f1 = u_f
    for n in range(10):
        u=adif.prior.sample()
        # calculate gradient
        t_start=timeit.default_timer()
        ll_xact,dll_xact = adif.get_geom(u,[0,1])[:2]
        t_used[0] += timeit.default_timer()-t_start
        # emulate gradient
        t_start=timeit.default_timer()
        u_img=vec2img(u)
        dll_emul = adif.img2vec(cnnrnn.gradient(u_img[None,:,:,None], loglik))
        t_used[1] += timeit.default_timer()-t_start
        # test difference
        dif_fun = np.abs(ll_xact - ll_emul)
        if eldeg>1:
            u_f.vector().set_local(dll_xact)
            dll_xact = u_f.compute_vertex_values(adif.mesh)[d2v] # covert to dof order
        else:
            dll_xact = dll_xact.get_local()
        dif_grad = dll_xact - dll_emul
        dif[n] = np.array([dif_fun, np.linalg.norm(dif_grad)/np.linalg.norm(dll_xact)])
        print('Difference between the calculated and emulated gradients: min ({}), med ({}), max ({})'.format(dif.min(),np.median(dif.get_local()),dif.max()))
        
#         # check the gradient extracted from emulation
#         v=adif.prior.sample()
#         h=1e-4
#         dll_emul_fd_v=(logLik(u[None,:]+h*v[None,:])-logLik(u[None,:]))/h
#         reldif = abs(dll_emul_fd_v - dll_emul.flatten().dot(v))/np.linalg.norm(v)
#         print('Relative difference between finite difference and extracted results: {}'.format(reldif))
        
        # plot
        plt.subplot(121)
        u_f1.vector().set_local(dll_xact)
        df.plot(u_f1)
        plt.title('Calculated Gradient')
        plt.subplot(122)
        u_f1.vector().set_local(dll_emul)
        df.plot(u_f1)
        plt.title('Emulated Gradient')
        plt.draw()
        plt.pause(1.0/30.0)
        
    print('Time used to calculate vs emulate gradients: {} vs {}'.format(t_used))
    