#!/usr/bin/env python
"""
Variational AutoEncoder
Shiwei Lan @ASU, 2020
--------------------------------------
Standard AutoEncoder in TensorFlow 2.2
--------------------
Created June 23, 2020
"""
__author__ = "Shiwei Lan; Shuyi Li"
__copyright__ = "Copyright 2020"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model
# from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping


class VAE:
    def __init__(self, dim, half_depth=3, latent_dim=None, **kwargs):
        """
        Variational AutoEncoder with encoder that maps inputs to latent variables and decoder that reconstructs data from latent variables.
        Heuristic structure: inputs x --(encoder)-- latent variables z --(decoder)-- reconstructions x'.
        z = mu(x) + sigma(x) * eps, where mu and sigma are output of latent layer (encoded), eps ~ N(0,I)
        -------------------------------------------------------------------------------------------------
        dim: dimension of the original (input and output) space
        half_depth: the depth of the network of encoder and decoder if a symmetric structure is imposed (by default)
        latent_dim: the dimension of the latent space
        node_sizes: sizes of the nodes of the network, which can overwrite half_depth and induce an asymmetric structure.
        repatr_out: indicator whether to reparametrize output such that x' = mu'(z) + sigma'(z) * eps
        activation: specification of activation functions, can be a string or a Keras activation layer
        kernel_initializer: kernel_initializer corresponding to activation
        """
        self.dim = dim
        self.half_depth = half_depth
        self.latent_dim = latent_dim
        if self.latent_dim is None: self.latent_dim = np.ceil(self.dim/self.half_depth).astype('int')
        self.node_sizes = kwargs.pop('node_sizes',None)
        if self.node_sizes is None or np.size(self.node_sizes)!=2*self.half_depth+1:
            self.node_sizes = np.linspace(self.dim,self.latent_dim,self.half_depth+1,dtype=np.int)
            self.node_sizes = np.concatenate((self.node_sizes,self.node_sizes[-2::-1]))
        if not np.all([self.node_sizes[i]==self.dim for i in (0,-1)]):
            raise ValueError('End node sizes not matching input/output dimensions!')
        self.repatr_out = kwargs.pop('repatr_out',False)
        self.activation = kwargs.pop('activation','linear')
        self.kernel_initializer=kwargs.pop('kernel_initializer','glorot_uniform')
        # build neural network
        self.build(**kwargs)
    
    def _set_layers(self, input, coding='encode'):
        """
        Set network layers of encoder (coding 'encode') or decoder (coding 'decode') based on given node_sizes
        """
        node_sizes = {'encode':self.node_sizes[:self.half_depth+1],'decode':self.node_sizes[self.half_depth:]}[coding]
        output = input
        for i in range(self.half_depth):
            node_sz = node_sizes[i+1]*2**((i==self.half_depth-1)*{'encode':True,'decode':self.repatr_out}[coding])
            layer_name = "{}_out".format(coding) if i==self.half_depth-1 else "{}_layer{}".format(coding,i)
            if callable(self.activation):
                output = Dense(units=node_sz, kernel_initializer=self.kernel_initializer, name=layer_name)(output)
                output = self.activation(output)
            else:
                output = Dense(units=node_sz, activation=self.activation, kernel_initializer=self.kernel_initializer, name=layer_name)(output)
        return output
    
    def reparametrize(self, input, dim=None):
        """
        Reparametrize (latent variable) z ~ N(mu(x), sigma^2(x))
        """
        if dim is None: dim = self.latent_dim
        mean, std = tf.split(input, 2, axis=1)
        eps = tf.random.normal(shape=[dim])
        output = mean + std * eps
        return output
    
    def _loss(self, beta=1, custom_loss=None):
        """
        Wrapper to customize loss function (on latent space)
        """
        if custom_loss is None: custom_loss=self._nll_loss
        def loss(y_true, y_pred):
            L=beta*self._KL_loss(y_true)
            L+=custom_loss(y_true,y_pred)
#             L+=custom_loss(self.reparametrize(self.encoder(y_true)))
            return L
        return loss
    
    def _KL_loss(self, input=None):
        """
        Kullback–Leibler between q(z|x) and p(z), regularization
        """
        if input is None: input = self.model.input
        mean, std = tf.split(self.encoder(input), 2, axis=1)
        KL = 0.5*tf.reduce_sum(mean**2 + std**2 - 2*tf.math.log(tf.math.abs(std)) - 1, axis=1)
        return KL
    
    def _nll_loss(self, *args, **kwargs):
        """
        Expectation of negative Log-likelihood -log p(x|z) wrt q(z|x)
        """
        if not self.repatr_out:
            return .5*tf.keras.losses.MSE(*args, **kwargs)
        else:
            if len(args)==2:
                input, output = args[0], args[1]
            elif len(args)==1:
                input = args
                output = kwargs.pop('output',self.model.output)
            else:
                input = kwargs.pop('input',self.model.input)
                output = kwargs.pop('output',self.model.output)
            latent = self.reparametrize(self.encoder(input))
            mean, std = tf.split(self.decoder(latent), 2, axis=1)
            norm_dec = tf.compat.v1.distributions.Normal(mean, tf.math.abs(std))
#             norm_dec = tfp.distributions.Normal(mean, tf.math.abs(std))
#             nll = -tf.reduce_sum(norm_dec.log_prob(output),axis=1)
            nll = -tf.reduce_mean(tf.reduce_sum(norm_dec.log_prob(tf.expand_dims(output,1)),axis=2),axis=0)
#             nll = tf.reduce_mean(tf.reduce_sum(.5*tf.math.log(2*np.math.pi)+tf.math.log(tf.math.abs(std))+.5*((mean-tf.expand_dims(output,1))/std)**2,axis=2),axis=0)
#             nll1 = tf.reduce_mean(tf.reduce_sum(.5*tf.math.log(2*np.math.pi)+tf.math.log(tf.math.abs(tf.expand_dims(std,0)))+.5*((tf.expand_dims(mean,0)-tf.expand_dims(output,1))/tf.expand_dims(std,0))**2,axis=2),axis=0)
        return nll
    
    def build(self,**kwargs):
        """
        Set up the network structure and compile the model with optimizer, loss and metrics.
        """
        # this is our input placeholder
        input = Input(shape=(self.dim,), name='encoder_input')
        latent_input = Input(shape=(self.latent_dim,), name='decoder_input')
        
        encoded_out = self._set_layers(input, 'encode')
        decoded_out = self._set_layers(latent_input, 'decode')
        
        # encoder
        self.encoder = Model(input, encoded_out, name='encoder')
        # decoder
        self.decoder = Model(latent_input, decoded_out, name='decoder')
        
        # full auto-encoder model
        model_out = self.decoder(self.reparametrize(self.encoder(input)))
        if self.repatr_out: model_out = self.reparametrize(model_out, dim=self.dim)
        self.model = Model(inputs=input, outputs=model_out, name='autoencoder')
        
        # compile model
        optimizer = kwargs.pop('optimizer','adam')
        beta = kwargs.pop('beta',1)
        custom_loss = kwargs.pop('custom_loss',None)
        metrics = kwargs.pop('metrics',['mae'])
        self.model.compile(optimizer=optimizer, loss=self._loss(beta,custom_loss), metrics=metrics, **kwargs)
    
    def train(self, x_train, x_test=None, epochs=100, batch_size=32, verbose=0, **kwargs):
        """
        Train the model with data
        """
        num_samp=x_train.shape[0]
        if x_test is None:
            tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
            te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
            x_test = x_train[te_idx]
            x_train = x_train[tr_idx]
        patience = kwargs.pop('patience',0)
        es = EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=patience)
        self.history = self.model.fit(x_train, x_train,
                                      validation_data=(x_test, x_test),
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      callbacks=[es],
                                      verbose=verbose,**kwargs)
    
    def save(self, savepath='./'):
        """
        Save the trained model for future use
        """
        import os
        self.model.save(os.path.join(savepath,'ae_fullmodel.h5'))
        self.encoder.save(os.path.join(savepath,'ae_encoder.h5'))
        self.decoder.save(os.path.join(savepath,'ae_decoder.h5'))
    
    def encode(self, input):
        """
        Output encoded state
        """
        assert input.shape[1]==self.dim, 'Wrong input dimension for encoder!'
        output = self.reparametrize(self.encoder.predict(input)).numpy()
        return output
    
    def decode(self, input):
        """
        Output decoded state
        """
        assert input.shape[1]==self.latent_dim, 'Wrong input dimension for decoder!'
        output = self.decoder.predict(input)
        if self.repatr_out: output = self.reparametrize(output, self.dim).numpy()
        return output
    
    def jacobian(self, input, coding='encode'):
        """
        Obtain Jacobian matrix of encoder (coding encode) or decoder (coding decode)
        """
        model = getattr(self,coding+'r')
#         x = tf.constant(input, dtype=tf.float32)
        x = tf.Variable(input, trainable=True, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            y = model(x)
        jac = g.jacobian(y,x).numpy()
#         jac = g.jacobian(y,x,experimental_use_pfor=False).numpy() # use this for some problematic activations e.g. LeakyReLU
        return np.squeeze(jac)
    
    def logvol(self, input, coding='encode'):
        """
        Obtain the log-volume defined by Gram matrix determinant
        """
        jac = self.jacobian(input, coding)
        d = np.abs(np.linalg.svd(jac,compute_uv=False))
        return np.sum(np.log(d[d>0]))

if __name__ == '__main__':
    # set random seed
    np.random.seed(2020)
    
    # load data
    loaded=np.load(file='./vae_training.npz')
    X=loaded['X']
    num_samp=X.shape[0]
    
    tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
    te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
    x_train,x_test=X[tr_idx],X[te_idx]
    
    # define Auto-Encoder
    half_depth=3; latent_dim=441
    vae=VAE(num_samp, half_depth, latent_dim)
    epochs=200
    import timeit
    t_start=timeit.default_timer()
    vae.train(x_train,x_test,epochs,batch_size=64,verbose=1)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training variational AutoEncoder: {}'.format(t_used))
    
    # save Auto-Encoder
    vae.model.save('./result/vae_fullmodel.h5')
    vae.encoder.save('./result/vae_encoder.h5')
    vae.decoder.save('./result/vae_decoder.h5') # cannot save, but can be reconstructed by: 
    # how to laod model
#     from tensorflow.keras.models import load_model
#     reconstructed_model=load_model('XX_model.h5')
    