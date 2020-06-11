#!/usr/bin/env python
"""
AutoEncoder
Shiwei Lan @ASU, 2020
--------------------------------------
Standard AutoEncoder in TensorFlow 2.2
--------------------
Created June 4, 2020
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2020"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model
# from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping


class AutoEncoder:
    def __init__(self, dim, half_depth=3, latent_dim=None, **kwargs):
        """
        AutoEncoder with encoder that maps inputs to latent variables and decoder that reconstructs data from latent variables.
        Heuristic structure: inputs --(encoder)-- latent variables --(decoder)-- reconstructions.
        -----------------------------------------------------------------------------------------
        dim: dimension of the original (input and output) space
        half_depth: the depth of the network of encoder and decoder if a symmetric structure is imposed (by default)
        latent_dim: the dimension of the latent space
        activation: specification of activation functions, can be a string or a Keras activation layer
        node_sizes: sizes of the nodes of the network, which can overwrite half_depth and induce an asymmetric structure.
        """
        self.dim = dim
        self.half_depth = half_depth
        self.latent_dim = latent_dim
        if self.latent_dim is None: self.latent_dim = np.ceil(self.dim/self.half_depth).astype('int')
        self.activation = kwargs.pop('activation','linear')
        self.node_sizes = kwargs.pop('node_sizes',None)
        if self.node_sizes is None or np.size(self.node_sizes)!=2*self.half_depth+1:
            self.node_sizes = np.linspace(self.dim,self.latent_dim,self.half_depth+1,dtype=np.int)
            self.node_sizes = np.concatenate((self.node_sizes,self.node_sizes[-2::-1]))
        if not np.all([self.node_sizes[i]==self.dim for i in (0,-1)]):
            raise ValueError('End node sizes not matching input/output dimensions!')
        # build neural network
        self.build(**kwargs)
        
    def _set_layers(self, input, coding='encode'):
        """
        Set network layers of encoder (coding 'encode') or decoder (coding 'decode') based on given node_sizes
        """
        node_sizes = {'encode':self.node_sizes[:self.half_depth+1],'decode':self.node_sizes[self.half_depth:]}[coding]
        output = input
        for i in range(self.half_depth):
            layer_name = "{}_out".format(coding) if i==self.half_depth-1 else "{}_layer{}".format(coding,i)
            if callable(self.activation):
                output = Dense(units=node_sizes[i+1], name=layer_name)(output)
                output = self.activation(output)
            else:
                output = Dense(units=node_sizes[i+1], activation=self.activation, name=layer_name)(output)
        return output

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
        self.model = Model(inputs=input, outputs=self.decoder(self.encoder(input)), name='autoencoder')
        
        # compile model
        optimizer = kwargs.pop('optimizer','adam')
        loss = kwargs.pop('loss','mse')
        metrics = kwargs.pop('metrics',['mae'])
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, x_train, x_test=None, epochs=100, batch_size=32, verbose=0):
        """
        Train the model with data
        """
        num_samp=x_train.shape[0]
        if x_test is None:
            tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
            te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
            x_test = x_train[te_idx]
            x_train = x_train[tr_idx]
        es = EarlyStopping(monitor='loss', mode='auto', verbose=1)
        self.history = self.model.fit(x_train, x_train,
                                      validation_data=(x_test, x_test),
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      callbacks=[es],
                                      verbose=verbose)
    
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
        return self.encoder.predict(input)
    
    def decode(self, input):
        """
        Output decoded state
        """
        assert input.shape[1]==self.latent_dim, 'Wrong input dimension for decoder!'
        return self.decoder.predict(input)
    
    def jacobian(self, input, coding='encode'):
        """
        Obtain Jacobian matrix of encoder (coding encode) or decoder (coding decode)
        """
        model = getattr(self,coding+'r')
        x = tf.Variable(input, trainable=True, dtype=tf.float32)
#         x = tf.constant(input, dtype=tf.float32)
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
    loaded=np.load(file='./ae_training.npz')
    X=loaded['X']
    num_samp=X.shape[0]
    
    tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
    te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
    x_train,x_test=X[tr_idx],X[te_idx]
    
    # define Auto-Encoder
    half_depth=3; latent_dim=441
    ae=AutoEncoder(num_samp, half_depth, latent_dim)
    epochs=200
    import timeit
    t_start=timeit.default_timer()
    ae.train(ex_train,x_test,pochs,batch_size=64,verbose=1)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training AutoEncoder: {}'.format(t_used))
    
    # save Auto-Encoder
    ae.model.save('./result/ae_fullmodel.h5')
    ae.encoder.save('./result/ae_encoder.h5')
    ae.decoder.save('./result/ae_decoder.h5') # cannot save, but can be reconstructed by: 
    # how to laod model
#     from tensorflow.keras.models import load_model
#     reconstructed_model=load_model('XX_model.h5')
    