#!/usr/bin/env python
"""
Convolutional AutoEncoder
Shiwei Lan @ASU, 2020
-------------------------------------------
Convolutional AutoEncoder in TensorFlow 2.2
--------------------
Created June 14, 2020
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2020"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,Conv2DTranspose,MaxPooling2D,UpSampling2D,Flatten,Reshape,Dense
from tensorflow.keras.models import Model
# from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping


class ConvAutoEncoder:
    def __init__(self, im_shape, num_filters=[16,32], kernel_size=(3,3), pool_size=(2,2), strides=(1,1), latent_dim=None, **kwargs):
        """
        Convolutional AutoEncoder with encoder that maps inputs to latent variables and decoder that reconstructs data from latent variables.
        Heuristic structure: inputs --(encoder)-- latent variables --(decoder)-- reconstructions.
        -----------------------------------------------------------------------------------------
        im_shape: image shape (im_sz, chnl) both input and output
        num_filters: list of filter sizes of Conv2D
        half_depth: the depth of the network of encoder and decoder respectively
        latent_dim: the dimension of the latent space
        kernel_size: kernel size of Conv2D
        pool_size: pool size of MaxPooling2D
        strides: strides of Conv2D/MaxPooling2D
        padding: padding of Conv2D/MaxPooling2D
        activations: specification of activation functions, can be a list of strings or Keras activation layers
        kernel_initializer: kernel_initializer
        """
        self.im_shape = im_shape
        assert all([i %2==0 for i in self.im_shape[:2]]), 'Must have even image size!'
        self.num_filters = num_filters
        self.half_depth = len(self.num_filters)
        self.latent_dim = latent_dim
        if self.latent_dim is None: self.latent_dim = np.ceil(np.prod(self.im_shape)/self.half_depth).astype('int')
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.strides = strides
        self.padding = kwargs.pop('padding','same')
        assert self.padding=='same', 'Padding has to set as "same"!'
        self.activations = kwargs.pop('activations',{'conv':'relu','latent':'linear'})
        self.kernel_initializer=kwargs.pop('kernel_initializer','glorot_uniform')
        # build neural network
        self.build(**kwargs)
    
    def _set_encode_layers(self, input):
        """
        Set network layers of encoder
        """
        filters = self.num_filters
        output = input
        for i in range(self.half_depth):
            output=Conv2D(filters=filters[i], kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                          activation=self.activations['conv'], kernel_initializer=self.kernel_initializer, name='encode_conv_layer{}'.format(i))(output)
            output=MaxPooling2D(pool_size=self.pool_size, name='encode_pool_layer{}'.format(i))(output)
        if self.activations['latent'] is not None:
            output=Flatten()(output)
            if callable(self.activations['latent']):
                output = Dense(units=self.latent_dim, kernel_initializer=self.kernel_initializer, name='encode_out')(output)
                output = self.activations['latent'](output)
            else:
                output = Dense(units=self.latent_dim, activation=self.activations['latent'], kernel_initializer=self.kernel_initializer, name='encode_out')(output)
        else:
            self.latent_dim=output.shape[1:]
        return output
    
    def _set_decode_layers(self, input):
        """
        Set network layers of decoder
        """
        filters = self.num_filters[-2::-1]
        output = input
        if self.activations['latent'] is not None:
            pre_flatten_shape = self.encoder.get_layer(name='encode_pool_layer{}'.format(self.half_depth-1)).output.shape[1:]
            pre_flatten_dim = np.prod(pre_flatten_shape)
            if callable(self.activations['latent']):
                output = self.activations['latent'](output)
                output = Dense(units=pre_flatten_dim, kernel_initializer=self.kernel_initializer, name='decode_in')(output)
            else:
                output = Dense(units=pre_flatten_dim, activation=self.activations['latent'], kernel_initializer=self.kernel_initializer, name='decode_in')(output)
            output = Reshape(pre_flatten_shape)(output)
        for i in range(self.half_depth):
            output=UpSampling2D(size=self.pool_size, name='decode_unpool_layer{}'.format(i))(output)
            if i<self.half_depth-1:
                output=Conv2DTranspose(filters=filters[i], kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                       activation=self.activations['conv'], kernel_initializer=self.kernel_initializer, name='decode_deconv_layer{}'.format(i))(output)
            else:
                output=Conv2DTranspose(filters=1, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                       activation=self.activations['conv'], kernel_initializer=self.kernel_initializer, name='decode_out')(output)
        return output
    
    def _custom_loss(self,loss_f):
        """
        Wrapper to customize loss function (on latent space)
        """
        def loss(y_true, y_pred):
            L=.5*tf.math.reduce_sum((y_true-y_pred)**2,axis=[1,2,3]) # mse: prior
            L+=loss_f(self.encoder(y_true),y_pred) # potential on latent space: likelihood
#             L=loss_f(self.encoder(y_true),y_true-y_pred)
            return L
        return loss
        
    def build(self,**kwargs):
        """
        Set up the network structure and compile the model with optimizer, loss and metrics.
        """
        # encoder
        input = Input(shape=self.im_shape, name='encoder_input')
        encoded_out = self._set_encode_layers(input)
        self.encoder = Model(input, encoded_out, name='encoder')
        
        # decoder
        latent_input = Input(shape=self.encoder.output.shape[1:], name='decoder_input')
        decoded_out = self._set_decode_layers(latent_input)
        self.decoder = Model(latent_input, decoded_out, name='decoder')

        # full auto-encoder model
        self.model = Model(inputs=input, outputs=self.decoder(self.encoder(input)), name='autoencoder')
        
        # compile model
        optimizer = kwargs.pop('optimizer','adam')
        loss = kwargs.pop('loss','mse')
        metrics = kwargs.pop('metrics',['mae'])
        self.model.compile(optimizer=optimizer, loss=self._custom_loss(loss) if callable(loss) else loss, metrics=metrics, **kwargs)
    
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
        self.model.save(os.path.join(savepath,'cae_fullmodel.h5'))
        self.encoder.save(os.path.join(savepath,'cae_encoder.h5'))
        self.decoder.save(os.path.join(savepath,'cae_decoder.h5'))
    
    def encode(self, input):
        """
        Output encoded state
        """
        assert input.shape[1:]==self.im_shape, 'Wrong input dimension for encoder!'
        return self.encoder.predict(input)
    
    def decode(self, input):
        """
        Output decoded state
        """
        assert input.shape[1:]==self.latent_dim if self.activations['latent'] is None else input.shape[1]==self.latent_dim, 'Wrong input dimension for decoder!'
        return self.decoder.predict(input)
    
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
#         jac = g.jacobian(y,x).numpy()
        jac = g.jacobian(y,x,experimental_use_pfor=(coding=='encode')).numpy() # use this for some problematic activations e.g. LeakyReLU
        return np.squeeze(jac)
    
    def logvol(self, input, coding='encode'):
        """
        Obtain the log-volume defined by Gram matrix determinant
        """
        jac = self.jacobian(input, coding)
        jac = jac.reshape({'encode':(self.latent_dim,-1),'decode':(-1,self.latent_dim)}[coding])
        d = np.abs(np.linalg.svd(jac,compute_uv=False))
        return np.sum(np.log(d[d>0]))

if __name__ == '__main__':
    # set random seed
    np.random.seed(2020)
    
    # load data
    loaded=np.load(file='./cae_training.npz')
    X=loaded['X']
    num_samp=X.shape[0]
    
    tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
    te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
    x_train,x_test=X[tr_idx],X[te_idx]
    
    # define Convolutional Auto-Encoder
    num_filters=[16,32]
    activations={'conv':'relu','latent':tf.keras.layers.PReLU()}
    latent_dim=441
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    cae=ConvAutoEncoder(x_train.shape[1:], num_filters=num_filters,
                        latent_dim=latent_dim, activations=activations, optimizer=optimizer)
    epochs=200
    import timeit
    t_start=timeit.default_timer()
    cae.train(x_train,x_test,epochs,batch_size=64,verbose=1)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training Convolutional AutoEncoder: {}'.format(t_used))
    
    # save Convolutional Auto-Encoder
    cae.model.save('./result/cae_fullmodel.h5')
    cae.encoder.save('./result/cae_encoder.h5')
    cae.decoder.save('./result/cae_decoder.h5') # cannot save, but can be reconstructed by: 
    # how to laod model
#     from tensorflow.keras.models import load_model
#     reconstructed_model=load_model('XX_model.h5')
    