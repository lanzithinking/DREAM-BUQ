#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 21:21:25 2020

@author: apple
"""

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
__version__ = "0.4"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com"

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
#import tensorflow.compat.v1 as tfv1 
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model
# from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
#print(tf.executing_eagerly())
#K.clear_session()
#tf.compat.v1.reset_default_graph()
#tfv1.disable_v2_behavior()
#tf.enable_eager_execution()
#tf.config.experimental_run_functions_eagerly(True)
#tf.compat.v1.disable_eager_execution()
class VariationalAE:
    def __init__(self, dim, half_depth=3, latent_dim=None, hidden_dim=None,**kwargs):
        """
        AutoEncoder with encoder that maps inputs to latent variables and decoder that reconstructs data from latent variables.
        Heuristic structure: inputs --(encoder)-- latent variables --(decoder)-- reconstructions.
        -----------------------------------------------------------------------------------------
        dim: dimension of the original (input and output) space
        half_depth: the depth of the network of encoder and decoder if a symmetric structure is imposed (by default)
        latent_dim: the dimension of the latent space
        hidden_dim: the dimension of the hidden space, from dim->latent_dim
        activation: specification of activation functions, can be a string or a Keras activation layer
        node_sizes: sizes of the nodes of the network, which can overwrite half_depth and induce an asymmetric structure.
        """
        self.dim = dim
        self.half_depth = half_depth
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        if self.latent_dim is None: self.latent_dim = np.ceil(self.dim/self.half_depth).astype('int')
        self.activation = kwargs.pop('activation',{'hidden':'relu','latent':'linear','decode':'sigmoid'})
        self.node_sizes = kwargs.pop('node_sizes',None)
        if self.node_sizes is None or np.size(self.node_sizes)!=2*self.half_depth+1:
            #self.node_sizes = np.linspace(self.dim,self.latent_dim,self.half_depth+1,dtype=np.int)
            self.node_sizes = np.concatenate((np.array([self.dim]),np.array([self.hidden_dim]*(half_depth-1)),
                                              np.array([self.latent_dim*2])))
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
                if i != self.half_depth-1:
                    output = self.activation['hidden'](output)  
                elif coding=='encode':
                    output = self.activation['latent'](output)
                else:
                    output = self.activation['decode'](output)
            else:
                if i != self.half_depth-1:
                    output = Dense(units=node_sizes[i+1], activation=self.activation['hidden'], name=layer_name)(output)
                elif coding=='encode':
                    output = Dense(units=node_sizes[i+1], activation=self.activation['latent'], name=layer_name)(output)
                else:
                    output = Dense(units=node_sizes[i+1], activation=self.activation['decode'], name=layer_name)(output)
              
        return output
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=[self.latent_dim])
        return eps * tf.exp(logvar * .5) + mean 
    
    def build(self,**kwargs):
        """
        Set up the network structure and compile the model with optimizer, loss and metrics.
        """
        # this is our input placeholder
        input = Input(shape=(self.dim,), name='encoder_input')
        latent_input = Input(shape=(self.latent_dim,), name='decoder_input')
        
        
        encoded_out = self._set_layers(input, 'encode')
        
        # encoderQ(z|x)
        self.encoder = Model(input, encoded_out, name='encoder')
        mean = self.encoder(input)[:,:self.latent_dim]
        logvar = self.encoder(input)[:,self.latent_dim:]
        z = self.reparameterize(mean, logvar)
        out = self._set_layers(z,'decode')
        
        # decoderP(x|z)
        decoded_out = self._set_layers(latent_input, 'decode')
        self.decoder = Model(latent_input, decoded_out, name='decoder')
        
        # full auto-encoder model
        self.model = Model(inputs=input, outputs=out, name='autoencoder')
        
            
        '''def loss(self,y_true, y_pred):
            # E[log P(X|z)]
            recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
            # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
            kl = 0.5 * K.sum(K.exp(logvar) + K.square(mean) - 1. - logvar, axis=1)
            return tf.reduce_mean(recon + kl)'''
        # compile model
        self.optimizer = kwargs.pop('optimizer','adam')
        #loss = kwargs.pop('loss','mse')
        #metrics = kwargs.pop('metrics',['mae'])
        #self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, experimental_run_tf_function=False,**kwargs)
    
    def lossf(self,y_true, y_pred):
        mean, logvar = self.encode(y_true)
        # E[log P(X|z)]
        recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl = 0.5 * K.sum(K.exp(logvar) + K.square(mean) - 1. - logvar, axis=1)
        return tf.reduce_mean(recon + kl)
        
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def train_step(self,x, **kwargs):
        """Executes one training step and returns the loss.
        This computes the loss and gradients, and uses the latter to update the model's parameters.
        """
        #optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
        
        with tf.GradientTape() as tape:
            x_pred = self.model(x)
            loss = self.lossf(x, x_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss.numpy()
  
    def train(self, x_train, x_test=None, epochs=100, batch_size=32, verbose=0, **kwargs):
        """
        Train the model with data
        """
        x_traind = (tf.data.Dataset.from_tensor_slices(tf.cast(x_train,dtype=tf.float32))
                 .shuffle(x_train.shape[0]).batch(batch_size))
        if x_test.all():
            x_testd = (tf.data.Dataset.from_tensor_slices(tf.cast(x_test,dtype=tf.float32)).
                       shuffle(x_test.shape[0]).batch(batch_size))
        for epoch in range(1, epochs + 1):
            loss = np.zeros(epochs)
            for x_train in x_traind:
                loss[epoch]+=self.train_step(x_train)
            print('loss:{}'.format(loss[epoch]))
            if epoch>10:
                if loss[epoch]>loss[epoch-1]:
                    break
        '''num_samp=x_train.shape[0]
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
                                      verbose=verbose,**kwargs)'''
    
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),axis=raxis)
    
    '''def _custom_loss(self,loss_f):
        
        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
        
        q_z = tfp.distributions.Normal(loc=q_mu, scale=q_sigma)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decoder(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
        def loss(y_true, y_pred):
            L=.5*tf.keras.losses.MSE(y_true, y_pred) # mse: prior
            L+=loss_f(self.encoder(y_true),y_pred) # potential on latent space: likelihood
            return L
        return loss'''
    
    
    def save(self, savepath='./'):
        """
        Save the trained model for future use
        """
        import os
        self.model.save(os.path.join(savepath,'ae_fullmodel.h5'))
        self.encoder.save(os.path.join(savepath,'ae_encoder.h5'))
        self.decoder.save(os.path.join(savepath,'ae_decoder.h5'))
    
   
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
    algs=['EKI','EKS']
    num_algs=len(algs)
    alg_no=0
    
    # define the autoencoder (AE)
    # load data
    import os
    ensbl_sz = 500
    folder = './train_DNN'
    loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_AE.npz'))
    X=loaded['X']
    num_samp=X.shape[0]
    
    tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
    te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
    x_train,x_test=X[tr_idx],X[te_idx]
    
    # define Auto-Encoder
    half_depth=3; latent_dim=441
    vae=VariationalAE(X.shape[1], half_depth, 100,500)
    epochs=200
    import timeit
    t_start=timeit.default_timer()
    vae.train(x_train,x_test,epochs,batch_size=64,verbose=1)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training AutoEncoder: {}'.format(t_used))
    f_name=['vae_'+i+'_'+algs[alg_no]+str(ensbl_sz)+'.h5' for i in ('fullmodel','encoder','decoder')]

    # save Auto-Encoder
    vae.model.save(os.path.join(folder,f_name[0]))
    vae.encoder.save(os.path.join(folder,f_name[1]))
    vae.decoder.save(os.path.join(folder,f_name[2])) # cannot save, but can be reconstructed by: 
    # how to laod model
#     from tensorflow.keras.models import load_model
#     reconstructed_model=load_model('XX_model.h5')
    
    