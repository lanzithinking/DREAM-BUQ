import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model
# from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping


class AutoEncoder:
    def __init__(self, x_train, x_test=None, half_depth=3, latent_dim=None, **kwargs):
        self.x_train = x_train
        self.num_samp, self.dim = self.x_train.shape
        self.x_test = x_test
        if self.x_test is None:
            tr_idx=np.random.choice(self.num_samp,size=np.floor(.75*self.num_samp).astype('int'),replace=False)
            te_idx=np.setdiff1d(np.arange(self.num_samp),tr_idx)
            self.x_test = self.x_train[te_idx]
            self.x_train = self.x_train[tr_idx]
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
        node_sizes = {'encode':self.node_sizes[:self.half_depth+1],'decode':self.node_sizes[self.half_depth:]}[coding]
        output = input
        for i in range(self.half_depth):
            layer_name = "{}_out".format(coding) if i==self.half_depth-1 else "{}_layer{}".format(coding,i)
            output = Dense(units=node_sizes[i+1], activation=self.activation, name=layer_name)(output)
        return output

    def build(self,**kwargs):
        # this is our input placeholder
        input = Input(shape=(self.dim,), name='encoder_input')
        
        encoded_out = self._set_layers(input, 'encode')
        decoded_out = self._set_layers(encoded_out, 'decode')

        # this model maps an input to its reconstruction
        self.model = Model(input, decoded_out)

        #Get intermediate layer
        self.latent_model = Model(inputs=self.model.input,
                                  outputs=self.model.get_layer(name="encode_out").output)
        self.encoder = self.latent_model
        self.decoder = K.function(encoded_out, decoded_out)
        
        # compile model
        optimizer = kwargs.pop('optimizer','adam')
        loss = kwargs.pop('loss','mse')
        metrics = kwargs.pop('metrics',['mae'])
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, epochs, batch_size=32, verbose=0, dump_data=False):
        es = EarlyStopping(monitor='loss', mode='auto', verbose=1)
        self.history = self.model.fit(self.x_train,
                                      self.x_train,
                                      validation_data=(self.x_test, self.x_test),
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      callbacks=[es],
                                      verbose=verbose)
        if dump_data:
            self.x_train=[]
            self.x_test=[]
    
    def save(self, savepath='./'):
        import os
        self.model.save(os.path.join(savepath,'ae_fullmodel.h5'))
        self.encoder.save(os.path.join(savepath,'ae_encoder.h5'))
    
    def encode(self, input):
        assert input.shape[1]==self.dim, 'Wrong input dimension for encoder!'
        return self.encoder.predict(input)
    
    def decode(self, input):
        assert input.shape[1]==self.latent_dim, 'Wrong input dimension for decoder!'
        return self.decoder(input)
    
    def jacobian(self, input, coding='encode'):
        model = getattr(self,coding+'r')
        with tf.GradientTape() as g:
            x = tf.constant(input)
            g.watch(x)
            y = model(x)
        jac = g.jacobian(y,x).numpy()
        return np.squeeze(jac)
    
    def logvol(self, input, coding='encode'):
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
    ae=AutoEncoder(x_train, x_test, half_depth, latent_dim)
    epochs=200
    import timeit
    t_start=timeit.default_timer()
    ae.train(epochs,batch_size=64,verbose=1)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training AutoEncoder: {}'.format(t_used))
    
    # save Auto-Encoder
    ae.model.save('./result/ae_fullmodel.h5')
    ae.encoder.save('./result/ae_encoder.h5')
#     ae.decoder.save('./result/ae_decoder.h5') # cannot save, but can be reconstructed by: 
#     decoder=K.function(inputs=ae_fullmodel.get_layer(name="encode_out").output,outputs=ae_fullmodel.output)
    # how to laod model
#     from tensorflow.keras.models import load_model
#     reconstructed_model=load_model('XX_model.h5')
    