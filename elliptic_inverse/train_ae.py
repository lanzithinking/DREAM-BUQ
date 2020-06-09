"""
This is to train AutoEncoder for dimension reduction.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import sys,os
sys.path.append( "../" )
from Elliptic import Elliptic
# from util.dolfin_gadget import vec2fun,fun2img,img2fun
from nn.autoencoder import AutoEncoder
from tensorflow.keras.models import load_model

# set random seed
np.random.seed(2020)

# define the inverse problem
nx=40; ny=40
SNR=50
elliptic = Elliptic(nx=nx,ny=ny,SNR=SNR)
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1

# define the autoencoder (AE)
# load data
ensbl_sz = 500
folder = './train_DNN'
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_AE.npz'))
X=loaded['X']
# pre-processing: scale X to 0-1
# X-=np.nanmin(X,axis=(1,2),keepdims=True) # try axis=(1,2,3)
# X/=np.nanmax(X,axis=(1,2),keepdims=True)
# split train/test
num_samp=X.shape[0]
n_tr=np.int(num_samp*.75)
x_train=X[:n_tr]
x_test=X[n_tr:]

# define AE
half_depth=3; latent_dim=441
activation='linear'
# activation=tf.keras.layers.LeakyReLU(alpha=0.01)
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
ae=AutoEncoder(x_train, x_test=None, half_depth=half_depth, latent_dim=latent_dim,
               activation=activation, optimizer=optimizer)
try:
    ae.model=load_model(os.path.join(folder,'ae_fullmodel_'+algs[alg_no]+str(ensbl_sz)+'.h5'))
    print('ae_fullmodel'+algs[alg_no]+str(ensbl_sz)+'.h5'+' has been loaded!')
    ae.encoder=load_model(os.path.join(folder,'ae_encoder_'+algs[alg_no]+str(ensbl_sz)+'.h5'))
    print('ae_encoder'+algs[alg_no]+str(ensbl_sz)+'.h5'+' has been loaded!')
    from tensorflow.keras import backend as K
    ae.decoder=K.function(inputs=ae.model.get_layer(name="encode_out").output,outputs=ae.model.output)
    print('ae_decoder'+algs[alg_no]+str(ensbl_sz)+' has been configured!')
except Exception as err:
    print(err)
    print('Train AutoEncoder...\n')
    epochs=200
    import timeit
    t_start=timeit.default_timer()
    ae.train(epochs,batch_size=64,verbose=1)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training AE: {}'.format(t_used))
    # save AE
    ae.model.save(os.path.join(folder,'ae_fullmodel_'+algs[alg_no]+str(ensbl_sz)+'.h5'))
    ae.encoder.save(os.path.join(folder,'ae_encoder_'+algs[alg_no]+str(ensbl_sz)+'.h5'))

# some more test
# loglik = lambda x: 0.5*elliptic.misfit.prec*tf.math.reduce_sum((cnn.model(x)-elliptic.misfit.obs)**2,axis=1)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,6), facecolor='white')
plt.ion()
# plt.show(block=True)
u_f = df.Function(elliptic.pde.V)
for n in range(20):
    u=elliptic.prior.sample()
    # encode
    u_encoded=ae.encode(u.get_local()[None,:])
    # decode
    u_decoded=ae.decode(u_encoded)
    
    # plot
    plt.subplot(121)
    u_f.vector().set_local(u)
    df.plot(u_f)
    plt.title('Original Sample')
    plt.subplot(122)
    u_f.vector().set_local(u_decoded.flatten())
    df.plot(u_f)
    plt.title('Reconstructed Sample')
    plt.draw()
    plt.pause(1.0/10.0)
    