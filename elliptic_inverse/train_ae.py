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
tf.random.set_seed(2020)

# define the inverse problem
nx=40; ny=40
SNR=50
elliptic = Elliptic(nx=nx,ny=ny,SNR=SNR)
# define the latent (coarser) inverse problem
nx=20; ny=20
obs,nzsd,loc=[getattr(elliptic.misfit,i) for i in ('obs','nzsd','loc')]
elliptic_latent = Elliptic(nx=nx,ny=ny,SNR=SNR,obs=obs,nzsd=nzsd,loc=loc)
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
half_depth=3; latent_dim=elliptic_latent.pde.V.dim()
activation='linear'
# activation=tf.keras.layers.LeakyReLU(alpha=0.01)
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
ae=AutoEncoder(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim,
               activation=activation, optimizer=optimizer)
# # nll = lambda x: [-elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x_i.numpy().flatten()))[0] for x_i in x]
# # nll = lambda x: tf.map_fn(lambda x_i:-elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x_i.numpy().flatten()))[0], x)
# nll = lambda x,y: [(elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x[i].numpy().flatten()))[0]
#                     -elliptic.get_geom(elliptic.prior.gen_vector(y[i].numpy().flatten()))[0])**2 for i in range(x.shape[0])]
# ae=AutoEncoder(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim,
#                activation=activation, optimizer=optimizer, loss=nll, run_eagerly=True)
# folder=folder+'/saved_model'
f_name=['ae_'+i+'_'+algs[alg_no]+str(ensbl_sz)+'_customloss.h5' for i in ('fullmodel','encoder','decoder')]
try:
    ae.model=load_model(os.path.join(folder,f_name[0]),custom_objects={'loss':None})
    print(f_name[0]+' has been loaded!')
    ae.encoder=load_model(os.path.join(folder,f_name[1]),custom_objects={'loss':None})
    print(f_name[1]+' has been loaded!')
    ae.decoder=load_model(os.path.join(folder,f_name[2]),custom_objects={'loss':None})
    print(f_name[2]+' has been loaded!')
except Exception as err:
    print(err)
    print('Train AutoEncoder...\n')
    epochs=200
    patience=0
    import timeit
    t_start=timeit.default_timer()
    ae.train(x_train,x_test=x_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training AE: {}'.format(t_used))
    # save AE
    ae.model.save(os.path.join(folder,f_name[0]))
    ae.encoder.save(os.path.join(folder,f_name[1]))
    ae.decoder.save(os.path.join(folder,f_name[2]))

# plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,5), facecolor='white')
plt.ion()
# plt.show(block=True)
u_f = df.Function(elliptic.pde.V)
u_f_lat = df.Function(elliptic_latent.pde.V)
for n in range(20):
    u=elliptic.prior.sample()
    # encode
    u_encoded=ae.encode(u.get_local()[None,:])
    # decode
    u_decoded=ae.decode(u_encoded)
    
    # compute the log-volumes
    logvol_enc=ae.logvol(u.get_local()[None,:],'encode')
    print('Log-volume of encoder: {}'.format(logvol_enc))
    logvol_dec=ae.logvol(u_encoded,'decode')
    print('Log-volume of decoder: {}'.format(logvol_dec))
    
    # plot
    plt.subplot(131)
    u_f.vector().set_local(u)
    df.plot(u_f)
    plt.title('Original Sample')
    plt.subplot(132)
    u_f_lat.vector().set_local(u_encoded.flatten())
    df.plot(u_f_lat)
    plt.title('Latent Sample')
    plt.subplot(133)
    u_f.vector().set_local(u_decoded.flatten())
    df.plot(u_f)
    plt.title('Reconstructed Sample')
    plt.draw()
    plt.pause(1.0/10.0)
    