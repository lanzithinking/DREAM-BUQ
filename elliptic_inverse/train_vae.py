#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:47:22 2020

@author: apple
"""

"""
This is to train Convolutional AutoEncoder for dimension reduction.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import sys,os
sys.path.append( "../" )
from Elliptic import Elliptic
from util.dolfin_gadget import vec2fun,fun2img,img2fun
from nn.vae import VariationalAE
from tensorflow.keras.models import load_model

# set random seed
np.random.seed(2020)
tf.random.set_seed(2020)

# define the inverse problem
nx=40; ny=40
SNR=50
elliptic = Elliptic(nx=nx,ny=ny,SNR=SNR)
# define the latent (coarser) inverse problem
nx=10; ny=10
obs,nzsd,loc=[getattr(elliptic.misfit,i) for i in ('obs','nzsd','loc')]
elliptic_latent = Elliptic(nx=nx,ny=ny,SNR=SNR,obs=obs,nzsd=nzsd,loc=loc)
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1

# define the Variational autoencoder (AE)
# load data
ensbl_sz = 500
folder = './train_DNN'
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_AE.npz'))
X=loaded['X']
# pre-processing: scale X to 0-1
# X-=np.nanmin(X,axis=(1,2),keepdims=True) # try axis=(1,2,3)
# X/=np.nanmax(X,axis=(1,2),keepdims=True)
# X_min=np.nanmin(X)
# X-=X_min
# X_max=np.nanmax(X)
# X/=X_max

# split train/test
num_samp=X.shape[0]
n_tr=np.int(num_samp*.75)
x_train=X[:n_tr]
x_test=X[n_tr:]

# define AE
half_depth=3; latent_dim=elliptic_latent.pde.V.dim()
activation={'hidden':'relu','latent':'linear','decode':'sigmoid'}
# activation=tf.keras.layers.LeakyReLU(alpha=0.01)
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
vae=VariationalAE(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim,hidden_dim=500,
               activation=activation, optimizer=optimizer)

# activations={'conv':'linear','latent':tf.keras.layers.PReLU()}
# activations={'conv':'relu','latent':'linear'}
# activations={'conv':tf.keras.layers.LeakyReLU(alpha=0.1),'latent':'linear'}

#latent_dim=elliptic_latent.prior.dim

# folder=folder+'/saved_model'
f_name=['vae_'+i+'_'+algs[alg_no]+str(ensbl_sz)+'.h5' for i in ('fullmodel','encoder','decoder')]
try:
    vae.model=load_model(os.path.join(folder,f_name[0]),custom_objects={'loss':None, 'LeakyReLU': tf.keras.layers.LeakyReLU(alpha=0.1)})
    print(f_name[0]+' has been loaded!')
    vae.encoder=load_model(os.path.join(folder,f_name[1]),custom_objects={'loss':None, 'LeakyReLU': tf.keras.layers.LeakyReLU(alpha=0.1)})
    print(f_name[1]+' has been loaded!')
    vae.decoder=load_model(os.path.join(folder,f_name[2]),custom_objects={'loss':None, 'LeakyReLU': tf.keras.layers.LeakyReLU(alpha=0.1)})
    print(f_name[2]+' has been loaded!')
except Exception as err:
    print(err)
    print('Train Variational AutoEncoder...\n')
    epochs=100
    patience=0
    import timeit
    t_start=timeit.default_timer()
    vae.train(x_train,x_test=x_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training CAE: {}'.format(t_used))
    # save AE
    vae.model.save(os.path.join(folder,f_name[0]))
    vae.encoder.save(os.path.join(folder,f_name[1]))
    vae.decoder.save(os.path.join(folder,f_name[2]))

# # plot
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(15,5), facecolor='white')
# plt.ion()
# # plt.show(block=True)
# u_f = df.Function(elliptic.pde.V)
# u_f_lat = df.Function(elliptic_latent.pde.V)
# for n in range(20):
#     u=elliptic.prior.sample()
#     # encode
#     u_f.vector().set_local(u)
#     u_encoded=cae.encode(fun2img(u_f)[None,:-1,:-1,None])
#     # decode
#     u_decoded=cae.decode(u_encoded)
# #     u_decoded=cae.model.predict(fun2img(u_f)[None,:-1,:-1,None])
#     
#     # compute the log-volumes
# #     logvol_enc=cae.logvol(fun2img(u_f)[None,:-1,:-1,None],'encode')
# #     print('Log-volume of encoder: {}'.format(logvol_enc))
# #     logvol_dec=cae.logvol(u_encoded,'decode')
# #     print('Log-volume of decoder: {}'.format(logvol_dec))
#     
#     # plot
#     plt.subplot(131)
#     u_f.vector().set_local(u)
#     df.plot(u_f)
#     plt.title('Original Sample')
#     plt.subplot(132)
#     u_f_lat.vector().set_local(u_encoded.flatten())
#     df.plot(u_f_lat)
#     plt.title('Latent Sample')
#     plt.subplot(133)
#     u_decoded=np.squeeze(u_decoded)
#     u_decoded*=X_max; u_decoded+=X_min
#     u_decoded1=np.zeros([i+1 for i in u_decoded.shape])
#     u_decoded1[:-1,:-1]=u_decoded
#     u_f=img2fun(u_decoded1,elliptic.pde.V)
#     df.plot(u_f)
#     plt.title('Reconstructed Sample')
#     plt.draw()
#     plt.pause(1.0/10.0)

# read data and construct plot functions
u_f = df.Function(elliptic.pde.V)
u_f_lat = df.Function(elliptic_latent.pde.V)
# read MAP
try:
    f=df.HDF5File(elliptic.pde.mpi_comm, os.path.join('./result',"MAP_SNR"+str(SNR)+".h5"), "r")
    f.read(u_f,"parameter")
    f.close()
except:
    pass
u=u_f.vector()
# encode
u_encoded=vae.encoder.predict(u.get_local()[None,])[:,:vae.latent_dim]
# decode
u_decoded=vae.model.predict(u.get_local()[None,:])

# plot
import matplotlib.pyplot as plt
import matplotlib as mp
plt.rcParams['image.cmap'] = 'jet'
fig,axes = plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True,figsize=(15,5))
sub_figs=[None]*3
# plot
plt.axes(axes.flat[0])
# u_f.vector().set_local(u)
sub_figs[0]=df.plot(u_f)
plt.title('Original')
plt.axes(axes.flat[1])
u_f_lat.vector().set_local(u_encoded.flatten())
sub_figs[1]=df.plot(u_f_lat)
plt.title('Latent')
plt.axes(axes.flat[2])
u_f.vector().set_local(u_decoded.flatten())
sub_figs[2]=df.plot(u_f)
plt.title('Reconstructed')

# add common colorbar
from util.common_colorbar import common_colorbar
fig=common_colorbar(fig,axes,sub_figs)
# 
# # adjust common climit
# clims=np.array([sub_figs[i].get_clim() for i in range(3)])
# common_clim=np.min(clims[:,0]),np.max(clims[:,1])
# print(common_clim)
# [sub_fig.set_clim(common_clim) for sub_fig in sub_figs]
# # set color bar
# cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
# cbar=plt.colorbar(sub_figs[-1], cax=cax,cmap='jet', **kw)

# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(folder,algs[alg_no]+str(ensbl_sz)+'latent_reconsvae.png'),bbox_inches='tight')
# plt.show()