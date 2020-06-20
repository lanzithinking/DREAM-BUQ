"""
This is to plot latent and reconstructed samples.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import sys,os
sys.path.append( "../" )
from Elliptic import Elliptic
# from util.dolfin_gadget import vec2fun,fun2img,img2fun
from nn.ae import AutoEncoder
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
folder = './train_NN'
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
# # nll = lambda x: [-elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x_i.numpy().flatten()))[0] for x_i in x]
# # nll = lambda x: tf.map_fn(lambda x_i:-elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x_i.numpy().flatten()))[0], x)
# nll = lambda x,y: [(elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x[i].numpy().flatten()))[0]
#                     -elliptic.get_geom(elliptic_latent.prior.gen_vector(y[i].numpy().flatten()))[0])**2 for i in range(x.shape[0])]
ae=AutoEncoder(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim,
               activation=activation, optimizer=optimizer)
# ae=AutoEncoder(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim,
#                activation=activation, optimizer=optimizer, loss=nll, run_eagerly=True)
f_name=['ae_'+i+'_'+algs[alg_no]+str(ensbl_sz) for i in ('fullmodel','encoder','decoder')]
# folder=folder+'/saved_model'
try:
    ae.model=load_model(os.path.join(folder,f_name[0]+'.h5'),custom_objects={'loss':None})
    print(f_name[0]+' has been loaded!')
    ae.encoder=load_model(os.path.join(folder,f_name[1]+'.h5'),custom_objects={'loss':None})
    print(f_name[1]+' has been loaded!')
    ae.decoder=load_model(os.path.join(folder,f_name[2]+'.h5'),custom_objects={'loss':None})
    print(f_name[2]+' has been loaded!')
except Exception as err:
    print(err)
    print('Train AutoEncoder...\n')
    epochs=200
    import timeit
    t_start=timeit.default_timer()
    ae.train(x_train,x_test=x_test,epochs=epochs,batch_size=64,verbose=1)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training AE: {}'.format(t_used))
    # save AE
    ae.model.save(os.path.join(folder,f_name[0]+'.h5'))
    ae.encoder.save(os.path.join(folder,f_name[1]+'.h5'))
    ae.decoder.save(os.path.join(folder,f_name[2]+'.h5'))

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
u_encoded=ae.encode(u.get_local()[None,:])
# decode
u_decoded=ae.decode(u_encoded)

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
plt.savefig(os.path.join(folder,'latent_reconstructed.png'),bbox_inches='tight')
# plt.show()