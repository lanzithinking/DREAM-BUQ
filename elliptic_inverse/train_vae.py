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
from nn.vae import VAE
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
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_X.npz'))
X=loaded['X']
# pre-processing: scale X to 0-1
# X-=np.nanmin(X,axis=1,keepdims=True)
# X/=np.nanmax(X,axis=1,keepdims=True)
# split train/test
num_samp=X.shape[0]
# n_tr=np.int(num_samp*.75)
# x_train=X[:n_tr]
# x_test=X[n_tr:]
tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
x_train,x_test=X[tr_idx],X[te_idx]

# define VAE
half_depth=3; latent_dim=elliptic_latent.pde.V.dim()
repatr_out=False
# activation='linear'
activation=tf.keras.layers.LeakyReLU(alpha=0.01)
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
vae=VAE(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim, repatr_out=repatr_out,
        activation=activation, optimizer=optimizer, beta=100)
# nll = lambda x: [-elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x_i.numpy().flatten()))[0] for x_i in x]
# # nll = lambda x: tf.map_fn(lambda x_i:-elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x_i.numpy().flatten()))[0], x)
# nll = lambda x,y: [(elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x[i].numpy().flatten()))[0]
#                     -elliptic.get_geom(elliptic.prior.gen_vector(y[i].numpy().flatten()))[0])**2 for i in range(x.shape[0])]
# vae=VAE(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim, repatr_out=repatr_out,
#         activation=activation, optimizer=optimizer, custom_loss=nll, run_eagerly=True)
# folder=folder+'/saved_model'
f_name=['vae_'+i+'_'+algs[alg_no]+str(ensbl_sz) for i in ('fullmodel','encoder','decoder')]
try:
    vae.model=load_model(os.path.join(folder,f_name[0]+'.h5'),custom_objects={'loss':None})
    print(f_name[0]+' has been loaded!')
    vae.encoder=load_model(os.path.join(folder,f_name[1]+'.h5'),custom_objects={'loss':None})
    print(f_name[1]+' has been loaded!')
    vae.decoder=load_model(os.path.join(folder,f_name[2]+'.h5'),custom_objects={'loss':None})
    print(f_name[2]+' has been loaded!')
except Exception as err:
    print(err)
    print('Train variational AutoEncoder...\n')
    epochs=200
    patience=0
    import timeit
    t_start=timeit.default_timer()
    vae.train(x_train,x_test=x_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training VAE: {}'.format(t_used))
    # save VAE
    vae.model.save(os.path.join(folder,f_name[0]+'.h5'))
    vae.encoder.save(os.path.join(folder,f_name[1]+'.h5'))
    vae.decoder.save(os.path.join(folder,f_name[2]+'.h5'))

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
    u_encoded=vae.encode(u.get_local()[None,:])
    # decode
    u_decoded=vae.decode(u_encoded)
    
#     # compute the log-volumes
#     logvol_enc=vae.logvol(u.get_local()[None,:],'encode')
#     print('Log-volume of encoder: {}'.format(logvol_enc))
#     logvol_dec=vae.logvol(u_encoded,'decode')
#     print('Log-volume of decoder: {}'.format(logvol_dec))
    
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
u_encoded=vae.encode(u.get_local()[None,:])
# decode
u_decoded=vae.decode(u_encoded)

# plot
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

# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(folder,'latent_reconstructed_vae.png'),bbox_inches='tight')
# plt.show()