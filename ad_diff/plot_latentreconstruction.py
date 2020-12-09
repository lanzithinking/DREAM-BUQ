"""
This is to plot latent and reconstructed samples.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import sys,os
sys.path.append( "../" )
from advdiff import advdiff
from util.dolfin_gadget import *
from nn.ae import AutoEncoder
from tensorflow.keras.models import load_model

# set random seed
seed=2020
np.random.seed(seed)
tf.random.set_seed(seed)

# define the inverse problem
meshsz = (61,61)
eldeg = 1
gamma = 2.; delta = 10.
rel_noise = .5
nref = 1
adif = advdiff(mesh=meshsz, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed)
adif.prior.V=adif.prior.Vh
adif.misfit.obs=np.array([dat.get_local() for dat in adif.misfit.d.data]).flatten()
# define the latent (coarser) inverse problem
meshsz_latent = (21,21)
adif_latent = advdiff(mesh=meshsz_latent, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed)
adif_latent.prior.V=adif_latent.prior.Vh
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1

# define the autoencoder (AE)
# load data
ensbl_sz = 500
folder = './train_NN_eldeg'+str(eldeg)
savepath = './analysis_eldeg'+str(eldeg)
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
X=loaded['X']
# pre-processing: scale X to 0-1
# X-=np.nanmin(X,axis=(1,2),keepdims=True) # try axis=(1,2,3)
# X/=np.nanmax(X,axis=(1,2),keepdims=True)
# split train/test
num_samp=X.shape[0]
# n_tr=np.int(num_samp*.75)
# x_train=X[:n_tr]
# x_test=X[n_tr:]
tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
x_train,x_test=X[tr_idx],X[te_idx]

# define AE
half_depth=3; latent_dim=adif_latent.prior.V.dim()
activation='elu'
# activation=tf.keras.layers.LeakyReLU(alpha=1.5)
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
ae=AutoEncoder(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim,
               activation=activation, optimizer=optimizer)
f_name=['ae_'+i+'_'+algs[alg_no]+str(ensbl_sz) for i in ('fullmodel','encoder','decoder')]
# savepath=savepath+'/saved_model'
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
    patience=0
    noise=0.
    ae.train(x_train,x_test=x_test,epochs=epochs,batch_size=64,verbose=1,patience=patience,noise=noise)
    # save AE
    ae.model.save(os.path.join(folder,f_name[0]+'.h5'))
    ae.encoder.save(os.path.join(folder,f_name[1]+'.h5'))
    ae.decoder.save(os.path.join(folder,f_name[2]+'.h5'))

# read data and construct plot functions
u_f = df.Function(adif.prior.V)
u_f_lat = df.Function(adif_latent.prior.V)
# read MAP
try:
    f=df.XDMFFile(adif.mpi_comm, os.path.join(os.getcwd(),'properties/MAP.xdmf'))
    f.read_checkpoint(u_f,'m',0)
    f.close()
except:
    pass
u=u_f.vector()
# encode
u_encoded=ae.encode(vinP1(u,adif.prior.V).get_local()[None,:] if eldeg>1 else u.get_local()[None,:])
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
u_f.vector().set_local(u_decoded.flatten() if eldeg==1 else vinPn(u_decoded.flatten(), adif.prior.V))
sub_figs[2]=df.plot(u_f)
plt.title('Reconstructed')

# add common colorbar
# cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
ax=axes.flat[-1]
cax = fig.add_axes([ax.get_position().x1+0.02,ax.get_position().y0,0.02,axes.flat[0].get_position().y1-ax.get_position().y0])
cbar=plt.colorbar(sub_figs[2], cax=cax)
# from util.common_colorbar import common_colorbar
# fig=common_colorbar(fig,axes,sub_figs)

# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(savepath,'latent_reconstructed.png'),bbox_inches='tight')
# plt.show()