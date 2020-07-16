"""
This is to test VAE in reconstructing samples.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import sys,os
sys.path.append( '../' )
sys.path.append( '../../')
from Elliptic import Elliptic
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
nx=10; ny=10
obs,nzsd,loc=[getattr(elliptic.misfit,i) for i in ('obs','nzsd','loc')]
elliptic_latent = Elliptic(nx=nx,ny=ny,SNR=SNR,obs=obs,nzsd=nzsd,loc=loc)
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1

# define the autoencoder (AE)
# load data
ensbl_sz = 500
folder = '../train_NN'
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
X=loaded['X']
# pre-processing: scale X to 0-1
# X-=np.nanmin(X,axis=(1,2),keepdims=True) # try axis=(1,2,3)
# X/=np.nanmax(X,axis=(1,2),keepdims=True)
# split train/test
num_samp=X.shape[0]
# n_tr=np.int(num_samp*.75)
# x_train,y_train=X[:n_tr],Y[:n_tr]
# x_test,y_test=X[n_tr:],Y[n_tr:]
tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
x_train,x_test=X[tr_idx],X[te_idx]

# define VAE
half_depth=5; latent_dim=elliptic_latent.pde.V.dim()
repatr_out=False
beta=.1
# activation='linear'
activation=tf.keras.layers.LeakyReLU(alpha=0.01)
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
vae=VAE(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim, repatr_out=repatr_out,
        activation=activation, optimizer=optimizer, beta=beta)
# nll = lambda x: [-elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x_i.numpy().flatten()))[0] for x_i in x]
# # nll = lambda x: tf.map_fn(lambda x_i:-elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x_i.numpy().flatten()))[0], x)
# nll = lambda x,y: [(elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x[i].numpy().flatten()))[0]
#                     -elliptic.get_geom(elliptic.prior.gen_vector(y[i].numpy().flatten()))[0])**2 for i in range(x.shape[0])]
# vae=VAE(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim, repatr_out=repatr_out,
#         activation=activation, optimizer=optimizer, custom_loss=nll, run_eagerly=True)
folder='./saved_model'
import time
ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
f_name=['vae_'+i+'_'+algs[alg_no]+str(ensbl_sz)+'-'+ctime for i in ('fullmodel','encoder','decoder')]
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
fig,axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15,5), facecolor='white')
plt.ion()
# plt.show(block=True)
u_f = df.Function(elliptic.pde.V)
u_f_lat = df.Function(elliptic_latent.pde.V)
n_dif = 1000
dif = np.zeros(n_dif)
loaded=np.load(file=os.path.join('../train_NN',algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
prng=np.random.RandomState(2020)
sel4eval = prng.choice(num_samp,size=n_dif,replace=False)
X=loaded['X'][sel4eval]
sel4print = prng.choice(n_dif,size=10,replace=False)
prog=np.ceil(n_dif*(.1+np.arange(0,1,.1)))
for n in range(n_dif):
    u=elliptic.prior.gen_vector(X[n])
    # encode
    u_encoded=vae.encode(u.get_local()[None,:])
    # decode
    u_decoded=vae.decode(u_encoded)
    # test difference
    dif_ = np.abs(X[n] - u_decoded)
    dif[n] = np.linalg.norm(dif_)/np.linalg.norm(X[n])
    
    if n+1 in prog:
        print('{0:.0f}% evaluation has been completed.'.format(np.float(n+1)/n_dif*100))
    
    # plot
    if n in sel4print:
        print('Difference between the original and reconstructed values: min ({}), med ({}), max ({})\n'.format(dif_.min(),np.median(dif_),dif_.max()))
        plt.clf()
        ax=axes.flat[0]
        plt.axes(ax)
        u_f.vector().set_local(u)
        subfig=df.plot(u_f)
        plt.title('Original Sample')
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        fig.colorbar(subfig, cax=cax)
        
        ax=axes.flat[1]
        plt.axes(ax)
        u_f_lat.vector().set_local(u_encoded.flatten())
        subfig=df.plot(u_f_lat)
        plt.title('Latent Sample')
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        fig.colorbar(subfig, cax=cax)
        
        ax=axes.flat[2]
        plt.axes(ax)
        u_f.vector().set_local(u_decoded.flatten())
        subfig=df.plot(u_f)
        plt.title('Reconstructed Sample')
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        fig.colorbar(subfig, cax=cax)
        plt.draw()
        plt.pause(1.0/10.0)

# save to file
import pandas as pd
folder='./result'
file=os.path.join(folder,'dif-'+ctime+'.txt')
np.savetxt(file,dif)
con_str=np.array2string(np.array(node_sizes),separator=',').replace('[','').replace(']','') if 'node_sizes' in locals() or 'node_sizes' in globals() else str(half_depth)
# act_str=','.join([val.__name__ if type(val).__name__=='function' else val.name if callable(val) else val for val in activations.values()])
act_str=activation.__name__ if type(activation).__name__=='function' else activation.name if callable(activation) else activation
dif_sumry=[dif.min(),np.median(dif),dif.max()]
dif_str=np.array2string(np.array(dif_sumry),precision=2,separator=',').replace('[','').replace(']','') # formatter={'float': '{: 0.2e}'.format}
sumry_header=('Time','half_depth/node_sizes','latent_dim','repatr_out','activation','beta','dif (min,med,max)')
sumry_np=np.array([ctime,con_str,latent_dim,str(repatr_out),act_str,beta,dif_str])
file=os.path.join(folder,'dif_sumry.txt')
if not os.path.isfile(file):
    np.savetxt(file,sumry_np[None,:],fmt="%s",delimiter='\t|',header='\t|'.join(sumry_header))
else:
    with open(file, "ab") as f:
        np.savetxt(f,sumry_np[None,:],fmt="%s",delimiter='\t|')
sumry_pd=pd.DataFrame(data=[sumry_np],columns=sumry_header)
file=os.path.join(folder,'dif_sumry.csv')
if not os.path.isfile(file):
    sumry_pd.to_csv(file,index=False,header=sumry_header)
else:
    sumry_pd.to_csv(file,index=False,mode='a',header=False)

# read data and construct plot functions
u_f = df.Function(elliptic.pde.V)
# read MAP
try:
    f=df.HDF5File(elliptic.pde.mpi_comm, os.path.join('../result',"MAP_SNR"+str(SNR)+".h5"), "r")
    f.read(u_f,"parameter")
    f.close()
    print('MAP loaded!')
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
f_name='vae_'+algs[alg_no]+str(ensbl_sz)+'-'+ctime
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(folder,f_name+'.png'),bbox_inches='tight')
# plt.show()