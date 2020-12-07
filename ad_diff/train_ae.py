"""
This is to test AE in reconstructing samples.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import sys,os,pickle
sys.path.append( '../' )
from advdiff import advdiff
from pde import *
from nn.ae import AutoEncoder
from tensorflow.keras.models import load_model

# set to warn only once for the same warnings
tf.get_logger().setLevel('ERROR')
# set random seed
seed=2020
np.random.seed(seed)
tf.random.set_seed(seed)

## define Advection-Diffusion inverse problem ##
# mesh = dl.Mesh('ad_10k.xml')
meshsz = (61,61)
eldeg = 1
gamma = 2.; delta = 10.
rel_noise = .5
nref = 1
adif = advdiff(mesh=meshsz, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed)
adif.prior.V=adif.prior.Vh
adif.misfit.obs=np.array([dat.get_local() for dat in adif.misfit.d.data]).flatten()
# latent
meshsz_latent = (21,21)
pde_latent = TimeDependentAD(mesh=meshsz_latent, eldeg=eldeg)
V_latent = pde_latent.Vh[0]
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1

# define the autoencoder (AE)
# load data
ensbl_sz = 500
folder = './train_NN_eldeg'+str(eldeg)
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

# define AE
half_depth=3; latent_dim=V_latent.dim()
# node_sizes=[4,8,4,2,4,8,4]
droprate=0.
# activation='linear'
# activation=lambda x:1.1*x
# activation=tf.keras.layers.LeakyReLU(alpha=1.2)
activation='elu'
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
lambda_=0. # contractive autoencoder
ae=AutoEncoder(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim, droprate=droprate,
               activation=activation, optimizer=optimizer)
savepath=folder+'/AE/saved_model'
if not os.path.exists(savepath): os.makedirs(savepath)
import time
ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
f_name=['ae_'+i+'_'+algs[alg_no]+str(ensbl_sz)+'-'+ctime for i in ('fullmodel','encoder','decoder')]
try:
    ae.model=load_model(os.path.join(savepath,f_name[0]+'.h5'),custom_objects={'loss':None})
    print(f_name[0]+' has been loaded!')
    ae.encoder=load_model(os.path.join(savepath,f_name[1]+'.h5'),custom_objects={'loss':None})
    print(f_name[1]+' has been loaded!')
    ae.decoder=load_model(os.path.join(savepath,f_name[2]+'.h5'),custom_objects={'loss':None})
    print(f_name[2]+' has been loaded!')
except Exception as err:
    print(err)
    print('Train AutoEncoder...\n')
    epochs=200
    patience=0
    noise=0. # denoising autoencoder
    import timeit
    t_start=timeit.default_timer()
    ae.train(x_train,x_test=x_test,epochs=epochs,batch_size=64,verbose=1,patience=patience,noise=noise)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training AE: {}'.format(t_used))
    # save AE
    ae.model.save(os.path.join(savepath,f_name[0]+'.h5'))
    ae.encoder.save(os.path.join(savepath,f_name[1]+'.h5'))
    ae.decoder.save(os.path.join(savepath,f_name[2]+'.h5'))

# plot
import matplotlib.pyplot as plt
fig,axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15,5), facecolor='white')
plt.ion()
n_dif = 1000
dif = np.zeros(n_dif)
# loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
prng=np.random.RandomState(2020)
sel4eval = prng.choice(num_samp,size=n_dif,replace=False)
# X=loaded['X'][sel4eval]
X=X[sel4eval]
sel4print = prng.choice(n_dif,size=10,replace=False)
prog=np.ceil(n_dif*(.1+np.arange(0,1,.1)))
u_f = df.Function(adif.prior.V)
eldeg = adif.prior.V.ufl_element().degree()
if eldeg>1:
    V_P1 = df.FunctionSpace(adif.mesh,'Lagrange',1)
    d2v = df.dof_to_vertex_map(V_P1)
    u_f1 = df.Function(V_P1)
else:
    u_f1 = u_f
u_f_lat = df.Function(V_latent)
for n in range(n_dif):
    u=X[n]
    # encode
    u_encoded=ae.encode(u[None,:])
    # decode
    u_decoded=ae.decode(u_encoded)
    # test difference
    dif_ = np.abs(X[n] - u_decoded)
    dif[n] = np.linalg.norm(dif_)/np.linalg.norm(X[n])
    
    if n+1 in prog:
        print('{0:.0f}% evaluation has been completed.'.format(np.float(n+1)/n_dif*100))
    
    # plot
    if n in sel4print:
        print('Difference between the original and reconstructed values: min ({}), med ({}), max ({})\n'.format(dif_.min(),np.median(dif_),dif_.max()))
        ax=axes.flat[0]
        plt.axes(ax)
        u_f1.vector().set_local(u)
        subfig=df.plot(u_f1)
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
        u_f1.vector().set_local(u_decoded.flatten())
        subfig=df.plot(u_f1)
        plt.title('Reconstructed Sample')
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        fig.colorbar(subfig, cax=cax)
        plt.draw()
        plt.pause(1.0/10.0)

# save to file
import pandas as pd
savepath=folder+'/AE/summary'
if not os.path.exists(savepath): os.makedirs(savepath)
file=os.path.join(savepath,'dif-'+ctime+'.txt')
np.savetxt(file,dif)
con_str=np.array2string(np.array(node_sizes),separator=',').replace('[','').replace(']','') if 'node_sizes' in locals() or 'node_sizes' in globals() else str(half_depth)
# act_str=','.join([val.__name__ if type(val).__name__=='function' else val.name if callable(val) else val for val in activations.values()])
act_str=activation.__name__ if type(activation).__name__=='function' else activation.name if callable(activation) else activation
dif_sumry=[dif.min(),np.median(dif),dif.max()]
dif_str=np.array2string(np.array(dif_sumry),precision=2,separator=',').replace('[','').replace(']','') # formatter={'float': '{: 0.2e}'.format}
sumry_header=('Time','half_depth/node_sizes','latent_dim','droprate','activation','noise_std','contractive_lambda','dif (min,med,max)')
sumry_np=np.array([ctime,con_str,latent_dim,droprate,act_str,noise,lambda_,dif_str])
file=os.path.join(savepath,'dif_sumry.txt')
if not os.path.isfile(file):
    np.savetxt(file,sumry_np[None,:],fmt="%s",delimiter='\t|',header='\t|'.join(sumry_header))
else:
    with open(file, "ab") as f:
        np.savetxt(f,sumry_np[None,:],fmt="%s",delimiter='\t|')
sumry_pd=pd.DataFrame(data=[sumry_np],columns=sumry_header)
file=os.path.join(savepath,'dif_sumry.csv')
if not os.path.isfile(file):
    sumry_pd.to_csv(file,index=False,header=sumry_header)
else:
    sumry_pd.to_csv(file,index=False,mode='a',header=False)


# read data and construct plot functions
# load map
MAP_file=os.path.join(os.getcwd(),'properties/MAP.xdmf')
u_f=df.Function(adif.prior.V, name='MAP')
if os.path.isfile(MAP_file):
    f=df.XDMFFile(adif.mpi_comm,MAP_file)
    f.read_checkpoint(u_f,'m',0)
    f.close()
    u = u_f.vector()
else:
    u=adif.get_MAP(SAVE=True)
# encode
u_encoded=ae.encode(u_f.compute_vertex_values(adif.mesh)[d2v][None,:] if eldeg>1 else u.get_local()[None,:])
# decode
u_decoded=ae.decode(u_encoded)

# plot
plt.rcParams['image.cmap'] = 'jet'
fig,axes = plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True,figsize=(15,5))
sub_figs=[None]*3
# plot
plt.axes(axes.flat[0])
u_f.vector().set_local(u)
sub_figs[0]=df.plot(u_f)
plt.title('Original')
plt.axes(axes.flat[1])
u_f_lat.vector().set_local(u_encoded.flatten())
sub_figs[1]=df.plot(u_f_lat)
plt.title('Latent')
plt.axes(axes.flat[2])
u_f1.vector().set_local(u_decoded.flatten())
sub_figs[2]=df.plot(u_f1)
plt.title('Reconstructed')
# add common colorbar
from util.common_colorbar import common_colorbar
fig=common_colorbar(fig,axes,sub_figs)

# save plots
f_name='ae_'+algs[alg_no]+str(ensbl_sz)+'-'+ctime
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(savepath,f_name+'.png'),bbox_inches='tight')
# plt.show()