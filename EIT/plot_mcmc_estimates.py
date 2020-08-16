"""
Plot estimates of uncertainty field u in EIT inverse problem.
Shiwei Lan @ U of Warwick, 2016
-------------------------------
Modified for DREAM August 2020 @ ASU
"""

import os,pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mp

from EIT import EIT
import sys
sys.path.append( "../" )
from nn.ae import AutoEncoder
from nn.cae import ConvAutoEncoder
from nn.vae import VAE
from tensorflow.keras.models import load_model

# functions needed to make even image size
def pad(A,width=[1]):
    shape=A.shape
    if len(width)==1: width=np.tile(width,len(shape))
    if not any(width): return A
    assert len(width)==len(shape), 'non-matching padding width!'
    pad_width=tuple((0,i) for i in width)
    return np.pad(A, pad_width)
def chop(A,width=[1]):
    shape=A.shape
    if len(width)==1: width=np.tile(width,len(shape))
    if not any(width): return A
    assert len(width)==len(shape), 'non-matching chopping width!'
    chop_slice=tuple(slice(0,-i) for i in width)
    return A[chop_slice]

# define the inverse problem
np.random.seed(2020)
n_el = 16
bbox = [[-1,-1],[1,1]]
meshsz = .05
el_dist, step = 1, 1
anomaly = [{'x': 0.4, 'y': 0.4, 'd': 0.2, 'perm': 10},
           {'x': -0.4, 'y': -0.4, 'd': 0.2, 'perm': 0.1}]
lamb=1e-2
eit=EIT(n_el=n_el,bbox=bbox,meshsz=meshsz,el_dist=el_dist,step=step,anomaly=anomaly,lamb=lamb)
meshsz = 0.1
eit_latent=EIT(n_el=n_el,bbox=bbox,meshsz=meshsz,el_dist=el_dist,step=step,anomaly=anomaly,lamb=1,obs=eit.obs)

# ##------ define networks ------##
# # training data algorithms
# algs=['EKI','EKS']
# num_algs=len(algs)
# alg_no=1
# # load data
# ensbl_sz = 100
# folder = './train_NN'
# 
# ##---- AUTOENCODER ----##
# AE={0:'ae',1:'cae',2:'vae'}[0]
# # prepare for training data
# if 'c' in AE:
#     loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XimgY.npz'))
#     X=loaded['X']
#     X=X[:,:-1,:-1,None]
# else :
#     loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
#     X=loaded['X']
# num_samp=X.shape[0]
# # n_tr=np.int(num_samp*.75)
# # x_train=X[:n_tr]
# # x_test=X[n_tr:]
# tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
# te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
# x_train,x_test=X[tr_idx],X[te_idx]
# # define autoencoder
# if AE=='ae':
#     half_depth=3; latent_dim=eit_latent.dim
#     droprate=0.
# #     activation='linear'
#     activation=tf.keras.layers.LeakyReLU(alpha=2.00)
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
#     lambda_=0.
#     autoencoder=AutoEncoder(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim, droprate=droprate,
#                             activation=activation, optimizer=optimizer)
# elif AE=='cae':
#     num_filters=[16,8]; latent_dim=eit_latent.dim
# #     activations={'conv':tf.keras.layers.LeakyReLU(alpha=0.1),'latent':None} # [16,1]
#     activations={'conv':'elu','latent':'linear'}
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
#     autoencoder=ConvAutoEncoder(x_train.shape[1:], num_filters=num_filters, latent_dim=latent_dim,
#                                 activations=activations, optimizer=optimizer)
# elif AE=='vae':
#         half_depth=5; latent_dim=eit_latent.pde.V.dim()
#         repatr_out=False; beta=1.
#         activation='elu'
# #         activation=tf.keras.layers.LeakyReLU(alpha=0.01)
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
#         autoencoder=VAE(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim, repatr_out=repatr_out,
#                         activation=activation, optimizer=optimizer, beta=beta)
# f_name=[AE+'_'+i+'_'+algs[alg_no]+str(ensbl_sz) for i in ('fullmodel','encoder','decoder')]
# # load autoencoder
# try:
#     autoencoder.model=load_model(os.path.join(folder,f_name[0]+'.h5'),custom_objects={'loss':None})
#     print(f_name[0]+' has been loaded!')
#     autoencoder.encoder=load_model(os.path.join(folder,f_name[1]+'.h5'),custom_objects={'loss':None})
#     print(f_name[1]+' has been loaded!')
#     autoencoder.decoder=load_model(os.path.join(folder,f_name[2]+'.h5'),custom_objects={'loss':None})
#     print(f_name[2]+' has been loaded!')
# except:
#     print('\nNo autoencoder found. Training {}...\n'.format(AE))
#     epochs=200
#     patience=0
#     noise=0.2
#     kwargs={'patience':patience}
#     if AE=='ae' and noise: kwargs['noise']=noise
#     autoencoder.train(x_train,x_test=x_test,epochs=epochs,batch_size=64,verbose=1,**kwargs)
#     # save autoencoder
#     autoencoder.model.save(os.path.join(folder,f_name[0]+'.h5'))
#     autoencoder.encoder.save(os.path.join(folder,f_name[1]+'.h5'))
#     autoencoder.decoder.save(os.path.join(folder,f_name[2]+'.h5'))

# algorithms
algs=('pCN','infMALA','infHMC','epCN','einfMALA','einfHMC','DREAMpCN','DREAMinfMALA','DREAMinfHMC','DRinfmHMC')
alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','e-pCN','e-$\infty$-MALA','e-$\infty$-HMC','DREAM-pCN','DREAM-$\infty$-MALA','DREAM-$\infty$-HMC','DR-$\infty$-HMC')
num_algs=len(algs)
# obtain estimates
folder = './analysis'
mean_v=np.zeros((num_algs,eit.dim))
std_v=np.zeros((num_algs,eit.dim))
if os.path.exists(os.path.join(folder,'mcmc_est.pckl')):
    with open(os.path.join(folder,'mcmc_est.pckl'),"rb") as f:
        mean_v,std_v=pickle.load(f)
else:
    # preparation for estimates
    pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
    num_samp=2000
    prog=np.ceil(num_samp*(.1+np.arange(0,1,.1)))
    for i in range(num_algs):
        found=False
        # obtain weights
        wts=np.ones(num_samp)/num_samp
        if 'DREAM' in algs[i]:
            for f_i in pckl_files:
                if '_'+algs[i]+'_' in f_i:
                    try:
                        f=open(os.path.join(folder,f_i),'rb')
                        f_read=pickle.load(f)
                        logwts=f_read[4]; logwts-=logwts.max()
                        wts=np.exp(logwts); wts/=np.sum(wts)
                        f.close()
                        print(f_i+' has been read!')
                    except:
                        pass
        bip=eit_latent if 'DREAM' in algs[i] else eit
        # calculate posterior estimates
        found=False
        samp_mean=np.zeros(eit.dim); samp_std=np.zeros(eit.dim)
#         num_read=0
        for f_i in pckl_files:
            if '_'+algs[i]+'_' in f_i:
                try:
                    f=open(os.path.join(folder,f_i),'rb')
                    f_read=pickle.load(f)
                    samp=f_read[3]
                    for s in range(num_samp):
                        if s+1 in prog:
                            print('{0:.0f}% has been completed.'.format(np.float(s+1)/num_samp*100))
                        u=samp[s]
                        if '_whitened_latent' in f_i: u=bip.prior['cov'].dot(u)
                        if 'DREAM' in algs[i]:
                            if 'c' in AE:
                                u_latin=eit.vec2img(u)
                                width=tuple(np.mod(i,2) for i in u_latin.shape)
                                u_latin=chop(u_latin,width)[None,:,:,None] if autoencoder.activations['latent'] is None else u_latin.flatten()[None,:]
                                u=eit.img2vec(pad(np.squeeze(autoencoder.decode(u_latin)),width))
                            else:
                                u_latin=u[None,:]
                                u=autoencoder.decode(u_latin).flatten()
#                         else:
#                             u=u_
                        if '_whitened_emulated' in f_i: u=eit.prior['cov'].dot(u)
                        samp_mean+=wts[s]*u
                        samp_std+=wts[s]*u**2
#                         num_read+=1
                    f.close()
                    print(f_i+' has been read!')
                    found=True
                except:
                    pass
        if found:
#             samp_mean=samp_mean/num_read; samp_std=samp_std/num_read
            mean_v[i]=samp_mean
            std_v[i]=np.sqrt(samp_std - samp_mean*samp_mean)
    # save
    with open(os.path.join(folder,'mcmc_est.pckl'),"wb") as f:
        pickle.dump([mean_v,std_v],f)

# plot
num_algs-=1
plt.rcParams['image.cmap'] = 'jet'
num_rows=3
# posterior mean
fig,axes = plt.subplots(nrows=num_rows,ncols=np.int(np.ceil((num_algs)/num_rows)),sharex=True,sharey=True,figsize=(11,10))
sub_figs = [None]*len(axes.flat)
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
#     if i==0:
#         # plot MAP
#         try:
#             f=open(os.path.join('./result',str(eit.gdim)+'d_EIT_MAP_dim'+str(eit.dim)+'.pckl'),'rb')
#             MAP=pickle.load(f)[0]
#             f.close()
#             sub_figs[i]=eit.plot(MAP,ax=ax)
# #             fig.colorbar(sub_figs[i],ax=ax)
#             ax.set_title('MAP')
#         except:
#             pass
#     elif 1<=i<=num_algs:
    sub_figs[i]=eit.plot(mean_v[i],ax=ax)
    ax.set_title(alg_names[i])
    ax.set_aspect('auto')
    plt.axis([-1, 1, -1, 1])
# set color bar
# cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
# plt.colorbar(sub_fig, cax=cax, **kw)
from util.common_colorbar import common_colorbar
fig=common_colorbar(fig,axes,sub_figs)
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/mcmc_estimates_mean.png',bbox_inches='tight')
# plt.show()

# posterior std
fig,axes = plt.subplots(nrows=num_rows,ncols=np.int(np.ceil((num_algs)/num_rows)),sharex=True,sharey=True,figsize=(11,10))
sub_figs = [None]*len(axes.flat)
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    sub_figs[i]=eit.plot(std_v[i],ax=ax)
    ax.set_title(alg_names[i])
    ax.set_aspect('auto')
    plt.axis([-1, 1, -1, 1])
# set color bar
# cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
# plt.colorbar(sub_fig, cax=cax, **kw)
from util.common_colorbar import common_colorbar
fig=common_colorbar(fig,axes,sub_figs)
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/mcmc_estimates_std.png',bbox_inches='tight')
# plt.show()