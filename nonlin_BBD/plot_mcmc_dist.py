"""
Plot pairwise distribution of two samples
-----------------------------------------
Shiwei Lan @ ASU 2020
"""

import os,pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# the inverse problem
from BBD import BBD

import sys
sys.path.append( "../" )
from nn.ae import AutoEncoder
from nn.cae import ConvAutoEncoder
from nn.vae import VAE
from tensorflow.keras.models import load_model

np.random.seed(2020)
# set up
try:
    with open('./result/BBD.pickle','rb') as f:
        [nz_var,pr_cov,A,true_input,y]=pickle.load(f)
    print('Data loaded!\n')
    kwargs={'true_input':true_input,'A':A,'y':y}
except:
    print('No data found. Generate new data...\n')
    d=4; m=100
    nz_var=1; pr_cov=1
    true_input=np.random.randint(d,size=d)
    A=np.random.rand(m,d)
    kwargs={'true_input':true_input,'A':A}
bbd=BBD(nz_var=nz_var,pr_cov=pr_cov,**kwargs)
y=bbd.y
bbd.prior={'mean':np.zeros(bbd.input_dim),'cov':np.diag(bbd.pr_cov) if np.ndim(bbd.pr_cov)==1 else bbd.pr_cov,'sample':bbd.sample}
# set up latent
latent_dim=3
class BBD_lat:
    def __init__(self,input_dim):
        self.input_dim=input_dim
    def sample(self,num_samp=1):
        samp=np.random.randn(num_samp,self.input_dim)
        return np.squeeze(samp)
bbd_latent=BBD_lat(latent_dim)
bbd_latent.prior={'mean':np.zeros(bbd_latent.input_dim),'cov':np.eye(bbd_latent.input_dim), 'sample':bbd_latent.sample}

##------ define networks ------##
# training data algorithms
algs=['EKI','EKS']
alg_no=1
# load data
ensbl_sz = 100
folder='./train_NN'

##---- AUTOENCODER ----##
AE={0:'ae',1:'cae',2:'vae'}[0]
# prepare for training data
if 'c' in AE:
    loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XimgY.npz'))
    X=loaded['X']
    X=X[:,:-1,:-1,None]
else :
    loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_X.npz'))
    X=loaded['X']
num_samp=X.shape[0]
# n_tr=np.int(num_samp*.75)
# x_train=X[:n_tr]
# x_test=X[n_tr:]
tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
x_train,x_test=X[tr_idx],X[te_idx]
# define autoencoder
if AE=='ae':
    half_depth=2; latent_dim=3
    droprate=0.
#     activation='linear'
    activation=tf.keras.layers.LeakyReLU(alpha=2.)
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
    lambda_=0.
    autoencoder=AutoEncoder(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim, droprate=droprate,
                            activation=activation, optimizer=optimizer)
elif AE=='cae':
    num_filters=[16,8]; latent_dim=2
#     activations={'conv':tf.keras.layers.LeakyReLU(alpha=0.1),'latent':None} # [16,1]
    activations={'conv':'elu','latent':'linear'}
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    autoencoder=ConvAutoEncoder(x_train.shape[1:], num_filters=num_filters, latent_dim=latent_dim,
                                activations=activations, optimizer=optimizer)
elif AE=='vae':
        half_depth=5; latent_dim=2
        repatr_out=False; beta=1.
        activation='elu'
#         activation=tf.keras.layers.LeakyReLU(alpha=0.01)
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
        autoencoder=VAE(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim, repatr_out=repatr_out,
                        activation=activation, optimizer=optimizer, beta=beta)
f_name=[AE+'_'+i+'_'+algs[alg_no]+str(ensbl_sz) for i in ('fullmodel','encoder','decoder')]
# load autoencoder
try:
    autoencoder.model=load_model(os.path.join(folder,f_name[0]+'.h5'),custom_objects={'loss':None})
    print(f_name[0]+' has been loaded!')
    autoencoder.encoder=load_model(os.path.join(folder,f_name[1]+'.h5'),custom_objects={'loss':None})
    print(f_name[1]+' has been loaded!')
    autoencoder.decoder=load_model(os.path.join(folder,f_name[2]+'.h5'),custom_objects={'loss':None})
    print(f_name[2]+' has been loaded!')
except:
    print('\nNo autoencoder found. Training {}...\n'.format(AE))
    epochs=200
    patience=0
    noise=0.
    kwargs={'patience':patience}
    if AE=='ae' and noise: kwargs['noise']=noise
    autoencoder.train(x_train,x_test=x_test,epochs=epochs,batch_size=64,verbose=1,**kwargs)
    # save autoencoder
    autoencoder.model.save(os.path.join(folder,f_name[0]+'.h5'))
    autoencoder.encoder.save(os.path.join(folder,f_name[1]+'.h5'))
    autoencoder.decoder.save(os.path.join(folder,f_name[2]+'.h5'))

# algorithms
# algs=('pCN','infMALA','infHMC','epCN','einfMALA','einfHMC','DREAMpCN','DREAMinfMALA','DREAMinfHMC','DRinfmHMC')
# alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','e-pCN','e-$\infty$-MALA','e-$\infty$-HMC','DREAM-pCN','DREAM-$\infty$-MALA','DREAM-$\infty$-HMC','DR-$\infty$-HMC')
algs=['infHMC','einfHMC','DREAMinfHMC']
alg_names=('$\infty$-HMC','e-$\infty$-HMC','DREAM-$\infty$-HMC')
num_algs=len(algs)
folder='./analysis'
pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
num_samp=10000
samps=np.zeros((num_algs,num_samp,bbd.input_dim))

for i in range(num_algs):
    found=False
    for f_i in pckl_files:
        if 'HMC' in algs[i]:
            if '_'+algs[i]+'_' in f_i:
                try:
                    f=open(os.path.join(folder,f_i),'rb')
                    f_read=pickle.load(f)
                    samp=f_read[3]
                    f.close()
                    print(f_i+' has been read!')
                    found=True; break
                except:
                    pass
    if found:
        if 'DREAM' in algs[i]:
            if 'c' in AE:
                samp_lat=vec2img(samp)
                width=tuple(np.mod(i,2) for i in samp_lat.shape[1:])
                samp_lat=chop(samp_lat,width)[None,:,:,None] if autoencoder.activations['latent'] is None else samp_lat.flatten()[None,:]
                samp=(pad(np.squeeze(autoencoder.decode(samp_lat)),width)).reshape((samp.shape[0],-1))
            else:
                samp=autoencoder.decode(samp)
        samps[i]=samp
#         samp=pd.DataFrame(samp,columns=['$u_{}$'.format(i) for i in np.arange(1,samp.shape[1]+1)])
#         g=sns.PairGrid(samp,diag_sharey=False)
#         g.map_upper(plt.scatter)
#         g.map_lower(sns.kdeplot)
#         g.map_diag(sns.kdeplot,lw=2,legend=False)
# #         g.fig.suptitle(algs[i])
# #         g.fig.subplots_adjust(top=0.95)
#         plt.savefig(os.path.join(folder,algs[i]+'_dist.png'),bbox_inches='tight')

# form a big data frame
alg_array=np.hstack([[alg_names[i]]*num_samp for i in range(num_algs)])
df_samps=pd.DataFrame(samps.reshape((-1,bbd.input_dim)),columns=['$u_{}$'.format(i) for i in np.arange(1,bbd.input_dim+1)])
df_samps['algorithm']=alg_array.flatten()
g=sns.PairGrid(df_samps,hue='algorithm',diag_sharey=False)
g.map_upper(plt.scatter,s=1,alpha=0.5)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot,lw=2)
g.add_legend()
# g.fig.suptitle('MCMC')
# g.fig.subplots_adjust(top=0.95)
plt.savefig(os.path.join(folder,'allmcmc_dist.png'),bbox_inches='tight')