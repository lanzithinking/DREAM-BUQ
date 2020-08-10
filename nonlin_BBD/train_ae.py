"""
This is to test AE in reconstructing samples.
"""

import numpy as np
import tensorflow as tf
import sys,os,pickle
sys.path.append( '../' )
from BBD import BBD
from nn.ae import AutoEncoder
from tensorflow.keras.models import load_model

# set to warn only once for the same warnings
tf.get_logger().setLevel('ERROR')
# set random seed
np.random.seed(2020)
tf.random.set_seed(2020)

## define the BBD inverse problem ##
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
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1

# define the autoencoder (AE)
# load data
ensbl_sz = 100
# folder = './train_NN'
# loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_X.npz')) # training points from EnK are too crowded to be a good set for AE
# X=loaded['X']
import pickle
with open('./analysis/BBD_infHMC_dim4_2020-08-07-13-57-16.pckl','rb') as f:
    loaded=pickle.load(f)
X=loaded[3]
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
half_depth=3; latent_dim=2
# node_sizes=[4,8,4,2,4,8,4]
droprate=0.
# activation='linear'
# activation=lambda x:1.1*x
activation=tf.keras.layers.LeakyReLU(alpha=2.)
optimizer=tf.keras.optimizers.Adam(learning_rate=0.005,amsgrad=True)
lambda_=0. # contractive autoencoder
ae=AutoEncoder(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim, droprate=droprate,
               activation=activation, optimizer=optimizer)
folder='./train_NN/AE/saved_model'
import time
ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
f_name=['ae_'+i+'_'+algs[alg_no]+str(ensbl_sz)+'-'+ctime for i in ('fullmodel','encoder','decoder')]
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
    epochs=1000
    patience=10
    noise=0. # denoising autoencoder
    import timeit
    t_start=timeit.default_timer()
    ae.train(x_train,x_test=x_test,epochs=epochs,batch_size=64,verbose=1,patience=patience,noise=noise)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training AE: {}'.format(t_used))
    # save AE
    ae.model.save(os.path.join(folder,f_name[0]+'.h5'))
    ae.encoder.save(os.path.join(folder,f_name[1]+'.h5'))
    ae.decoder.save(os.path.join(folder,f_name[2]+'.h5'))

# plot
n_dif = 1000
dif = np.zeros(n_dif)
# loaded=np.load(file=os.path.join('./train_NN',algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_X.npz'))
prng=np.random.RandomState(2020)
sel4eval = prng.choice(num_samp,size=n_dif,replace=False)
# X=loaded['X'][sel4eval]
X=X[sel4eval]
sel4print = prng.choice(n_dif,size=10,replace=False)
prog=np.ceil(n_dif*(.1+np.arange(0,1,.1)))
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

# save to file
import pandas as pd
folder='./train_NN/AE/summary'
file=os.path.join(folder,'dif-'+ctime+'.txt')
np.savetxt(file,dif)
con_str=np.array2string(np.array(node_sizes),separator=',').replace('[','').replace(']','') if 'node_sizes' in locals() or 'node_sizes' in globals() else str(half_depth)
# act_str=','.join([val.__name__ if type(val).__name__=='function' else val.name if callable(val) else val for val in activations.values()])
act_str=activation.__name__ if type(activation).__name__=='function' else activation.name if callable(activation) else activation
dif_sumry=[dif.min(),np.median(dif),dif.max()]
dif_str=np.array2string(np.array(dif_sumry),precision=2,separator=',').replace('[','').replace(']','') # formatter={'float': '{: 0.2e}'.format}
sumry_header=('Time','half_depth/node_sizes','latent_dim','droprate','activation','noise_std','contractive_lambda','dif (min,med,max)')
sumry_np=np.array([ctime,con_str,latent_dim,droprate,act_str,noise,lambda_,dif_str])
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
# import pandas as pd
test=pd.DataFrame(x_test,columns=['$u_{}$'.format(i) for i in np.arange(1,5)])
reconstructed=pd.DataFrame(ae.model.predict(test),columns=['$u_{}$'.format(i) for i in np.arange(1,5)])

# plot
import matplotlib.pyplot as plt
import seaborn as sns
# plt.rcParams['image.cmap'] = 'jet'
g=sns.PairGrid(test)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot,lw=2,legend=False)
g.fig.suptitle('Original')
g.fig.subplots_adjust(top=0.95)
f_name='ae_'+algs[alg_no]+str(ensbl_sz)+'_original'+'-'+ctime
plt.savefig(os.path.join(folder,f_name+'.png'),bbox_inches='tight')

g=sns.PairGrid(reconstructed)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot,lw=2)
g.fig.suptitle('Reconstructed')
g.fig.subplots_adjust(top=0.95)
f_name='ae_'+algs[alg_no]+str(ensbl_sz)+'_reconstructed'+'-'+ctime
plt.savefig(os.path.join(folder,f_name+'.png'),bbox_inches='tight')
