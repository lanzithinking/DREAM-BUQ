"""
This is to test DNN in emulating (extracted) gradients compared with those exactly calculated.
"""

import numpy as np
import tensorflow as tf
import sys,os,pickle
sys.path.append( '../' )
from EIT import EIT
from nn.dnn import DNN
from tensorflow.keras.models import load_model

# tf.compat.v1.disable_eager_execution() # needed to train with custom loss # comment to plot
# set random seed
np.random.seed(2020)
tf.random.set_seed(2020)

## define the EIT inverse problem ##
n_el = 16
bbox = [[-1,-1],[1,1]]
meshsz = .04
el_dist, step = 1, 1
anomaly = [{'x': 0.4, 'y': 0.4, 'd': 0.2, 'perm': 10},
           {'x': -0.4, 'y': -0.4, 'd': 0.2, 'perm': 0.1}]
nz_var=1e-2; lamb=1e-1; rho=.25
eit=EIT(n_el=n_el,bbox=bbox,meshsz=meshsz,el_dist=el_dist,step=step,anomaly=anomaly,nz_var=nz_var,lamb=lamb,rho=rho)
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1

# define the emulator (DNN)
# load data
ensbl_sz = 100
folder = './train_NN'
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
X=loaded['X']
Y=loaded['Y']
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
y_train,y_test=Y[tr_idx],Y[te_idx]

# define DNN
depth=3
# node_sizes=[eit.dim,8,16,32,64,len(eit.obs)]
# activations={'hidden':tf.keras.layers.LeakyReLU(alpha=.01),'output':'linear'}
# activations={'hidden':tf.math.sin,'output':'linear'}
activations={'hidden':'softplus','output':'linear'}
droprate=0.
# sin_init=lambda n:tf.random_uniform_initializer(minval=-tf.math.sqrt(6/n), maxval=tf.math.sqrt(6/n))
# kernel_initializers={'hidden':'he_uniform','output':'he_uniform'}
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
# optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001)
dnn=DNN(x_train.shape[1], y_train.shape[1], depth=depth, droprate=droprate,
        activations=activations, optimizer=optimizer)
loglik = lambda y: -0.5*tf.math.reduce_sum((y-eit.obs[None,:])**2/eit.nz_var[None,:],axis=1)
# custom_loss = lambda y_true, y_pred: [tf.square(loglik(y_true)-loglik(y_pred)), (y_true-y_pred)/eit.nz_var[None,:]]
# dnn=DNN(x_train.shape[1], y_train.shape[1], depth=depth, droprate=droprate,
#         activations=activations, kernel_initializers=kernel_initializers, optimizer=optimizer, loss=custom_loss)
folder='./train_NN/DNN/saved_model'
if not os.path.exists(folder): os.makedirs(folder)
import time
ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
f_name='dnn_'+algs[alg_no]+str(ensbl_sz)+'-'+ctime
try:
#     dnn.model=load_model(os.path.join(folder,f_name+'.h5'))
#     dnn.model=load_model(os.path.join(folder,f_name+'.h5'),custom_objects={'loss':None})
    dnn.model.load_weights(os.path.join(folder,f_name+'.h5'))
    print(f_name+' has been loaded!')
except Exception as err:
    print(err)
    print('Train DNN...\n')
    epochs=200
    patience=0
    import timeit
    t_start=timeit.default_timer()
    dnn.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training DNN: {}'.format(t_used))
    # save DNN
#     dnn.model.save(os.path.join(folder,f_name+'.h5'))
#     dnn.save(folder,f_name) # fails due to the custom kernel_initializer
    dnn.model.save_weights(os.path.join(folder,f_name+'.h5'))

# select some gradients to evaluate and compare
logLik = lambda x: loglik(dnn.model(x))
import timeit
t_used = np.zeros(2)
import matplotlib.pyplot as plt
fig,axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12,6), facecolor='white')
plt.ion()
n_dif = 100
dif = np.zeros((n_dif,2))
loaded=np.load(file=os.path.join('./train_NN',algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY'+'.npz'))
prng=np.random.RandomState(2020)
sel4eval = prng.choice(num_samp,size=n_dif,replace=False)
X=loaded['X'][sel4eval]; Y=loaded['Y'][sel4eval]
sel4print = prng.choice(n_dif,size=10,replace=False)
prog=np.ceil(n_dif*(.1+np.arange(0,1,.1)))
for n in range(n_dif):
    u=X[n]
    # calculate gradient
    t_start=timeit.default_timer()
    ll_xact,dll_xact = eit.get_geom(u,[0,1],force_posperm=True)[:2]
    t_used[0] += timeit.default_timer()-t_start
    # emulate gradient
    t_start=timeit.default_timer()
    ll_emul = logLik(u[None,:]).numpy()[0]
    dll_emul = dnn.gradient(u[None,:], logLik)
    t_used[1] += timeit.default_timer()-t_start
    # test difference
    dif_fun = np.abs(ll_xact - ll_emul)
    dif_grad = dll_xact - dll_emul
    dif[n] = np.array([dif_fun, np.linalg.norm(dif_grad)/np.linalg.norm(dll_xact)])
    
#     # check the gradient extracted from emulation
#     v=eit.sample()
#     h=1e-4
#     dll_emul_fd_v=(logLik(u[None,:]+h*v[None,:])-logLik(u[None,:]))/h
#     reldif = abs(dll_emul_fd_v - dll_emul.flatten().dot(v))/np.linalg.norm(v)
#     print('Relative difference between finite difference and extracted results: {}'.format(reldif))
    
    if n+1 in prog:
        print('{0:.0f}% evaluation has been completed.'.format(np.float(n+1)/n_dif*100))
    
    # plot
    if n in sel4print:
        print('Difference between the calculated and emulated values: {}'.format(dif_fun))
        print('Difference between the calculated and emulated gradients: min ({}), med ({}), max ({})\n'.format(dif_grad.min(),np.median(dif_grad),dif_grad.max()))
        plt.clf()
        ax=axes.flat[0]
        plt.axes(ax)
        subfig=eit.plot(dll_xact,ax=ax)
        plt.title('Calculated Gradient')
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        fig.colorbar(subfig, cax=cax)
        
        ax=axes.flat[1]
        plt.axes(ax)
        subfig=eit.plot(dll_emul,ax=ax)
        plt.title('Emulated Gradient')
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        fig.colorbar(subfig, cax=cax)
        plt.draw()
        plt.pause(1.0/10.0)
    
print('Time used to calculate vs emulate gradients: {} vs {}'.format(*t_used.tolist()))
# save to file
import pandas as pd
folder='./train_NN/DNN/summary'
if not os.path.exists(folder): os.makedirs(folder)
file=os.path.join(folder,'dif-'+ctime+'.txt')
np.savetxt(file,dif)
con_str=np.array2string(np.array(node_sizes),separator=',').replace('[','').replace(']','') if 'node_sizes' in locals() or 'node_sizes' in globals() else str(depth)
act_str=','.join([val.__name__ if type(val).__name__=='function' else val.name if callable(val) else val for val in activations.values()])
dif_fun_sumry=[dif[:,0].min(),np.median(dif[:,0]),dif[:,0].max()]
dif_fun_str=np.array2string(np.array(dif_fun_sumry),precision=2,separator=',').replace('[','').replace(']','') # formatter={'float': '{: 0.2e}'.format}
dif_grad_sumry=[dif[:,1].min(),np.median(dif[:,1]),dif[:,1].max()]
dif_grad_str=np.array2string(np.array(dif_grad_sumry),precision=2,separator=',').replace('[','').replace(']','')
sumry_header=('Time','depth/node_sizes','activations','droprate','dif_fun (min,med,max)','dif_grad (min,med,max)')
sumry_np=np.array([ctime,con_str,act_str,droprate,dif_fun_str,dif_grad_str])
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


# load map
with open(os.path.join('./result',str(eit.gdim)+'d_EIT_MAP_dim'+str(eit.dim)+'.pckl'),'rb') as f:
    u=pickle.load(f)[0]
u+=eit._pts2sim(eit.sample())
# calculate gradient
dll_xact = eit.get_geom(u,[0,1],force_posperm=True)[1]
# emulate gradient
dll_emul = dnn.gradient(eit._sim2pts(u)[None,:], logLik)

# plot
plt.rcParams['image.cmap'] = 'jet'
fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=False,figsize=(12,5))
sub_figs=[None]*2

plt.axes(axes.flat[0])
sub_figs[0]=eit.plot(dll_xact,ax=axes.flat[0])
axes.flat[0].axis('equal')
axes.flat[0].set_title(r'Calculated Gradient')
plt.axes(axes.flat[1])
sub_figs[1]=eit.plot(dll_emul,ax=axes.flat[1])
axes.flat[1].axis('equal')
axes.flat[1].set_title(r'Emulated Gradient')
# add common colorbar
from util.common_colorbar import common_colorbar
fig=common_colorbar(fig,axes,sub_figs)
plt.subplots_adjust(wspace=0.2, hspace=0)
# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(folder,f_name+'.png'),bbox_inches='tight')
# plt.show()