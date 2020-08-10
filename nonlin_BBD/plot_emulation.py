"""
This is to plot emulation by DNN vs GP.
"""

import numpy as np
import tensorflow as tf
import gpflow as gpf
import sys,os,pickle
sys.path.append( '../' )
sys.path.append( '../gp')
from BBD import BBD
from nn.dnn import DNN
from multiGP import multiGP as GP
from tensorflow.keras.models import load_model

# tf.compat.v1.disable_eager_execution() # needed to train with custom loss # comment to plot
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
loglik = lambda y: -0.5*tf.math.reduce_sum((y-bbd.y[None,:])**2/bbd.nz_var[None,:],axis=1)
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1
ensbl_sz = 100

# define the emulator (DNN)
# load data
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


folder = './analysis'
# define DNN
depth=5
node_sizes=[bbd.input_dim,8,16,32,64,bbd.output_dim]
activations={'hidden':'softplus','output':'linear'}
droprate=0.
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
# optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001)
dnn=DNN(x_train.shape[1], y_train.shape[1], depth=depth, node_sizes=node_sizes, droprate=droprate,
        activations=activations, optimizer=optimizer)
f_name='dnn_'+algs[alg_no]+str(ensbl_sz)
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
    dnn.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
    # save DNN
# #     dnn.model.save(os.path.join(folder,f_name+'.h5'))
# #     dnn.save(folder,f_name) # fails due to the custom kernel_initializer
#     dnn.model.save_weights(os.path.join(folder,f_name+'.h5'))

# define the emulator (GP)
n_train = 1000
prng=np.random.RandomState(2020)
sel4train = prng.choice(num_samp,size=n_train,replace=False)
tr_idx=np.random.choice(sel4train,size=np.floor(.75*n_train).astype('int'),replace=False)
te_idx=np.setdiff1d(sel4train,tr_idx)
x_train,x_test=X[tr_idx],X[te_idx]
y_train,y_test=Y[tr_idx],Y[te_idx]

# define GP
latent_dim=y_train.shape[1]
kernel=gpf.kernels.SquaredExponential(lengthscales=np.random.rand(x_train.shape[1])) + gpf.kernels.Linear()
gp=GP(x_train.shape[1], y_train.shape[1], latent_dim=latent_dim,
      kernel=kernel)
f_name='gp_'+algs[alg_no]+str(ensbl_sz)
try:
    gp.model=tf.saved_model.load(os.path.join(folder,f_name))
    gp.evaluate=lambda x:gp.model.predict(x)[0] # cannot take gradient!
    print(f_name+' has been loaded!')
except Exception as err:
    print(err)
    print('Train GP model...\n')
#     gp.induce_num=np.min((np.ceil(.1*x_train.shape[1]).astype('int'),ensbl_sz))
    epochs=200
#     batch_size=128
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    kwargs={'maxiter':epochs}
#     kwargs={'epochs':epochs,'batch_size':batch_size,'optimizer':optimizer}
    gp.train(x_train,y_train,x_test=x_test,y_test=y_test,**kwargs)
    # save GP
#     save_dir=folder+'/'+f_name
#     if not os.path.exists(save_dir): os.makedirs(save_dir)
#     gp.save(save_dir)

# select some gradients to evaluate and compare
logLik_d = lambda x: loglik(dnn.model(x))
logLik_g = lambda x: loglik(gp.evaluate(x))

# plot
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'
fig,axes = plt.subplots(nrows=1,ncols=3,sharex=True,sharey=False,figsize=(15,5))
sub_figs=[None]*3

dim=[0,1]
levels=20
grad=True
plt.axes(axes.flat[0])
sub_figs[0]=bbd.plot_2dcontour(dim=dim,type='likelihood',ax=axes.flat[0],grad=grad)
plt.title('Calculated')
plt.axes(axes.flat[1])
x=np.linspace(bbd.true_input[dim[0]]-2.,bbd.true_input[dim[0]]+2.)
y=np.linspace(bbd.true_input[dim[1]]-2.,bbd.true_input[dim[1]]+2.)
X,Y=np.meshgrid(x,y)
Input=np.zeros((X.size,bbd.input_dim))
Input[:,dim[0]],Input[:,dim[1]]=X.flatten(),Y.flatten()
Z_d=logLik_d(Input).numpy().reshape(X.shape)
Z_g=logLik_g(Input).numpy().reshape(X.shape)
if grad:
    x=np.linspace(bbd.true_input[dim[0]]-2.,bbd.true_input[dim[0]]+2.,10)
    y=np.linspace(bbd.true_input[dim[1]]-2.,bbd.true_input[dim[1]]+2.,10)
    X_,Y_=np.meshgrid(x,y)
    Input=np.zeros((X_.size,bbd.input_dim))
    Input[:,dim[0]],Input[:,dim[1]]=X_.flatten(),Y_.flatten()
    G=dnn.gradient(Input, logLik_d)
    U_d,V_d=G[:,dim[0]].reshape(X_.shape),G[:,dim[1]].reshape(X_.shape)
    G=gp.gradient(Input, logLik_g)
    U_g,V_g=G[:,dim[0]].reshape(X_.shape),G[:,dim[1]].reshape(X_.shape)
sub_figs[1]=axes.flat[1].contourf(X,Y,Z_d,levels)
axes.flat[1].set_xlabel('$u_{}$'.format(dim[0]+1))
axes.flat[1].set_ylabel('$u_{}$'.format(dim[1]+1),rotation=0)
if grad: axes.flat[1].quiver(X_,Y_,U_d,V_d)
plt.title('Emulated (CNN)')
sub_figs[2]=axes.flat[2].contourf(X,Y,Z_g,levels)
axes.flat[2].set_xlabel('$u_{}$'.format(dim[0]+1))
axes.flat[2].set_ylabel('$u_{}$'.format(dim[1]+1),rotation=0)
if grad: axes.flat[2].quiver(X_,Y_,U_g,V_g)
plt.title('Emulated (GP)')
from util.common_colorbar import common_colorbar
fig=common_colorbar(fig,axes,sub_figs)
plt.subplots_adjust(wspace=0.2, hspace=0)
# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(folder,'emulation.png'),bbox_inches='tight')
# plt.show()