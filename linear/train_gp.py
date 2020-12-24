"""
This is to test GP in emulating (extracted) gradients compared with those exactly calculated.
"""

import numpy as np
import tensorflow as tf
import gpflow as gpf
import sys,os,pickle
sys.path.append( '../' )
from LiN import LiN
sys.path.append( '../gp')
from multiGP import multiGP as GP

# tf.compat.v1.disable_eager_execution() # needed to train with custom loss # comment to plot
# set random seed
np.random.seed(2020)
tf.random.set_seed(2020)

## define the linear-Gaussian inverse problem ##
# set up
d=3; m=100
try:
    with open('./result/lin.pickle','rb') as f:
        [nz_var,pr_cov,A,true_input,y]=pickle.load(f)
    print('Data loaded!\n')
    kwargs={'true_input':true_input,'A':A,'y':y}
except:
    print('No data found. Generate new data...\n')
    nz_var=.1; pr_cov=1.
    true_input=np.arange(-np.floor(d/2),np.ceil(d/2))
    A=np.random.rand(m,d)
    kwargs={'true_input':true_input,'A':A}
lin=LiN(d,m,nz_var=nz_var,pr_cov=pr_cov,**kwargs)
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1

# define the emulator (GP)
# load data
ensbl_sz = 100
n_train = 1000
folder = './train_NN'
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
X=loaded['X']
Y=loaded['Y']
# pre-processing: scale X to 0-1
# X-=np.nanmin(X,axis=(1,2),keepdims=True) # try axis=(1,2,3)
# X/=np.nanmax(X,axis=(1,2),keepdims=True)
# split train/test
num_samp=X.shape[0]
prng=np.random.RandomState(2020)
sel4train = prng.choice(num_samp,size=n_train,replace=False)
tr_idx=np.random.choice(sel4train,size=np.floor(.75*n_train).astype('int'),replace=False)
te_idx=np.setdiff1d(sel4train,tr_idx)
x_train,x_test=X[tr_idx],X[te_idx]
y_train,y_test=Y[tr_idx],Y[te_idx]

# define GP
latent_dim=y_train.shape[1]
# kernel=gpf.kernels.SquaredExponential() + gpf.kernels.Linear()
kernel=gpf.kernels.SquaredExponential(lengthscales=np.random.rand(x_train.shape[1])) + gpf.kernels.Linear()
# kernel=gpf.kernels.Matern32()
# kernel=gpf.kernels.Matern52(lengthscales=np.random.rand(x_train.shape[1]))
gp=GP(x_train.shape[1], y_train.shape[1], latent_dim=latent_dim,
      kernel=kernel)
loglik = lambda y: -0.5*tf.math.reduce_sum((y-lin.y[None,:])**2/lin.nz_var[None,:],axis=1)
folder='./train_NN/GP/saved_model'
if not os.path.exists(folder): os.makedirs(folder)
import time
ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
f_name='gp_'+algs[alg_no]+str(ensbl_sz)+'-'+ctime
try:
    gp.model=tf.saved_model.load(os.path.join(folder,f_name))
    gp.evaluate=lambda x:gp.model.predict(x)[0] # cannot take gradient!
    print(f_name+' has been loaded!')
except Exception as err:
    print(err)
    print('Train GP model...\n')
#     gp.induce_num=np.min((np.ceil(.1*x_train.shape[1]).astype('int'),ensbl_sz))
    epochs=100
    batch_size=128
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    kwargs={'maxiter':epochs}
#     kwargs={'epochs':epochs,'batch_size':batch_size,'optimizer':optimizer}
    import timeit
    t_start=timeit.default_timer()
    gp.train(x_train,y_train,x_test=x_test,y_test=y_test,**kwargs)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training GP: {}'.format(t_used))
    # save GP
    save_dir=folder+'/'+f_name
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    gp.save(save_dir)

# select some gradients to evaluate and compare
logLik = lambda x: loglik(gp.evaluate(x))
import timeit
t_used = np.zeros(2)
n_dif = 100
dif = np.zeros((n_dif,2))
loaded=np.load(file=os.path.join('./train_NN',algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
prng=np.random.RandomState(2020)
sel4eval = prng.choice(num_samp,size=n_dif,replace=False)
X=loaded['X'][sel4eval]; Y=loaded['Y'][sel4eval]
sel4print = prng.choice(n_dif,size=10,replace=False)
prog=np.ceil(n_dif*(.1+np.arange(0,1,.1)))
for n in range(n_dif):
    u=X[n]
    # calculate gradient
    t_start=timeit.default_timer()
    ll_xact,dll_xact = lin.get_geom(u,[0,1])[:2]
    t_used[0] += timeit.default_timer()-t_start
    # emulate gradient
    t_start=timeit.default_timer()
    ll_emul = logLik(u[None,:]).numpy()
    dll_emul = gp.gradient(u[None,:], logLik)
    t_used[1] += timeit.default_timer()-t_start
    # test difference
    dif_fun = np.abs(ll_xact - ll_emul)
    dif_grad = dll_xact - dll_emul
    dif[n] = np.array([dif_fun, np.linalg.norm(dif_grad)/np.linalg.norm(dll_xact)])
    
#     # check the gradient extracted from emulation
#     v=lin.sample()
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
    
print('Time used to calculate vs emulate gradients: {} vs {}'.format(*t_used.tolist()))
# save to file
import pandas as pd
folder='./train_NN/GP/summary'
if not os.path.exists(folder): os.makedirs(folder)
file=os.path.join(folder,'dif-'+ctime+'.txt')
np.savetxt(file,dif)
ker_str='+'.join([ker.name for ker in kernel.kernels]) if kernel.name=='sum' else kernel.name
dif_fun_sumry=[dif[:,0].min(),np.median(dif[:,0]),dif[:,0].max()]
dif_fun_str=np.array2string(np.array(dif_fun_sumry),precision=2,separator=',').replace('[','').replace(']','') # formatter={'float': '{: 0.2e}'.format}
dif_grad_sumry=[dif[:,1].min(),np.median(dif[:,1]),dif[:,1].max()]
dif_grad_str=np.array2string(np.array(dif_grad_sumry),precision=2,separator=',').replace('[','').replace(']','')
sumry_header=('Time','train_size','latent_dim','kernel','dif_fun (min,med,max)','dif_grad (min,med,max)')
sumry_np=np.array([ctime,n_train,latent_dim,ker_str,dif_fun_str,dif_grad_str])
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


# plot
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'
fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=False,figsize=(12,5))
sub_figs=[None]*2

dim=[0,1]
levels=20
grad=True
plt.axes(axes.flat[0])
sub_figs[0]=lin.plot_2dcontour(dim=dim,type='likelihood',ax=axes.flat[0],grad=grad)
plt.title('Calculated')
plt.axes(axes.flat[1])
x=np.linspace(lin.true_input[dim[0]]-2.,lin.true_input[dim[0]]+2.)
y=np.linspace(lin.true_input[dim[1]]-2.,lin.true_input[dim[1]]+2.)
X,Y=np.meshgrid(x,y)
# Input=np.zeros((X.size,lin.input_dim))
Input=np.tile(lin.true_input,(X.size,1))
Input[:,dim[0]],Input[:,dim[1]]=X.flatten(),Y.flatten()
Z=logLik(Input).numpy().reshape(X.shape)
if grad:
    x=np.linspace(lin.true_input[dim[0]]-2.,lin.true_input[dim[0]]+2.,10)
    y=np.linspace(lin.true_input[dim[1]]-2.,lin.true_input[dim[1]]+2.,10)
    X_,Y_=np.meshgrid(x,y)
#     Input=np.zeros((X_.size,lin.input_dim))
    Input=np.tile(lin.true_input,(X_.size,1))
    Input[:,dim[0]],Input[:,dim[1]]=X_.flatten(),Y_.flatten()
    G=gp.gradient(Input, logLik)
    U,V=G[:,dim[0]].reshape(X_.shape),G[:,dim[1]].reshape(X_.shape)
sub_figs[1]=axes.flat[1].contourf(X,Y,Z,levels)
axes.flat[1].set_xlabel('$u_{}$'.format(dim[0]+1))
axes.flat[1].set_ylabel('$u_{}$'.format(dim[1]+1),rotation=0)
if grad: axes.flat[1].quiver(X_,Y_,U,V)
plt.title('Emulated')
from util.common_colorbar import common_colorbar
fig=common_colorbar(fig,axes,sub_figs)
plt.subplots_adjust(wspace=0.2, hspace=0)
# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(folder,f_name+'.png'),bbox_inches='tight')
# plt.show()