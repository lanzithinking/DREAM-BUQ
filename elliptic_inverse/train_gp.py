"""
This is to test GP in emulating (extracted) gradients compared with those exactly calculated.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import gpflow as gpf
import sys,os
sys.path.append( '../' )
from Elliptic import Elliptic
sys.path.append( '../gp')
from multiGP import multiGP as GP

# tf.compat.v1.disable_eager_execution() # needed to train with custom loss # comment to plot
# set random seed
np.random.seed(2020)
tf.random.set_seed(2020)

# define the inverse problem
nx=40; ny=40
SNR=50
elliptic = Elliptic(nx=nx,ny=ny,SNR=SNR)
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1
whiten=False

# define the emulator (GP)
# load data
ensbl_sz = 500
n_train = 1000
folder = './train_NN'
ifwhiten='_whitened' if whiten else ''
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY'+ifwhiten+'.npz'))
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
kernel=gpf.kernels.SquaredExponential() + gpf.kernels.Linear()
# kernel=gpf.kernels.SquaredExponential(lengthscales=np.random.rand(x_train.shape[1])) + gpf.kernels.Linear()
# kernel=gpf.kernels.Matern32()
# kernel=gpf.kernels.Matern52(lengthscales=np.random.rand(x_train.shape[1]))
gp=GP(x_train.shape[1], y_train.shape[1], latent_dim=latent_dim,
      kernel=kernel)
loglik = lambda y: -0.5*elliptic.misfit.prec*tf.math.reduce_sum((y-elliptic.misfit.obs)**2,axis=1)
# folder='./saved_model'
f_name='gp_'+algs[alg_no]+str(ensbl_sz)+'-'+ifwhiten
try:
    gp.model=tf.saved_model.load(os.path.join(folder,f_name))
    print(f_name+' has been loaded!')
except Exception as err:
    print(err)
    print('Train GP model...\n')
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
    save_dir='./result/GP'+f_name
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    gp.save(save_dir)

# select some gradients to evaluate and compare
logLik = lambda x: loglik(gp.evaluate(x))
import timeit
t_used = np.zeros(2)
import matplotlib.pyplot as plt
fig,axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12,6), facecolor='white')
plt.ion()
# plt.show(block=True)
u_f = df.Function(elliptic.pde.V)
for n in range(20):
    u=elliptic.prior.sample()
    # calculate gradient
    t_start=timeit.default_timer()
    ll_xact,dll_xact = elliptic.get_geom(u,[0,1],whiten)[:2]
    t_used[0] += timeit.default_timer()-t_start
    # emulate gradient
    t_start=timeit.default_timer()
    ll_emul = logLik(u.get_local()[None,:]).numpy()
    dll_emul = gp.gradient(u.get_local()[None,:], logLik) #* grad_scalfctr
    t_used[1] += timeit.default_timer()-t_start
    # test difference
    dif_fun = np.abs(ll_xact - ll_emul)
    dif_grad = dll_xact.get_local() - dll_emul
    print('Difference between the calculated and emulated values: {}'.format(dif_fun))
    print('Difference between the calculated and emulated gradients: min ({}), med ({}), max ({})\n'.format(dif_grad.min(),np.median(dif_grad),dif_grad.max()))
    
#     # check the gradient extracted from emulation
#     v=elliptic.prior.sample()
#     h=1e-4
#     dll_emul_fd_v=(logLik(u.get_local()[None,:]+h*v.get_local()[None,:])-logLik(u.get_local()[None,:]))/h
#     reldif = abs(dll_emul_fd_v - dll_emul.flatten().dot(v.get_local()))/v.norm('l2')
#     print('Relative difference between finite difference and extracted results: {}'.format(reldif))
    
    # plot
    plt.clf()
    ax=axes.flat[0]
    plt.axes(ax)
    u_f.vector().set_local(dll_xact)
    subfig=df.plot(u_f)
    plt.title('Calculated Gradient')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    fig.colorbar(subfig, cax=cax)
    
    ax=axes.flat[1]
    plt.axes(ax)
    u_f.vector().set_local(dll_emul)
    subfig=df.plot(u_f)
    plt.title('Emulated Gradient')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    fig.colorbar(subfig, cax=cax)
    plt.draw()
    plt.pause(1.0/10.0)
    
print('Time used to calculate vs emulate gradients: {} vs {}'.format(*t_used.tolist()))


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
# u=elliptic.prior.sample()
# logLik = lambda x: loglik(gp.model(x))
# calculate gradient
dll_xact = elliptic.get_geom(u,[0,1],whiten)[1]
# emulate gradient
dll_emul = gp.gradient(u.get_local()[None,:], logLik)

# plot
plt.rcParams['image.cmap'] = 'jet'
fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(12,6))
sub_figs=[None]*2
# plot
plt.axes(axes.flat[0])
u_f.vector().set_local(dll_xact)
sub_figs[0]=df.plot(u_f)
plt.title('Calculated Gradient')
plt.axes(axes.flat[1])
u_f.vector().set_local(dll_emul)
sub_figs[1]=df.plot(u_f)
plt.title('Emulated Gradient')
# add common colorbar
from util.common_colorbar import common_colorbar
fig=common_colorbar(fig,axes,sub_figs)

# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(folder,f_name+'.png'),bbox_inches='tight')
# plt.show()