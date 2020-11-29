"""
This is to test GP in emulating (extracted) gradients compared with those exactly calculated.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import gpflow as gpf
import sys,os,pickle
sys.path.append( '../' )
from advdiff import advdiff
sys.path.append( '../gp')
from multiGP import multiGP as GP

# tf.compat.v1.disable_eager_execution() # needed to train with custom loss # comment to plot
# set random seed
seed=2020
np.random.seed(seed)
tf.random.set_seed(seed)

## define Advection-Diffusion inverse problem ##
# mesh = dl.Mesh('ad_10k.xml')
mesh = (51,51)
kappa = 1e-3
rel_noise = .5
nref = 1
adif = advdiff(mesh=mesh, kappa=kappa, rel_noise=rel_noise, nref=nref, seed=seed)
adif.prior.V=adif.prior.Vh
adif.misfit.obs=np.array([dat.get_local() for dat in adif.misfit.d.data]).flatten()
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1

# define the emulator (GP)
# load data
ensbl_sz = 500
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
kernel=gpf.kernels.SquaredExponential() + gpf.kernels.Linear()
# kernel=gpf.kernels.SquaredExponential(lengthscales=np.random.rand(x_train.shape[1])) + gpf.kernels.Linear()
# kernel=gpf.kernels.Matern32()
# kernel=gpf.kernels.Matern52(lengthscales=np.random.rand(x_train.shape[1]))
gp=GP(x_train.shape[1], y_train.shape[1], latent_dim=latent_dim,
      kernel=kernel)
loglik = lambda y: -0.5*tf.math.reduce_sum((y-adif.misfit.obs)**2/adif.misfit.noise_variance,axis=1)
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
import matplotlib.pyplot as plt
fig,axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12,6), facecolor='white')
plt.ion()
# plt.show(block=True)
n_dif = 100
dif = np.zeros((n_dif,2))
loaded=np.load(file=os.path.join('./train_NN',algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
prng=np.random.RandomState(2020)
sel4eval = prng.choice(num_samp,size=n_dif,replace=False)
X=loaded['X'][sel4eval]; Y=loaded['Y'][sel4eval]
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
for n in range(n_dif):
    u=X[n]
    # calculate gradient
    t_start=timeit.default_timer()
    u_f1.vector().set_local(u); u_v = u_f1.vector() # u already in dof order
    if eldeg>1:
        u_f.interpolate(u_f1)
        u_v = u_f.vector()
#     u_f = img2fun(u, adif.prior.V); u_v = u_f.vector() # for u in vertex order
    ll_xact,dll_xact = eit.get_geom(u,[0,1])[:2]
    t_used[0] += timeit.default_timer()-t_start
    # emulate gradient
    t_start=timeit.default_timer()
    ll_emul = logLik(u[None,:]).numpy()
    dll_emul = gp.gradient(u[None,:], logLik)
    t_used[1] += timeit.default_timer()-t_start
    # test difference
    dif_fun = np.abs(ll_xact - ll_emul)
    if eldeg>1:
        u_f.vector().set_local(dll_xact)
        dll_xact = u_f.compute_vertex_values(adif.mesh)[d2v] # covert to dof order
    else:
        dll_xact = dll_xact.get_local()
    dif_grad = dll_xact - dll_emul
    dif[n] = np.array([dif_fun, np.linalg.norm(dif_grad)/np.linalg.norm(dll_xact)])
    
#     # check the gradient extracted from emulation
#     v=adif.prior.sample()
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
        u_f1.vector().set_local(dll_xact)
        subfig=df.plot(u_f1)
        plt.title('Calculated Gradient')
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        fig.colorbar(subfig, cax=cax)
        
        ax=axes.flat[1]
        plt.axes(ax)
        u_f1.vector().set_local(dll_emul)
        subfig=df.plot(u_f1)
        plt.title('Emulated Gradient')
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        fig.colorbar(subfig, cax=cax)
        plt.draw()
        plt.pause(1.0/10.0)
    
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


# load map
MAP_file=os.path.join(os.getcwd(),'results/MAP.xdmf')
u_f=df.Function(adif.prior.V, name='MAP')
if os.path.isfile(MAP_file):
    f=df.XDMFFile(adif.mpi_comm,MAP_file)
    f.read_checkpoint(u_f,'m',0)
    f.close()
    u = u_f.vector()
else:
    u=adif.get_MAP(SAVE=True)
# calculate gradient
dll_xact = eit.get_geom(u,[0,1])[1]
# emulate gradient
dll_emul = gp.gradient(u_f.compute_vertex_values(adif.mesh)[d2v][None,:] if eldeg>1 else u.get_local()[None,:], logLik)

# plot
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'
fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=False,figsize=(12,5))
sub_figs=[None]*2

plt.axes(axes.flat[0])
u_f.vector().set_local(dll_xact)
sub_figs[0]=df.plot(u_f)
axes.flat[0].axis('equal')
axes.flat[0].set_title(r'Calculated Gradient')
plt.axes(axes.flat[1])
u_f1.vector().set_local(dll_emul)
sub_figs[1]=df.plot(u_f1)
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