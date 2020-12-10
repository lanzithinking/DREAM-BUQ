"""
This is to test cnnlstm in emulating (extracted) gradients compared with those exactly calculated.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import sys,os,pickle
sys.path.append( '../' )
from advdiff import advdiff
from nn.cnn_lstm import CNN_LSTM
from tensorflow.keras.models import load_model

# tf.compat.v1.disable_eager_execution() # needed to train with custom loss # comment to plot
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
#meshsz = (51,51)
#rel_noise = .5
#nref = 1
n_time = 16
n_spatio = 80
#adif = advdiff(mesh=meshsz, rel_noise=rel_noise, nref=nref, seed=seed)
adif.prior.V=adif.prior.Vh
adif.misfit.obs=np.array([dat.get_local() for dat in adif.misfit.d.data]).flatten()
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1

# define the emulator (DNN)
# load data
ensbl_sz = 500
folder = './train_NN'
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XimgY.npz'))
X=loaded['X']
Y=loaded['Y']
# pre-processing: scale X to 0-1
# X-=np.nanmin(X,axis=(1,2),keepdims=True) # try axis=(1,2,3)
# X/=np.nanmax(X,axis=(1,2),keepdims=True)
X=X[:,:,:,None]
# split train/test
num_samp=X.shape[0]
# n_tr=np.int(num_samp*.75)
# x_train,y_train=X[:n_tr],Y[:n_tr]
# x_test,y_test=X[n_tr:],Y[n_tr:]
tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
x_train,x_test=X[tr_idx],X[te_idx]
y_train,y_test=Y[tr_idx],Y[te_idx]
y_train = (y_train).reshape((-1, n_time, n_spatio))
y_test = (y_test).reshape((-1, n_time, n_spatio))

# define cnnlstm
num_filters=[16,8,8] # best for non-whiten [16,8,8]
activations={'conv':'softplus','latent':'softmax','output':'linear','lstm':'tanh'} # best for non-whiten
# activations={'conv':tf.keras.layers.LeakyReLU(alpha=0.1),'latent':tf.keras.layers.PReLU(),'output':'linear'}
# activations={'conv':'relu','latent':tf.math.sin,'output':tf.keras.layers.PReLU()}
latent_dim=1024 # best for non-whiten 256
droprate=0.25 # best for non-whiten .5
sin_init=lambda n:tf.random_uniform_initializer(minval=-tf.math.sqrt(6/n), maxval=tf.math.sqrt(6/n))
kernel_initializers={'conv':'he_uniform','latent':sin_init,'output':'glorot_uniform'}
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
# optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001)
cnnlstm=CNN_LSTM(X.shape[1:], Y.shape[1], n_time=n_time, n_spatio=n_spatio, num_filters=num_filters, latent_dim=latent_dim, droprate=droprate,
        activations=activations, optimizer=optimizer)

loglik = lambda y: -0.5*tf.math.reduce_sum((y-adif.misfit.obs.reshape((16,80)))**2/adif.misfit.noise_variance,axis=[1,2])
# custom_loss = lambda y_true, y_pred: [tf.square(loglik(y_true)-loglik(y_pred)), (y_true-y_pred)/adif.misfit.noise_variance]
# dnn=DNN(x_train.shape[1], y_train.shape[1], depth=depth, droprate=droprate,
#         activations=activations, kernel_initializers=kernel_initializers, optimizer=optimizer, loss=custom_loss)
folder='./train_NN/'
if not os.path.exists(folder): os.makedirs(folder)
import time
ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
f_name='cnn_lstm_'+algs[alg_no]+str(ensbl_sz)+ctime
keywd = 'cnn_lstm_' + algs[alg_no] + str(ensbl_sz)
try:
#     cnn.model=load_model(os.path.join(folder,f_name+'.h5'))
#     cnn.model=load_model(os.path.join(folder,f_name+'.h5'),custom_objects={'loss':None})
    cnnlstm.model.load_weights(os.path.join(folder,f_name+'.h5'))
    print(f_name+' has been loaded!')
except Exception as err:
    print(err)
    print('Train CNN...\n')
    epochs=200
    patience=0
    import timeit
    t_start=timeit.default_timer()
    cnnlstm.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training cnnlstm: {}'.format(t_used))
    # save cnnlstm
#     cnnlstm.model.save(os.path.join(folder,f_name+'.h5'))
#     cnnlstm.save(folder,f_name)
    cnnlstm.model.save_weights(os.path.join(folder,f_name+'.h5'))

# select some gradients to evaluate and compare
logLik = lambda x: loglik(cnnlstm.model(x))
#logLik = lambda x: loglik(np.reshape(cnnlstm.model(x),(-1,16*80)))
import timeit
t_used = np.zeros(2)
import matplotlib.pyplot as plt
fig,axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12,6), facecolor='white')
plt.ion()
n_dif = 100
dif = np.zeros((n_dif,2))
loaded=np.load(file=os.path.join('./train_NN',algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XimgY'+'.npz'))
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
    ll_xact,dll_xact = adif.get_geom(adif.img2vec(u,adif.prior.V if eldeg>1 else None),[0,1])[:2]
    t_used[0] += timeit.default_timer()-t_start
    # emulate gradient
    t_start=timeit.default_timer()
    ll_emul = logLik(u[None,:,:,None]).numpy()[0]
    dll_emul = adif.img2vec(cnnlstm.gradient(u[None,:,:,None], logLik))
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
folder='./train_NN/'
if not os.path.exists(folder): os.makedirs(folder)
file=os.path.join(folder,'dif-'+ctime+'.txt')
np.savetxt(file,dif)
flt_str=np.array2string(np.array(num_filters),separator=',').replace('[','').replace(']','')
act_str=','.join([val.__name__ if type(val).__name__=='function' else val.name if callable(val) else val for val in activations.values()])
dif_fun_sumry=[dif[:,0].min(),np.median(dif[:,0]),dif[:,0].max()]
dif_fun_str=np.array2string(np.array(dif_fun_sumry),precision=2,separator=',').replace('[','').replace(']','') # formatter={'float': '{: 0.2e}'.format}
dif_grad_sumry=[dif[:,1].min(),np.median(dif[:,1]),dif[:,1].max()]
dif_grad_str=np.array2string(np.array(dif_grad_sumry),precision=2,separator=',').replace('[','').replace(']','')
sumry_header=('Time','num_filters','activations','latent_dim','droprate','dif_fun (min,med,max)','dif_grad (min,med,max)')
sumry_np=np.array([ctime,flt_str,act_str,latent_dim,droprate,dif_fun_str,dif_grad_str])
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
dll_xact = adif.get_geom(u,[0,1])[1]
# emulate gradient
dll_emul = adif.img2vec(cnnlstm.gradient(adif.vec2img(u)[None,:,:,None], logLik))

# plot
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