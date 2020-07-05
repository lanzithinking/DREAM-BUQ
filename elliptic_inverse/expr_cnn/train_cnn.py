"""
This is to test CNN in emulating (extracted) gradients compared with those exactly calculated.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import sys,os
sys.path.append( '../' )
sys.path.append( '../../')
from Elliptic import Elliptic
from util.dolfin_gadget import vec2fun,fun2img,img2fun
from nn.cnn import CNN
from tensorflow.keras.models import load_model

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

# define the emulator (CNN)
# load data
ensbl_sz = 500
folder = '../train_NN'
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XimgY.npz'))
X=loaded['X']
Y=loaded['Y']
# pre-processing: scale X to 0-1
# X-=np.nanmin(X,axis=(1,2),keepdims=True) # try axis=(1,2,3)
# X/=np.nanmax(X,axis=(1,2),keepdims=True)
X=X[:,:,:,None]
# split train/test
num_samp=X.shape[0]
n_tr=np.int(num_samp*.75)
# x_train,y_train=X[:n_tr],Y[:n_tr]
# x_test,y_test=X[n_tr:],Y[n_tr:]
tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
x_train,x_test=X[tr_idx],X[te_idx]
y_train,y_test=Y[tr_idx],Y[te_idx]

# define CNN
num_filters=[16,8]
activations={'conv':'relu','latent':tf.keras.layers.PReLU(),'output':'linear'}
# activations={'conv':'linear','latent':tf.keras.layers.PReLU(),'output':'linear'}
# activations={'conv':'relu','latent':tf.math.sin,'output':tf.keras.layers.PReLU()}
latent_dim=128
droprate=.5
# sin_init=lambda n:tf.random_uniform_initializer(minval=-tf.math.sqrt(6/n), maxval=tf.math.sqrt(6/n))
# kernel_initializers={'conv':'he_uniform','latent':sin_init,'output':'glorot_uniform'}
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
# optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001)
cnn=CNN(x_train.shape[1:], y_train.shape[1], num_filters=num_filters, latent_dim=latent_dim, droprate=droprate,
        activations=activations, optimizer=optimizer)
loglik = lambda y: -0.5*elliptic.misfit.prec*tf.math.reduce_sum((y-elliptic.misfit.obs)**2,axis=1)
# custom_loss = lambda y_true, y_pred: [tf.square(loglik(y_true)-loglik(y_pred)), elliptic.misfit.prec*(y_true-y_pred)]
# cnn=CNN(x_train.shape[1:], y_train.shape[1], num_filters=num_filters, latent_dim=latent_dim, droprate=droprate,
#         activations=activations, kernel_initializers=kernel_initializers, optimizer=optimizer, loss=custom_loss)
folder='./saved_model'
import time
ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
f_name='cnn_'+algs[alg_no]+str(ensbl_sz)+ctime
try:
#     cnn.model=load_model(os.path.join(folder,f_name+'.h5'))
#     cnn.model=load_model(os.path.join(folder,f_name+'.h5'),custom_objects={'loss':None})
    cnn.model.load_weights(os.path.join(folder,f_name+'.h5'))
    print(f_name+' has been loaded!')
except Exception as err:
    print(err)
    print('Train CNN...\n')
    epochs=200
    patience=0
    import timeit
    t_start=timeit.default_timer()
    cnn.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training CNN: {}'.format(t_used))
    # save CNN
#     cnn.model.save(os.path.join(folder,f_name+'.h5'))
#     cnn.save(folder,f_name)
    cnn.model.save_weights(os.path.join(folder,f_name+'.h5'))

# select some gradients to evaluate and compare
logLik = lambda x: loglik(cnn.model(x))
import timeit
t_used = np.zeros(2)
import matplotlib.pyplot as plt
fig,axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12,6), facecolor='white')
plt.ion()
# plt.show(block=True)
u_f = df.Function(elliptic.pde.V)
n_dif = 1000
dif = np.zeros((n_dif,2))
loaded=np.load(file=os.path.join('../train_NN',algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
prng=np.random.RandomState(2020)
sel4eval = prng.choice(num_samp,size=n_dif,replace=False)
X=loaded['X'][sel4eval]; Y=loaded['Y'][sel4eval]
sel4print = prng.choice(n_dif,size=10,replace=False)
prog=np.ceil(n_dif*(.1+np.arange(0,1,.1)))
# batch_size=tf.shape(cnn.model.input)[0]
# batch_size=cnn.model.input.shape[0]
# grad_scalfctr = x_train.shape[0]/batch_size
# grad_scalfctr = np.sqrt(cnn.model.history.params['steps'])
for n in range(n_dif):
    u=elliptic.prior.gen_vector(X[n])
    # calculate gradient
    t_start=timeit.default_timer()
    ll_xact,dll_xact = elliptic.get_geom(u,[0,1])[:2]
    t_used[0] += timeit.default_timer()-t_start
    # emulate gradient
    t_start=timeit.default_timer()
    u_img=fun2img(vec2fun(u,elliptic.pde.V))
    ll_emul = logLik(u_img[None,:,:,None]).numpy()
    dll_emul = cnn.gradient(u_img[None,:,:,None], logLik) #* grad_scalfctr
    t_used[1] += timeit.default_timer()-t_start
    # test difference
    dif_fun = np.abs(ll_xact - ll_emul)
    dif_grad = dll_xact - img2fun(dll_emul,elliptic.pde.V).vector()
    dif[n] = np.array([dif_fun,dif_grad.norm('l2')/dll_xact.norm('l2')])
    
#     # check the gradient extracted from emulation
#     v=elliptic.prior.sample()
#     v_img=fun2img(vec2fun(v,elliptic.pde.V))
#     h=1e-4
#     dll_emul_fd_v=(logLik(u_img[None,:,:,None]+h*v_img[None,:,:,None])-logLik(u_img[None,:,:,None]))/h
#     reldif = abs(dll_emul_fd_v - dll_emul.flatten().dot(v_img.flatten()))/v.norm('l2')
#     print('Relative difference between finite difference and extracted results: {}'.format(reldif))
    
    if n+1 in prog:
        print('{0:.0f}% evaluation has been completed.'.format(np.float(n+1)/n_dif*100))
    
    # plot
    if n in sel4print:
        print('Difference between the calculated and emulated values: {}'.format(dif_fun))
        print('Difference between the calculated and emulated gradients: min ({}), med ({}), max ({})\n'.format(dif_grad.min(),np.median(dif_grad.get_local()),dif_grad.max()))
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
        u_f=img2fun(dll_emul,elliptic.pde.V)
        subfig=df.plot(u_f)
        plt.title('Emulated Gradient')
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        fig.colorbar(subfig, cax=cax)
        plt.draw()
        plt.pause(1.0/10.0)
    
print('Time used to calculate vs emulate gradients: {} vs {}'.format(*t_used.tolist()))
# save to file
import pandas as pd
folder='./result'
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
logLik = lambda x: loglik(cnn.model(x))
# calculate gradient
dll_xact = elliptic.get_geom(u,[0,1])[1]
# emulate gradient
u_img=fun2img(vec2fun(u,elliptic.pde.V))
dll_emul = cnn.gradient(u_img[None,:,:,None], logLik)

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
u_f=img2fun(dll_emul,elliptic.pde.V)
sub_figs[1]=df.plot(u_f)
plt.title('Emulated Gradient')
# add common colorbar
from util.common_colorbar import common_colorbar
fig=common_colorbar(fig,axes,sub_figs)

# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(folder,f_name+'.png'),bbox_inches='tight')
# plt.show()