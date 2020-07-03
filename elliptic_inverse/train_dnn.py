"""
This is to train DNN to emulate (extracted) gradients compared with those exactly calculated.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import sys,os
sys.path.append( "../" )
from Elliptic import Elliptic
from nn.dnn import DNN
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

# define the emulator (DNN)
# load data
ensbl_sz = 500
folder = './train_NN'
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
X=loaded['X']
Y=loaded['Y']
# pre-processing: scale X to 0-1
# X-=np.nanmin(X,axis=1,keepdims=True)
# X/=np.nanmax(X,axis=1,keepdims=True)
# split train/test
num_samp=X.shape[0]
n_tr=np.int(num_samp*.75)
x_train,y_train=X[:n_tr],Y[:n_tr]
x_test,y_test=X[n_tr:],Y[n_tr:]

# define DNN
depth=3
# node_sizes=[elliptic.prior.dim,1024,512,128,len(elliptic.misfit.idx)]
# activation='linear'
# activation=tf.keras.layers.LeakyReLU(alpha=0.1)
# activations={'hidden':tf.keras.layers.LeakyReLU(alpha=0.01),'output':'linear'}
activations={'hidden':tf.math.sin,'output':'linear'}
droprate=.4
sin_init=lambda n:tf.random_uniform_initializer(minval=-tf.math.sqrt(6/n), maxval=tf.math.sqrt(6/n))
kernel_initializers={'hidden':sin_init,'output':'he_uniform'}
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
# optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001)
dnn=DNN(x_train.shape[1], y_train.shape[1], depth=depth, droprate=droprate,
        activations=activations, kernel_initializers=kernel_initializers, optimizer=optimizer)
loglik = lambda y: -0.5*elliptic.misfit.prec*tf.math.reduce_sum((y-elliptic.misfit.obs)**2,axis=1)
# custom_loss = lambda y_true, y_pred: [tf.square(loglik(y_true)-loglik(y_pred)), elliptic.misfit.prec*(y_true-y_pred)]
# dnn=DNN(x_train.shape[1], y_train.shape[1], depth=depth, droprate=droprate,
#         activations=activations, kernel_initializers=kernel_initializers, optimizer=optimizer, loss=custom_loss)
# folder=folder+'/saved_model'
f_name='dnn_'+algs[alg_no]+str(ensbl_sz)#+'_customloss'
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

# some more test
logLik = lambda x: loglik(dnn.model(x))
import timeit
t_used = np.zeros(2)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,6), facecolor='white')
plt.ion()
# plt.show(block=True)
u_f = df.Function(elliptic.pde.V)
for n in range(20):
    u=elliptic.prior.sample()
    # calculate gradient
    t_start=timeit.default_timer()
    dll_xact = elliptic.get_geom(u,[0,1])[1]
    t_used[0] += timeit.default_timer()-t_start
    # emulate gradient
    t_start=timeit.default_timer()
    dll_emul = dnn.gradient(u.get_local()[None,:], logLik)
    t_used[1] += timeit.default_timer()-t_start
    # test difference
    dif = dll_xact.get_local() - dll_emul
    print('Difference between the calculated and emulated gradients: min ({}), med ({}), max ({})'.format(dif.min(),np.median(dif),dif.max()))
    
    # check the gradient extracted from emulation
    v=elliptic.prior.sample()
    h=1e-3
    dll_emul_fd_v=(logLik(u.get_local()[None,:]+h*v.get_local()[None,:])-logLik(u.get_local()[None,:]))/h
    reldif = abs(dll_emul_fd_v - dll_emul.flatten().dot(v.get_local()))/v.norm('l2')
    print('Relative difference between finite difference and extracted results: {}'.format(reldif))
    
    # plot
    plt.subplot(121)
    u_f.vector().set_local(dll_xact)
    df.plot(u_f)
    plt.title('Calculated Gradient')
    plt.subplot(122)
    u_f.vector().set_local(dll_emul)
    df.plot(u_f)
    plt.title('Emulated Gradient')
    plt.draw()
    plt.pause(1.0/10.0)
    
print('Time used to calculate vs emulate gradients: {} vs {}'.format(*t_used.tolist()))


# read data and construct plot functions
u_f = df.Function(elliptic.pde.V)
# read MAP
try:
    f=df.HDF5File(elliptic.pde.mpi_comm, os.path.join('./result',"MAP_SNR"+str(SNR)+".h5"), "r")
    f.read(u_f,"parameter")
    f.close()
except:
    pass
u=u_f.vector()
# u=elliptic.prior.sample()
logLik = lambda x: loglik(dnn.model(x))
# calculate gradient
dll_xact = elliptic.get_geom(u,[0,1])[1]
# emulate gradient
dll_emul = dnn.gradient(u.get_local()[None,:], logLik)
 
# plot
import matplotlib.pyplot as plt
import matplotlib as mp
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