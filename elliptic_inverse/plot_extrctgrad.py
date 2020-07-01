"""
This is to plot emulated (extracted) gradients compared with those exactly calculated.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import sys,os
sys.path.append( "../" )
from Elliptic import Elliptic
from util.dolfin_gadget import vec2fun,fun2img,img2fun
from nn.cnn import CNN
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
# load data
ensbl_sz = 500
folder = './analysis_f_SNR'+str(SNR)
if not os.path.exists(folder): os.makedirs(folder)

## define the emulator (CNN) ##
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
x_train,y_train=X[:n_tr],Y[:n_tr]
x_test,y_test=X[n_tr:],Y[n_tr:]

# define CNN
num_filters=[16,8]
activations={'conv':'relu','latent':tf.keras.layers.PReLU(),'output':'linear'}
latent_dim=128
droprate=.5
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
cnn=CNN(x_train.shape[1:], y_train.shape[1], num_filters=num_filters, latent_dim=latent_dim, droprate=droprate,
        activations=activations, optimizer=optimizer)
f_name='cnn_'+algs[alg_no]+str(ensbl_sz)
try:
    cnn.model=load_model(os.path.join(folder,f_name+'.h5'),custom_objects={'loss':None})
#     cnn.model.load_weights(os.path.join(folder,f_name+'.h5'))
    print(f_name+' has been loaded!')
except Exception as err:
    print(err)
    print('Train CNN...\n')
    epochs=200
    patience = 0
    import timeit
    t_start=timeit.default_timer()
    cnn.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training CNN: {}'.format(t_used))
    # save CNN
    cnn.model.save(os.path.join(folder,f_name+'.h5'))
#     cnn.model.save_weights(os.path.join(folder,f_name+'.h5'))

# define the emulator (DNN)
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

## define DNN ##
depth=3
activations={'hidden':tf.math.sin,'output':'linear'}
droprate=.4
sin_init=lambda n:tf.random_uniform_initializer(minval=-tf.math.sqrt(6/n), maxval=tf.math.sqrt(6/n))
kernel_initializers={'hidden':sin_init,'output':'he_uniform'}
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
dnn=DNN(x_train.shape[1], y_train.shape[1], depth=depth, droprate=droprate,
        activations=activations, kernel_initializers=kernel_initializers, optimizer=optimizer)
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
loglik = lambda y: -0.5*elliptic.misfit.prec*tf.math.reduce_sum((y-elliptic.misfit.obs)**2,axis=1)
loglik_cnn = lambda x: loglik(cnn.model(x))
loglik_dnn = lambda x: loglik(dnn.model(x))
# calculate gradient
dll_xact = elliptic.get_geom(u,[0,1])[1]
# emulate gradient
u_img=fun2img(vec2fun(u,elliptic.pde.V))
dll_cnn = cnn.gradient(u_img[None,:,:,None], loglik_cnn)
dll_dnn = dnn.gradient(u.get_local()[None,:], loglik_dnn)

# plot
import matplotlib.pyplot as plt
import matplotlib as mp
plt.rcParams['image.cmap'] = 'jet'
fig,axes = plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True,figsize=(15,5))
sub_figs=[None]*3
# plot
plt.axes(axes.flat[0])
u_f.vector().set_local(dll_xact)
sub_figs[0]=df.plot(u_f)
plt.title('Calculated Gradient')
plt.axes(axes.flat[1])
u_f=img2fun(dll_cnn,elliptic.pde.V)
sub_figs[1]=df.plot(u_f)
plt.title('Emulated Gradient (CNN)')
plt.axes(axes.flat[2])
u_f=vec2fun(dll_dnn,elliptic.pde.V)
sub_figs[2]=df.plot(u_f)
plt.title('Emulated Gradient (DNN)')
# add common colorbar
from util.common_colorbar import common_colorbar
fig=common_colorbar(fig,axes,sub_figs)

# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(folder,'extrctgrad.png'),bbox_inches='tight')
# plt.show()