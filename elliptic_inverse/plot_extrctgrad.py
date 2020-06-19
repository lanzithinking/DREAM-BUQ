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
folder = './train_NN'
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_CNN.npz'))
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
# activations={'conv':tf.keras.layers.LeakyReLU(alpha=0.1),'latent':tf.keras.layers.PReLU(),'output':'linear'}
latent_dim=128
droprate=.5
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
# optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001)
loglik = lambda y: -0.5*elliptic.misfit.prec*tf.math.reduce_sum((y-elliptic.misfit.obs)**2,axis=1)
# custom_loss = lambda y_true, y_pred: [tf.square(loglik(y_true)-loglik(y_pred)), elliptic.misfit.prec*(y_true-y_pred)]
cnn=CNN(x_train.shape[1:], y_train.shape[1], num_filters=num_filters, latent_dim=latent_dim,
        activations=activations, droprate=droprate, optimizer=optimizer, padding='same')
# cnn=CNN(x_train.shape[1:], y_train.shape[1], num_filters=num_filters, latent_dim=latent_dim,
#         activations=activations, droprate=droprate, optimizer=optimizer, loss=custom_loss)
# folder=folder+'/saved_model'
f_name='cnn_'+algs[alg_no]+str(ensbl_sz)+'.h5'
try:
    cnn.model=load_model(os.path.join(folder,f_name),custom_objects={'loss':None})
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
    cnn.model.save(os.path.join(folder,f_name))
#     cnn.save(folder,'cnn_'+algs[alg_no]+str(ensbl_sz))


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
logLik = lambda x: loglik(cnn.model(x))
# calculate gradient
dll_xact = elliptic.get_geom(u,[0,1])[1]
# emulate gradient
u_img=fun2img(vec2fun(u,elliptic.pde.V))
dll_emul = cnn.gradient(u_img[None,:,:,None], logLik)

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
u_f=img2fun(dll_emul,elliptic.pde.V)
sub_figs[1]=df.plot(u_f)
plt.title('Emulated Gradient')
# add common colorbar
from util.common_colorbar import common_colorbar
fig=common_colorbar(fig,axes,sub_figs)

# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(folder,'extrctgrad.png'),bbox_inches='tight')
# plt.show()