"""
This is to plot emulated (extracted) gradients compared with those exactly calculated.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import sys,os
sys.path.append( "../" )
from advdiff import advdiff
from util.dolfin_gadget import *
from nn.cnn import CNN
from nn.cnn_rnn import CNN_RNN
from tensorflow.keras.models import load_model

# tf.compat.v1.disable_eager_execution() # needed to train with custom loss # comment to plot
# set random seed
seed=2020
np.random.seed(seed)
tf.random.set_seed(seed)

# define the inverse problem
meshsz = (61,61)
eldeg = 1
gamma = 2.; delta = 10.
rel_noise = .5
nref = 1
adif = advdiff(mesh=meshsz, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed)
adif.prior.V=adif.prior.Vh
adif.misfit.obs=np.array([dat.get_local() for dat in adif.misfit.d.data]).flatten()
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
# alg_no=1
# load data
ensbl_sz = 500
folder = './train_NN_eldeg'+str(eldeg)
savepath = './analysis_eldeg'+str(eldeg)
if not os.path.exists(savepath): os.makedirs(savepath)

## define the emulator (CNN) ##
loaded=np.load(file=os.path.join(folder,algs[1]+'_ensbl'+str(ensbl_sz)+'_training_XimgY.npz'))
X=loaded['X']
Y=loaded['Y']
# loaded=np.load(file=os.path.join(folder,algs[1]+'_ensbl'+str(ensbl_sz)+'_training_XimgY.npz'))
# X=np.vstack((X,loaded['X']))
# Y=np.vstack((Y,loaded['Y']))
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

# define CNN
num_filters=[16,8,4]
activations={'conv':tf.keras.layers.LeakyReLU(alpha=0.2),'latent':tf.keras.layers.PReLU(),'output':'linear'}
latent_dim=1024
droprate=.5
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
cnn=CNN(x_train.shape[1:], y_train.shape[1], num_filters=num_filters, latent_dim=latent_dim, droprate=droprate,
        activations=activations, optimizer=optimizer)
# f_name='cnn_combined_J'+str(ensbl_sz)
f_name='cnn_'+algs[1]+str(ensbl_sz)
try:
    cnn.model=load_model(os.path.join(folder,f_name+'.h5'),custom_objects={'loss':None})
#     cnn.model.load_weights(os.path.join(folder,f_name+'.h5'))
    print(f_name+' has been loaded!')
except Exception as err:
    print(err)
    try:
        cnn.model.load_weights(os.path.join(folder,f_name+'.h5'))
        print(f_name+' has been loaded!')
    except:
        print('Train CNN...\n')
        epochs=200
        patience=0
        cnn.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
        # save CNN
        cnn.model.save(os.path.join(folder,f_name+'.h5'))
    #     cnn.model.save_weights(os.path.join(folder,f_name+'.h5'))

# define the emulator (CNN-RNN)
loaded=np.load(file=os.path.join(folder,algs[1]+'_ensbl'+str(ensbl_sz)+'_training_XimgY.npz'))
X=loaded['X']
Y=loaded['Y']
# loaded=np.load(file=os.path.join(folder,algs[1]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
# X=np.vstack((X,loaded['X']))
# Y=np.vstack((Y,loaded['Y']))
Y=Y.reshape((-1,len(adif.misfit.observation_times),adif.misfit.targets.shape[0]))
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

# define CNN-RNN
num_filters=[16,8,4] # best for non-whiten [16,8,8]
# activations={'conv':'softplus','latent':'softmax','output':'linear'} # best for non-whiten
activations={'conv':tf.keras.layers.LeakyReLU(alpha=0.2),'latent':tf.keras.layers.PReLU(),'output':'sigmoid','lstm':'tanh'}
# activations={'conv':'relu','latent':tf.math.sin,'output':tf.keras.layers.PReLU()}
latent_dim=128 # best for non-whiten 256
droprate=0.25 # best for non-whiten .5
sin_init=lambda n:tf.random_uniform_initializer(minval=-tf.math.sqrt(6/n), maxval=tf.math.sqrt(6/n))
kernel_initializers={'conv':'he_uniform','latent':sin_init,'output':'glorot_uniform'}
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
# optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001)
cnnrnn=CNN_RNN(x_train.shape[1:], y_train.shape[1:], num_filters=num_filters, latent_dim=latent_dim, droprate=droprate,
               activations=activations, optimizer=optimizer)
# f_name='cnn_rnn_combined_J'+str(ensbl_sz)#+'_customloss'
f_name='cnn_rnn_'+algs[1]+str(ensbl_sz)#+'_customloss'
try:
#     cnnrnn.model=load_model(os.path.join(folder,f_name+'.h5'))
    cnnrnn.model=load_model(os.path.join(folder,f_name+'.h5'),custom_objects={'loss':None})
#     cnnrnn.model.load_weights(os.path.join(folder,f_name+'.h5'))
    print(f_name+' has been loaded!')
except Exception as err:
    print(err)
    try:
        cnnrnn.model.load_weights(os.path.join(folder,f_name+'.h5'))
        print(f_name+' has been loaded!')
    except:
        print('Train CNN-RNN...\n')
        epochs=200
        patience=0
        cnnrnn.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
        # save CNN-RNN
        cnnrnn.model.save(os.path.join(folder,f_name+'.h5'))
    #     cnnrnn.save(folder,f_name) # fails due to the custom kernel_initializer
    #     cnnrnn.model.save_weights(os.path.join(folder,f_name+'.h5'))


# read data and construct plot functions
u_f = df.Function(adif.prior.V)
# read MAP
try:
    f=df.XDMFFile(adif.mpi_comm, os.path.join(os.getcwd(),'properties/MAP.xdmf'))
    f.read_checkpoint(u_f,'m',0)
    f.close()
except:
    pass
u=u_f.vector()
# u=adif.prior.sample()
loglik = lambda y: -0.5*tf.math.reduce_sum((y-adif.misfit.obs)**2/adif.misfit.noise_variance,axis=1)
loglik_cnn = lambda x: loglik(cnn.model(x))
loglik_cnnrnn = lambda x: loglik(tf.reshape(cnnrnn.model(x),(-1,len(adif.misfit.obs))))
# calculate gradient
dll_xact = adif.get_geom(u,[0,1])[1]
# emulate gradient
dll_cnn = cnn.gradient(adif.vec2img(u)[None,:,:,None], loglik_cnn)
dll_cnnrnn = cnnrnn.gradient(adif.vec2img(u)[None,:,:,None], loglik_cnnrnn)

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
u_f.vector().set_local(adif.img2vec(dll_cnn,adif.prior.V if eldeg>1 else None))
sub_figs[1]=df.plot(u_f)
plt.title('Emulated Gradient (CNN)')
plt.axes(axes.flat[2])
u_f.vector().set_local(adif.img2vec(dll_cnnrnn,adif.prior.V if eldeg>1 else None))
sub_figs[2]=df.plot(u_f)
plt.title('Emulated Gradient (CNN-RNN)')
# add common colorbar
from util.common_colorbar import common_colorbar
fig=common_colorbar(fig,axes,sub_figs)

# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(savepath,'extrctgrad_cnn_rnn.png'),bbox_inches='tight')
# plt.show()