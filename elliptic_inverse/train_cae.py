"""
This is to train Convolutional AutoEncoder for dimension reduction.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import sys,os
sys.path.append( "../" )
from Elliptic import Elliptic
from util.dolfin_gadget import vec2fun,fun2img,img2fun
from nn.cae import ConvAutoEncoder
from tensorflow.keras.models import load_model

# functions needed to make even image size
def pad(A,sz=1):
    A_padded=np.zeros([i+sz for i in A.shape])
    A_padded[:-sz,:-sz]=A
    return A_padded
def chop(A,sz=1):
    return A[:-sz,:-sz]

# set random seed
np.random.seed(2020)
tf.random.set_seed(2020)

# define the inverse problem
nx=40; ny=40
SNR=50
elliptic = Elliptic(nx=nx,ny=ny,SNR=SNR)
# define the latent (coarser) inverse problem
nx=10; ny=10
obs,nzsd,loc=[getattr(elliptic.misfit,i) for i in ('obs','nzsd','loc')]
elliptic_latent = Elliptic(nx=nx,ny=ny,SNR=SNR,obs=obs,nzsd=nzsd,loc=loc)
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1

# define the convolutional autoencoder (AE)
# load data
ensbl_sz = 500
folder = './train_NN'
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_CNN.npz'))
X=loaded['X']
# pre-processing: scale X to 0-1
# X-=np.nanmin(X,axis=(1,2),keepdims=True) # try axis=(1,2,3)
# X/=np.nanmax(X,axis=(1,2),keepdims=True)
# X_min=np.nanmin(X)
# X-=X_min
# X_max=np.nanmax(X)
# X/=X_max
X=X[:,:-1,:-1,None]
# split train/test
num_samp=X.shape[0]
n_tr=np.int(num_samp*.75)
x_train=X[:n_tr]
x_test=X[n_tr:]

# define AE
num_filters=[16,1]
# activations={'conv':'linear','latent':tf.keras.layers.PReLU()}
# activations={'conv':'relu','latent':'linear'}
# activations={'conv':tf.keras.layers.LeakyReLU(alpha=0.1),'latent':'linear'}
activations={'conv':tf.keras.layers.LeakyReLU(alpha=0.1),'latent':None}
latent_dim=elliptic_latent.prior.dim
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
# optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001)
cae=ConvAutoEncoder(x_train.shape[1:], num_filters=num_filters, latent_dim=latent_dim,
                    activations=activations, optimizer=optimizer)
# nlpr = lambda y: tf.map_fn(lambda y_i:-elliptic.prior.logpdf(img2fun(np.squeeze(y_i.numpy()),elliptic.pde.V).vector()), y)
# # nll = lambda x: [-elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x_i.numpy().flatten()))[0] for x_i in x]
# nll = lambda x,_: tf.map_fn(lambda x_i:-elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x_i.numpy().flatten()))[0], x)
# nll = lambda x,y: [(elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x[i].numpy().flatten()))[0]
#                     -elliptic.get_geom(img2fun(np.squeeze(y[i].numpy()),elliptic.pde.V).vector())[0])**2 for i in range(x.shape[0])]
# nlpost = lambda x,y: [-elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x[i].numpy().flatten()))[0]
#                       -elliptic.prior.logpdf(img2fun(pad(np.squeeze(y[i].numpy())),elliptic.pde.V).vector()) for i in range(x.shape[0])]
# cae=ConvAutoEncoder(x_train.shape[1:], num_filters=num_filters, latent_dim=latent_dim,
#                     activations=activations, optimizer=optimizer, loss=nlpost, run_eagerly=True)
# folder=folder+'/saved_model'
f_name=['cae_'+i+'_'+algs[alg_no]+str(ensbl_sz)+'.h5' for i in ('fullmodel','encoder','decoder')]
try:
    cae.model=load_model(os.path.join(folder,f_name[0]),custom_objects={'loss':None})
    print(f_name[0]+' has been loaded!')
    cae.encoder=load_model(os.path.join(folder,f_name[1]),custom_objects={'loss':None})
    print(f_name[1]+' has been loaded!')
    cae.decoder=load_model(os.path.join(folder,f_name[2]),custom_objects={'loss':None})
    print(f_name[2]+' has been loaded!')
except Exception as err:
    print(err)
    print('Train Convolutional AutoEncoder...\n')
    epochs=200
    patience=0
    import timeit
    t_start=timeit.default_timer()
    cae.train(x_train,x_test=x_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training CAE: {}'.format(t_used))
    # save AE
    cae.model.save(os.path.join(folder,f_name[0]))
    cae.encoder.save(os.path.join(folder,f_name[1]))
    cae.decoder.save(os.path.join(folder,f_name[2]))

# # plot
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(15,5), facecolor='white')
# plt.ion()
# # plt.show(block=True)
# u_f = df.Function(elliptic.pde.V)
# u_f_lat = df.Function(elliptic_latent.pde.V)
# for n in range(20):
#     u=elliptic.prior.sample()
#     # encode
#     u_f.vector().set_local(u)
#     u_encoded=cae.encode(fun2img(u_f)[None,:-1,:-1,None])
#     # decode
#     u_decoded=cae.decode(u_encoded)
# #     u_decoded=cae.model.predict(fun2img(u_f)[None,:-1,:-1,None])
#     
#     # compute the log-volumes
# #     logvol_enc=cae.logvol(fun2img(u_f)[None,:-1,:-1,None],'encode')
# #     print('Log-volume of encoder: {}'.format(logvol_enc))
# #     logvol_dec=cae.logvol(u_encoded,'decode')
# #     print('Log-volume of decoder: {}'.format(logvol_dec))
#     
#     # plot
#     plt.subplot(131)
#     u_f.vector().set_local(u)
#     df.plot(u_f)
#     plt.title('Original Sample')
#     plt.subplot(132)
#     u_f_lat.vector().set_local(u_encoded.flatten())
#     df.plot(u_f_lat)
#     plt.title('Latent Sample')
#     plt.subplot(133)
#     u_decoded=np.squeeze(u_decoded)
#     u_decoded*=X_max; u_decoded+=X_min
#     u_decoded1=np.zeros([i+1 for i in u_decoded.shape])
#     u_decoded1[:-1,:-1]=u_decoded
#     u_f=img2fun(u_decoded1,elliptic.pde.V)
#     df.plot(u_f)
#     plt.title('Reconstructed Sample')
#     plt.draw()
#     plt.pause(1.0/10.0)

# read data and construct plot functions
u_f = df.Function(elliptic.pde.V)
u_f_lat = df.Function(elliptic_latent.pde.V)
# read MAP
try:
    f=df.HDF5File(elliptic.pde.mpi_comm, os.path.join('./result',"MAP_SNR"+str(SNR)+".h5"), "r")
    f.read(u_f,"parameter")
    f.close()
except:
    pass
u=u_f.vector()
# encode
u_encoded=cae.encode(fun2img(u_f)[None,:-1,:-1,None])
# decode
u_decoded=cae.decode(u_encoded)

# plot
import matplotlib.pyplot as plt
import matplotlib as mp
plt.rcParams['image.cmap'] = 'jet'
fig,axes = plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True,figsize=(15,5))
# fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
sub_figs=[None]*3
# plot
plt.axes(axes.flat[0])
# u_f.vector().set_local(u)
sub_figs[0]=df.plot(u_f)
# sub_figs[0]=plt.imshow(fun2img(u_f),origin='lower')
plt.title('Original')
plt.axes(axes.flat[1])
# u_f_lat.vector().set_local(u_encoded.flatten())
# u_f_lat=img2fun(u_encoded.reshape((nx+1,ny+1)),elliptic_latent.pde.V)
u_f_lat=img2fun(pad(np.squeeze(u_encoded)),elliptic_latent.pde.V)
sub_figs[1]=df.plot(u_f_lat)
# sub_figs[1]=plt.imshow(u_encoded.reshape((nx+1,ny+1)),origin='lower')
plt.title('Latent')
plt.axes(axes.flat[2])
u_decoded=np.squeeze(u_decoded)
# u_decoded*=X_max; u_decoded+=X_min
u_f=img2fun(pad(u_decoded),elliptic.pde.V)
sub_figs[2]=df.plot(u_f)
# sub_figs[2]=plt.imshow(u_decoded,origin='lower')
plt.title('Reconstructed')

# add common colorbar
from util.common_colorbar import common_colorbar
fig=common_colorbar(fig,axes,sub_figs)

# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(folder,'latent_reconstructed_cae.png'),bbox_inches='tight')
# plt.show()