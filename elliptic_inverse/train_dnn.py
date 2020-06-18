#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:03:47 2020

@author: apple
"""

"""
This is to train AutoEncoder for dimension reduction.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import sys,os
sys.path.append( "../" )
from Elliptic import Elliptic
# from util.dolfin_gadget import vec2fun,fun2img,img2fun
from nn.dnn import DNN
from util.dolfin_gadget import vec2fun,fun2img,img2fun
from tensorflow.keras.models import load_model

#tf.compat.v1.disable_eager_execution()
# set random seed
np.random.seed(2020)
tf.random.set_seed(2020)

# define the inverse problem
nx=40; ny=40
SNR=50
elliptic = Elliptic(nx=nx,ny=ny,SNR=SNR)
# define the latent (coarser) inverse problem
nx=20; ny=20
obs,nzsd,loc=[getattr(elliptic.misfit,i) for i in ('obs','nzsd','loc')]
elliptic_latent = Elliptic(nx=nx,ny=ny,SNR=SNR,obs=obs,nzsd=nzsd,loc=loc)
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=0

# define the autoencoder (AE)
# load data
ensbl_sz = 250
folder = './train_DNN'
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_CNN.npz'))
X=loaded['X']
Y=loaded['Y']
X = X.reshape((-1,41*41))
# pre-processing: scale X to 0-1
# X-=np.nanmin(X,axis=(1),keepdims=True) # try axis=(1,2,3)
# X/=np.nanmax(X,axis=(1),keepdims=True)
# split train/test
num_samp=X.shape[0]
n_tr=np.int(num_samp*.75)
x_train,y_train=X[:n_tr],Y[:n_tr]
x_test,y_test=X[n_tr:],Y[n_tr:]

# define DNN
activation='linear'
# activation=tf.keras.layers.LeakyReLU(alpha=0.01)
droprate=.25
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
loglik = lambda y: -0.5*elliptic.misfit.prec*tf.math.reduce_sum((y-elliptic.misfit.obs)**2,axis=1)
custom_loss = lambda y_true, y_pred: [tf.square(loglik(y_true)-loglik(y_pred)), elliptic.misfit.prec*(y_true-y_pred)]

dnn=DNN(x_train.shape[1], y_train.shape[1], depth=4, node_sizes = np.array([512,256,128,25]), 
        activation=activation, droprate=droprate, optimizer=optimizer,loss=custom_loss)
f_name='dnn_'+algs[alg_no]+str(ensbl_sz)+'.h5'

# # nll = lambda x: [-elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x_i.numpy().flatten()))[0] for x_i in x]
# # nll = lambda x: tf.map_fn(lambda x_i:-elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x_i.numpy().flatten()))[0], x)
# nll = lambda x,y: [(elliptic_latent.get_geom(elliptic_latent.prior.gen_vector(x[i].numpy().flatten()))[0]
#                     -elliptic.get_geom(elliptic.prior.gen_vector(y[i].numpy().flatten()))[0])**2 for i in range(x.shape[0])]
# ae=AutoEncoder(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim,
#                activation=activation, optimizer=optimizer, loss=nll, run_eagerly=True)
# folder=folder+'/saved_model'
try:
    dnn.model=load_model(os.path.join(folder,f_name),custom_objects={'loss':None})
    print(f_name+' has been loaded!')
except Exception as err:
    print(err)
    print('Train AutoEncoder...\n')
    epochs=200
    patience=0
    import timeit
    t_start=timeit.default_timer()
    dnn.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=32,patience=5,verbose=1)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training AE: {}'.format(t_used))
    # save DNN
    dnn.model.save(os.path.join(folder,f_name))

# some more test
logLik = lambda x: loglik(dnn.model(x))
import timeit
t_used = np.zeros(2)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,6), facecolor='white')
plt.ion()
# plt.show(block=True)
u_f = df.Function(elliptic.pde.V)
for n in range(10):
    u=elliptic.prior.sample() #(1681,)
    # calculate gradient
    t_start=timeit.default_timer()
    dll_xact = elliptic.get_geom(u,[0,1])[1] #(1681,)
    t_used[0] += timeit.default_timer()-t_start
    # emulate gradient
    t_start=timeit.default_timer()
    #u_img=fun2img(vec2fun(u,elliptic.pde.V)) transform u(1681) to u_img(41,41)
    dll_emul = dnn.gradient(u.get_local()[None,:],logLik) #(1681,)
    t_used[1] += timeit.default_timer()-t_start
    # test difference
    dif = dll_xact.get_local() - dll_emul
    print('Difference between the calculated and emulated gradients: min ({}), med ({}), max ({})'.format(dif.min(),np.median(dif),dif.max()))
    
    # check the gradient extracted from emulation
    v=elliptic.prior.sample()
    #v_img=fun2img(vec2fun(v,elliptic.pde.V))
    h=1e-4
    #dll_emul_fd_v=(logLik(u_img[None,:,:,None]+h*v_img[None,:,:,None])-logLik(u_img[None,:,:,None]))/h
    dll_emul_fd_v=(logLik(u.get_local()[None,:]+h*v.get_local()[None,:])-logLik(u.get_local()[None,:]))/h
    reldif = abs(dll_emul_fd_v - dll_emul.flatten().dot(v.get_local().flatten()))/v.norm('l2')
    print('Relative difference between finite difference and extracted results: {}'.format(reldif))
    
    # plot
    plt.subplot(121)
    u_f.vector().set_local(dll_xact)
    df.plot(u_f)
    plt.title('Calculated Gradient')
    plt.subplot(122)
    #u_f=img2fun(dll_emul,elliptic.pde.V)
    u_f.vector().set_local(dll_emul)
    df.plot(u_f)
    plt.title('Emulated Gradient')
    plt.draw()
    plt.pause(1.0/10.0)
    
print('Time used to calculate vs emulate gradients: {} vs {}'.format(*t_used.tolist()))

