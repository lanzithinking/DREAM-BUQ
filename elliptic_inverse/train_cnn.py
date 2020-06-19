"""
This is to train CNN to emulate (extracted) gradients compared with those exactly calculated.
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

#tf.compat.v1.disable_eager_execution() # needed to train with custom loss # comment to plot
#tf.config.experimental_run_functions_eagerly(True)
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
folder = './train_DNN'
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
# activations={'conv':'relu','latent':tf.keras.layers.PReLU(),'output':'linear'}
activations={'conv':tf.keras.layers.LeakyReLU(alpha=0.1),'latent':tf.keras.layers.PReLU(),'output':'linear'}
latent_dim=128
droprate=.5
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
# optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001)
#cnn=CNN(x_train.shape[1:], y_train.shape[1], num_filters=num_filters, latent_dim=latent_dim,
#        activations=activations, droprate=droprate, optimizer=optimizer, padding='same')
loglik = lambda y: -0.5*elliptic.misfit.prec*tf.math.reduce_sum((y-elliptic.misfit.obs)**2,axis=1)
custom_loss = lambda y_true, y_pred: [tf.square(loglik(y_true)-loglik(y_pred)), elliptic.misfit.prec*(y_true-y_pred)]
cnn=CNN(x_train.shape[1:], y_train.shape[1], num_filters=num_filters, latent_dim=latent_dim,
         activations=activations, droprate=droprate, optimizer=optimizer, loss=custom_loss)
# folder=folder+'/saved_model'
f_name='cnn_'+algs[alg_no]+str(ensbl_sz)+'.h5'
try:
    cnn.model=load_model(os.path.join(folder,f_name),custom_objects={'loss':None})
    #cnn.model=load_model(os.path.join(folder,f_name),custom_objects={"loss": custom_loss,'LeakyReLU': tf.keras.layers.LeakyReLU(alpha=0.1)})
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
    cnn.model.save(os.path.join(folder,f_name))
#     cnn.save(folder,'cnn_'+algs[alg_no]+str(ensbl_sz))

# some more test
logLik = lambda x: loglik(cnn.model(x))
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
    u_img=fun2img(vec2fun(u,elliptic.pde.V))
    dll_emul = cnn.gradient(u_img[None,:,:,None], logLik)
    t_used[1] += timeit.default_timer()-t_start
    # test difference
    dif = dll_xact - img2fun(dll_emul,elliptic.pde.V).vector()
    print('Difference between the calculated and emulated gradients: min ({}), med ({}), max ({})'.format(dif.min(),np.median(dif.get_local()),dif.max()))
    
    # check the gradient extracted from emulation
    v=elliptic.prior.sample()
    v_img=fun2img(vec2fun(v,elliptic.pde.V))
    h=1e-4
    dll_emul_fd_v=(logLik(u_img[None,:,:,None]+h*v_img[None,:,:,None])-logLik(u_img[None,:,:,None]))/h
    reldif = abs(dll_emul_fd_v - dll_emul.flatten().dot(v_img.flatten()))/v.norm('l2')
    print('Relative difference between finite difference and extracted results: {}'.format(reldif))
    
    # plot
    plt.subplot(121)
    u_f.vector().set_local(dll_xact)
    df.plot(u_f)
    plt.title('Calculated Gradient')
    plt.subplot(122)
    u_f=img2fun(dll_emul,elliptic.pde.V)
    df.plot(u_f)
    plt.title('Emulated Gradient')
    plt.draw()
    plt.pause(1.0/10.0)
    
print('Time used to calculate vs emulate gradients: {} vs {}'.format(*t_used.tolist()))