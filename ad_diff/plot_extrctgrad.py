"""
This is to plot emulated (extracted) gradients compared with those exactly calculated.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import sys,os
sys.path.append( "../" )
from advdiff import advdiff
from util.dolfin_gadget import vec2fun
from nn.cnn import CNN
from nn.dnn import DNN
from tensorflow.keras.models import load_model

# functions to convert vectors between P1 and Pn
def vinP1(v, V):
    """project v to P1 space"""
    mesh = V.mesh()
    V_P1 = df.FunctionSpace(mesh, V.ufl_element().family(), 1)
    d2v = df.dof_to_vertex_map(V_P1)
    f = df.Function(V)
    f.vector().set_local(v)
#     vec = df.Function(V_P1).vector()
    vec = df.Vector(mesh.mpi_comm(),mesh.num_vertices())
    vec.set_local(f.compute_vertex_values(mesh)[d2v])
    return vec

def vinP(v, V):
    """project v to Pn space"""
    V_P1 = df.FunctionSpace(V.mesh(), V.ufl_element().family(), 1)
    f_P1 = df.Function(V_P1)
    f_P1.vector().set_local(v)
    f = df.Function(V)
    f.interpolate(f_P1)
    return f.vector()

# tf.compat.v1.disable_eager_execution() # needed to train with custom loss # comment to plot
# set random seed
seed=2020
np.random.seed(seed)
tf.random.set_seed(seed)

# define the inverse problem
meshsz = (51,51)
eldeg = 2
rel_noise = .5
nref = 1
adif = advdiff(mesh=meshsz, eldeg=eldeg, rel_noise=rel_noise, nref=nref, seed=seed)
adif.prior.V=adif.prior.Vh
adif.misfit.obs=np.array([dat.get_local() for dat in adif.misfit.d.data]).flatten()
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
# alg_no=1
# load data
ensbl_sz = 500
folder = './train_NN'
savepath = './analysis_eldeg'+str(eldeg)
if not os.path.exists(savepath): os.makedirs(savepath)

## define the emulator (CNN) ##
loaded=np.load(file=os.path.join(folder,algs[0]+'_ensbl'+str(ensbl_sz)+'_training_XimgY.npz'))
X=loaded['X']
Y=loaded['Y']
loaded=np.load(file=os.path.join(folder,algs[1]+'_ensbl'+str(ensbl_sz)+'_training_XimgY.npz'))
X=np.vstack((X,loaded['X']))
Y=np.vstack((Y,loaded['Y']))
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
num_filters=[16,8,8]
activations={'conv':'softplus','latent':'softmax','output':'linear'}
latent_dim=1024
droprate=.25
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
cnn=CNN(x_train.shape[1:], y_train.shape[1], num_filters=num_filters, latent_dim=latent_dim, droprate=droprate,
        activations=activations, optimizer=optimizer)
f_name='cnn_combined_J'+str(ensbl_sz)
try:
    cnn.model=load_model(os.path.join(folder,f_name+'.h5'),custom_objects={'loss':None})
#     cnn.model.load_weights(os.path.join(folder,f_name+'.h5'))
    print(f_name+' has been loaded!')
except Exception as err:
    print(err)
    print('Train CNN...\n')
    epochs=200
    patience=0
    cnn.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
    # save CNN
    cnn.model.save(os.path.join(folder,f_name+'.h5'))
#     cnn.model.save_weights(os.path.join(folder,f_name+'.h5'))

# define the emulator (DNN)
loaded=np.load(file=os.path.join(folder,algs[0]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
X=loaded['X']
Y=loaded['Y']
loaded=np.load(file=os.path.join(folder,algs[1]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
X=np.vstack((X,loaded['X']))
Y=np.vstack((Y,loaded['Y']))
# pre-processing: scale X to 0-1
# X-=np.nanmin(X,axis=1,keepdims=True)
# X/=np.nanmax(X,axis=1,keepdims=True)
# split train/test
num_samp=X.shape[0]
# n_tr=np.int(num_samp*.75)
# x_train,y_train=X[:n_tr],Y[:n_tr]
# x_test,y_test=X[n_tr:],Y[n_tr:]
tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
x_train,x_test=X[tr_idx],X[te_idx]
y_train,y_test=Y[tr_idx],Y[te_idx]

## define DNN ##
depth=5
activations={'hidden':tf.keras.layers.LeakyReLU(alpha=.01),'output':'linear'}
droprate=.25
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
dnn=DNN(x_train.shape[1], y_train.shape[1], depth=depth, droprate=droprate,
        activations=activations, optimizer=optimizer)
f_name='dnn_combined_J'+str(ensbl_sz)#+'_customloss'
try:
#     dnn.model=load_model(os.path.join(folder,f_name+'.h5'))
    dnn.model=load_model(os.path.join(folder,f_name+'.h5'),custom_objects={'loss':None})
#     dnn.model.load_weights(os.path.join(folder,f_name+'.h5'))
    print(f_name+' has been loaded!')
except Exception as err:
    print(err)
    print('Train DNN...\n')
    epochs=200
    patience=0
    dnn.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
    # save DNN
    dnn.model.save(os.path.join(folder,f_name+'.h5'))
#     dnn.save(folder,f_name) # fails due to the custom kernel_initializer
#     dnn.model.save_weights(os.path.join(folder,f_name+'.h5'))


# read data and construct plot functions
u_f = df.Function(adif.prior.V)
# read MAP
try:
    f=df.XDMFFile(adif.mpi_comm, os.path.join(os.getcwd(),'results/MAP.xdmf'))
    f.read_checkpoint(u_f,'m',0)
    f.close()
except:
    pass
u=u_f.vector()
# u=adif.prior.sample()
loglik = lambda y: -0.5*tf.math.reduce_sum((y-adif.misfit.obs)**2/adif.misfit.noise_variance,axis=1)
loglik_cnn = lambda x: loglik(cnn.model(x))
loglik_dnn = lambda x: loglik(dnn.model(x))
# calculate gradient
dll_xact = adif.get_geom(u,[0,1])[1]
# emulate gradient
dll_cnn = cnn.gradient(adif.vec2img(u)[None,:,:,None], loglik_cnn)
dll_dnn = dnn.gradient(u.get_local()[None,:] if eldeg==1 else vinP1(u,adif.prior.V).get_local()[None,:], loglik_dnn)

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
u_f.vector().set_local(dll_dnn if eldeg==1 else vinP(dll_dnn,adif.prior.V))
sub_figs[2]=df.plot(u_f)
plt.title('Emulated Gradient (DNN)')
# add common colorbar
from util.common_colorbar import common_colorbar
fig=common_colorbar(fig,axes,sub_figs)

# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(savepath,'extrctgrad.png'),bbox_inches='tight')
# plt.show()