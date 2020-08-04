"""
This is to compare GP and CNN in emulating (extracted) gradients taking reference to those exactly calculated.
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import gpflow as gpf
import sys,os
sys.path.append( '../' )
from Elliptic import Elliptic
sys.path.append( '../gp')
from multiGP import multiGP as GP
from util.dolfin_gadget import vec2fun,fun2img,img2fun
from util.multivector import *
from nn.cnn import CNN
from tensorflow.keras.models import load_model
import timeit,pickle
import matplotlib.pyplot as plt

# tf.compat.v1.disable_eager_execution() # needed to train with custom loss # comment to plot
# set random seed
seeds = [2020+i*10 for i in range(10)]
n_seed = len(seeds)
# training/testing sizes
train_sizes = [50,100,200,500,1000]
n_train = len(train_sizes)
test_size = 100
# save relative errors and times
fun_errors = np.zeros((2,n_seed,n_train,test_size))
grad_errors = np.zeros((2,n_seed,n_train,test_size))
train_times = np.zeros((2,n_seed,n_train))
pred_times = np.zeros((3,n_seed,n_train))

# define the inverse problem
nx=40; ny=40
SNR=50
elliptic = Elliptic(nx=nx,ny=ny,SNR=SNR)
loglik = lambda y: -0.5*elliptic.misfit.prec*tf.math.reduce_sum((y-elliptic.misfit.obs)**2,axis=1)
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1
ensbl_sz = 500

# load data
folder = './train_NN'

# load data for GP
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
Xg=loaded['X']
Yg=loaded['Y']

# define GP
latent_dim=Yg.shape[1]
kernel=gpf.kernels.SquaredExponential() + gpf.kernels.Linear()
gp=GP(Xg.shape[1], Yg.shape[1], latent_dim=latent_dim, kernel=kernel)

# load data for CNN
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XimgY.npz'))
Xc=loaded['X']
Yc=loaded['Y']
Xc=Xc[:,:,:,None]

# define CNN
num_filters=[16,8,8]
activations={'conv':'softplus','latent':'softmax','output':'linear'}
latent_dim=256
droprate=.5
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
cnn=CNN(Xc.shape[1:], Yc.shape[1], num_filters=num_filters, latent_dim=latent_dim, droprate=droprate,
        activations=activations, optimizer=optimizer)

# split train/test
num_samp=Xg.shape[0]
folder = './analysis_f_SNR'+str(SNR)
try:
    with open(os.path.join(folder,'compare_gp_cnn.pckl'),'rb') as f:
        fun_errors,grad_errors,train_times,pred_times=pickle.load(f)
    print('Comparison results loaded!')
except:
    print('Obtaining comparison results...\n')
    for s in range(n_seed):
        np.random.seed(seeds[s])
        tf.random.set_seed(seeds[s])
        prng=np.random.RandomState(seeds[s])
        for t in range(n_train):
            # select training and testing data
            sel4train = prng.choice(num_samp,size=train_sizes[t],replace=False)
            sel4test = prng.choice(np.setdiff1d(range(num_samp),sel4train),size=test_size,replace=False)
            
            # train GP
            f_name='gp_'+algs[alg_no]+str(ensbl_sz)+'_seed'+str(seeds[s])+'_trainsz'+str(train_sizes[t])
            try:
                gp.model=tf.saved_model.load(os.path.join(folder+'/GP/',f_name))
                gp.evaluate=gp.model.predict # cannot take gradient!
                print(f_name+' has been loaded!')
            except Exception as err:
                print(err)
                print('Train GP model with seed {} and training size {}...\n'.format(seeds[s],train_sizes[t]))
                gp.induce_num=np.min((np.ceil(.1*Xg.shape[1]).astype('int'),train_sizes[t]))
                epochs=200
        #         batch_size=128
        #         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
                kwargs={'maxiter':epochs} # train with scipy
            #     kwargs={'epochs':epochs,'batch_size':batch_size,'optimizer':optimizer} # train with tensorflow
                t_start=timeit.default_timer()
                try:
                    gp.train(Xg[sel4train],Yg[sel4train],x_test=Xg[sel4test],y_test=Yg[sel4test],**kwargs)
                except Exception as err:
                    print(err)
                    pass
                t_used=timeit.default_timer()-t_start
                train_times[0,s,t]=t_used
                print('\nTime used for training GP: {}'.format(t_used))
                # save GP
                save_dir=folder+'/GP/'+f_name
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                gp.save(save_dir)
            
            # train CNN
            f_name='cnn_'+algs[alg_no]+str(ensbl_sz)+'_seed'+str(seeds[s])+'_trainsz'+str(train_sizes[t])
            try:
                cnn.model=load_model(os.path.join(folder+'/CNN/',f_name+'.h5'),custom_objects={'loss':None})
                print(f_name+' has been loaded!')
            except:
                try:
                    cnn.model.load_weights(os.path.join(folder+'/CNN/',f_name+'.h5'))
                    print(f_name+' has been loaded!')
                except Exception as err:
                    print(err)
                    print('Train CNN with seed {} and training size {}...\n'.format(seeds[s],train_sizes[t]))
                    epochs=200
                    patience=0
                    t_start=timeit.default_timer()
                    try:
                        cnn.train(Xc[sel4train],Yc[sel4train],x_test=Xc[sel4test],y_test=Yc[sel4test],epochs=epochs,batch_size=64,verbose=1,patience=patience)
                    except Exception as err:
                        print(err)
                        pass
                    t_used=timeit.default_timer()-t_start
                    train_times[1,s,t]=t_used
                    print('\nTime used for training CNN: {}'.format(t_used))
                    # save CNN
                    save_dir=folder+'/CNN/'
                    if not os.path.exists(save_dir): os.makedirs(save_dir)
                    try:
                        cnn.model.save(os.path.join(save_dir,f_name+'.h5'))
                    except:
                        cnn.model.save_weights(os.path.join(save_dir,f_name+'.h5'))
            
            # test
            logLik_g = lambda x: loglik(gp.evaluate(x))
            logLik_c = lambda x: loglik(cnn.model(x))
            t_used = np.zeros(3)
            print('Testing trained models...\n')
            for n in range(test_size):
                u=elliptic.prior.gen_vector(Xg[sel4test][n])
                # calculate gradient
                t_start=timeit.default_timer()
                ll_xact,dll_xact = elliptic.get_geom(u,[0,1])[:2]
                t_used[0] += timeit.default_timer()-t_start
                
                # emulation by GP
                t_start=timeit.default_timer()
                ll_emul = logLik_g(u.get_local()[None,:]).numpy()
                dll_emul = gp.gradient(u.get_local()[None,:], logLik_g)
                t_used[1] += timeit.default_timer()-t_start
                # record difference
                dif_fun = np.abs(ll_xact - ll_emul)
                dif_grad = dll_xact.get_local() - dll_emul
                fun_errors[0,s,t,n]=dif_fun
                grad_errors[0,s,t,n]=np.linalg.norm(dif_grad)/dll_xact.norm('l2')
                
                # emulation by CNN
                t_start=timeit.default_timer()
                u_img=fun2img(vec2fun(u,elliptic.pde.V))
                ll_emul = logLik_c(u_img[None,:,:,None]).numpy()
                dll_emul = cnn.gradient(u_img[None,:,:,None], logLik_c) #* grad_scalfctr
                t_used[2] += timeit.default_timer()-t_start
                # record difference
                dif_fun = np.abs(ll_xact - ll_emul)
                dif_grad = dll_xact - img2fun(dll_emul,elliptic.pde.V).vector()
                fun_errors[1,s,t,n]=dif_fun
                grad_errors[1,s,t,n]=dif_grad.norm('l2')/dll_xact.norm('l2')
                
            print('Time used for calculation: {} vs GP-emulation: {} vs CNN-emulation: {}'.format(*t_used.tolist()))
            pred_times[0,s,t]=t_used[1]; pred_times[1,s,t]=t_used[2]; pred_times[2,s,t]=t_used[0]
    
    # save results
    with open(os.path.join(folder,'compare_gp_cnn.pckl'),'wb') as f:
        pickle.dump([fun_errors,grad_errors,train_times,pred_times],f)

# make some pots
import pandas as pd
import seaborn as sns
# prepare for the data
alg_array=np.hstack((['GP']*n_seed*n_train,['CNN']*n_seed*n_train))
trs_array=np.zeros((2,n_seed,n_train),dtype=np.int)
for t in range(n_train): trs_array[:,:,t]=train_sizes[t]
fun_err_array=np.median(fun_errors,axis=3)
grad_err_array=np.median(grad_errors,axis=3)
train_time_array=train_times
# test_time_array=pred_times[:2]
df_err=pd.DataFrame({'algorithm':alg_array.flatten(),
                     'training_size':trs_array.flatten(),
                     'function_error':fun_err_array.flatten(),
                     'gradient_error':grad_err_array.flatten(),
                     'training_time':train_time_array.flatten(),
#                      'testing_time':test_time_array.flatten()
                     })

alg_array=np.hstack((['GP']*n_seed*n_train,['CNN']*n_seed*n_train,['FE']*n_seed*n_train))
trs_array=np.zeros((3,n_seed,n_train),dtype=np.int)
for t in range(n_train): trs_array[:,:,t]=train_sizes[t]
test_time_array=pred_times
df_time=pd.DataFrame({'algorithm':alg_array.flatten(),
                      'training_size':trs_array.flatten(),
                      'testing_time':test_time_array.flatten()
                     })

# plot errors
fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=False,figsize=(12,5))
# plot
plt.axes(axes.flat[0])
sns.barplot(x='training_size',y='function_error',hue='algorithm',data=df_err,errwidth=1,capsize=.1)
# plt.title('Error of Function')
plt.gca().legend().set_title('')
plt.axes(axes.flat[1])
sns.barplot(x='training_size',y='gradient_error',hue='algorithm',data=df_err,errwidth=1,capsize=.1)
# plt.title('Error of Gradient')
plt.ylim(.5,1.5)
plt.gca().legend().set_title('')
# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(folder,'error_gp_cnn.png'),bbox_inches='tight')
# plt.show()

# plot times
fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=False,figsize=(12,5))
# plot
plt.axes(axes.flat[0])
# sns.pointplot(x='training_size',y='training_time',hue='algorithm',data=df_err,errwidth=.8,capsize=.1,scale=.5)
sns.pointplot(x='training_size',y='training_time',hue='algorithm',data=df_err,ci=None)
# plt.title('Training Time')
plt.gca().legend().set_title('')
plt.axes(axes.flat[1])
sns.pointplot(x='training_size',y='testing_time',hue='algorithm',data=df_time,ci=None)
# plt.title('Testint Time')
plt.gca().legend().set_title('')
# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(folder,'time_gp_cnn.png'),bbox_inches='tight')
# plt.show()