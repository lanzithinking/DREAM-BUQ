"""
This is to compare GP and DNN in emulating (extracted) gradients taking reference to those exactly calculated.
"""

import numpy as np
import tensorflow as tf
import gpflow as gpf
import sys,os,pickle
sys.path.append( '../' )
from BBD import BBD
sys.path.append( '../gp')
from multiGP import multiGP as GP
from nn.dnn import DNN
from tensorflow.keras.models import load_model
import timeit,pickle
import matplotlib.pyplot as plt

# tf.compat.v1.disable_eager_execution() # needed to train with custom loss # comment to plot
# set random seed
seeds = [2020+i*10 for i in range(20)]
n_seed = len(seeds)
# training/testing sizes
train_sizes = [50,100,200,500,1000]
n_train = len(train_sizes)
test_size = 100
# save relative errors and times
fun_errors = np.zeros((2,n_seed,n_train,test_size))
grad_errors = np.zeros((2,n_seed,n_train,test_size))
train_times = np.zeros((2,n_seed,n_train))
test_times = np.zeros((3,n_seed,n_train))

## define the BBD inverse problem ##
# set up
try:
    with open('./result/BBD.pickle','rb') as f:
        [nz_var,pr_cov,A,true_input,y]=pickle.load(f)
    print('Data loaded!\n')
    kwargs={'true_input':true_input,'A':A,'y':y}
except:
    print('No data found. Generate new data...\n')
    d=4; m=100
    nz_var=1; pr_cov=1
    true_input=np.random.randint(d,size=d)
    A=np.random.rand(m,d)
    kwargs={'true_input':true_input,'A':A}
bbd=BBD(nz_var=nz_var,pr_cov=pr_cov,**kwargs)
loglik = lambda y: -0.5*tf.math.reduce_sum((y-bbd.y[None,:])**2/bbd.nz_var[None,:],axis=1)
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1
ensbl_sz = 100

# load data
folder = './train_NN'

# load data for GP and DNN
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
X=loaded['X']
Y=loaded['Y']

# define GP
latent_dim=Y.shape[1]
kernel=gpf.kernels.SquaredExponential(lengthscales=np.random.rand(X.shape[1])) + gpf.kernels.Linear()
gp=GP(X.shape[1], Y.shape[1], latent_dim=latent_dim, kernel=kernel)

# define DNN
depth=3
activations={'hidden':'softplus','output':'linear'}
droprate=0.
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
dnn=DNN(X.shape[1], Y.shape[1], depth=depth, droprate=droprate,
        activations=activations, optimizer=optimizer)


# split train/test
num_samp=X.shape[0]
folder = './analysis'
try:
    with open(os.path.join(folder,'compare_gp_dnn_cnn.pckl'),'rb') as f:
        fun_errors,grad_errors,train_times,test_times=pickle.load(f)
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
                gp.evaluate=lambda x:gp.model.predict(x)[0] # cannot take gradient!
                print(f_name+' has been loaded!')
            except Exception as err:
                print(err)
                print('Train GP model with seed {} and training size {}...\n'.format(seeds[s],train_sizes[t]))
#                 gp.induce_num=np.min((np.ceil(.1*X.shape[1]).astype('int'),train_sizes[t]))
                epochs=200
        #         batch_size=128
        #         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
                kwargs={'maxiter':epochs} # train with scipy
            #     kwargs={'epochs':epochs,'batch_size':batch_size,'optimizer':optimizer} # train with tensorflow
                t_start=timeit.default_timer()
                try:
                    gp.train(X[sel4train],Y[sel4train],x_test=X[sel4test],y_test=Y[sel4test],**kwargs)
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
            
            # train DNN
            f_name='dnn_'+algs[alg_no]+str(ensbl_sz)+'_seed'+str(seeds[s])+'_trainsz'+str(train_sizes[t])
            try:
                dnn.model=load_model(os.path.join(folder+'/DNN/',f_name+'.h5'),custom_objects={'loss':None})
                print(f_name+' has been loaded!')
            except:
                try:
                    dnn.model.load_weights(os.path.join(folder+'/DNN/',f_name+'.h5'))
                    print(f_name+' has been loaded!')
                except Exception as err:
                    print(err)
                    print('Train DNN with seed {} and training size {}...\n'.format(seeds[s],train_sizes[t]))
                    epochs=200
                    patience=0
                    t_start=timeit.default_timer()
                    try:
                        dnn.train(X[sel4train],Y[sel4train],x_test=X[sel4test],y_test=Y[sel4test],epochs=epochs,batch_size=32,verbose=1,patience=patience)
                    except Exception as err:
                        print(err)
                        pass
                    t_used=timeit.default_timer()-t_start
                    train_times[1,s,t]=t_used
                    print('\nTime used for training DNN: {}'.format(t_used))
                    # save CNN
                    save_dir=folder+'/DNN/'
                    if not os.path.exists(save_dir): os.makedirs(save_dir)
                    try:
                        dnn.model.save(os.path.join(save_dir,f_name+'.h5'))
                    except:
                        dnn.model.save_weights(os.path.join(save_dir,f_name+'.h5'))
            
            # test
            logLik_g = lambda x: loglik(gp.evaluate(x))
            logLik_d = lambda x: loglik(dnn.model(x))
            t_used = np.zeros(3)
            print('Testing trained models...\n')
#             for n in range(test_size):
            u=X[sel4test]
            # calculate gradient
            t_start=timeit.default_timer()
            ll_xact,dll_xact = bbd.get_geom(u,[0,1])[:2]
            t_used[0] += timeit.default_timer()-t_start
            
            # emulation by GP
            t_start=timeit.default_timer()
            ll_emul = logLik_g(u).numpy()
            dll_emul = gp.gradient(u, logLik_g)
            t_used[1] += timeit.default_timer()-t_start
            # record difference
            dif_fun = np.abs(ll_xact - ll_emul)
            dif_grad = dll_xact - dll_emul
            fun_errors[0,s,t,:]=dif_fun
            grad_errors[0,s,t,:]=[np.linalg.norm(dif_grad[n])/np.linalg.norm(dll_xact[n]) for n in range(test_size)]
            
            # emulation by DNN
            t_start=timeit.default_timer()
            ll_emul = logLik_d(u).numpy()
            dll_emul = dnn.gradient(u, logLik_d)
            t_used[2] += timeit.default_timer()-t_start
            # record difference
            dif_fun = np.abs(ll_xact - ll_emul)
            dif_grad = dll_xact - dll_emul
            fun_errors[1,s,t,:]=dif_fun
            grad_errors[1,s,t,:]=[np.linalg.norm(dif_grad[n])/np.linalg.norm(dll_xact[n]) for n in range(test_size)]
                
            print('Time used for calculation: {} vs GP-emulation: {} vs DNN-emulation: {}'.format(*t_used.tolist()))
            test_times[0,s,t]=t_used[1]; test_times[1,s,t]=t_used[2]; test_times[2,s,t]=t_used[0]
    
    # save results
    with open(os.path.join(folder,'compare_gp_dnn.pckl'),'wb') as f:
        pickle.dump([fun_errors,grad_errors,train_times,test_times],f)

# make some pots
import pandas as pd
import seaborn as sns
# prepare for the data
alg_array=np.hstack((['GP']*n_seed*n_train,['DNN']*n_seed*n_train))
trs_array=np.zeros((2,n_seed,n_train),dtype=np.int)
for t in range(n_train): trs_array[:,:,t]=train_sizes[t]
fun_err_array=np.median(fun_errors,axis=3)
grad_err_array=np.median(grad_errors,axis=3)
train_time_array=train_times
# test_time_array=test_times[:3]
df_err=pd.DataFrame({'algorithm':alg_array.flatten(),
                     'training_size':trs_array.flatten(),
                     'function_error':fun_err_array.flatten(),
                     'gradient_error':grad_err_array.flatten(),
                     'training_time':train_time_array.flatten(),
#                      'testing_time':test_time_array.flatten()
                     })

alg_array=np.hstack((['GP']*n_seed*n_train,['DNN']*n_seed*n_train,['CALC']*n_seed*n_train))
trs_array=np.zeros((3,n_seed,n_train),dtype=np.int)
for t in range(n_train): trs_array[:,:,t]=train_sizes[t]
test_time_array=test_times
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
# plt.ylim(.5,1.5)
plt.gca().legend().set_title('')
# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(folder,'error_gp_dnn.png'),bbox_inches='tight')
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
plt.savefig(os.path.join(folder,'time_gp_dnn.png'),bbox_inches='tight')
# plt.show()