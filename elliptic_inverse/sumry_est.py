"""
Get summary of estimates of uncertainty field u in Elliptic inverse problem.
Shiwei Lan @ ASU, 2020
"""

# import modules
import numpy as np
import dolfin as df
from Elliptic import Elliptic
import sys
sys.path.append( "../" )
from nn.autoencoder import AutoEncoder
from tensorflow.keras.models import load_model
import os,pickle
from scipy.stats import norm
import timeit

# np.random.seed(2020)

## define the inverse elliptic problem ##
# parameters for PDE model
nx=40;ny=40;
# parameters for prior model
sigma=1.25;s=0.0625
# parameters for misfit model
SNR=50 # 100
# define the inverse problem
elliptic=Elliptic(nx=nx,ny=ny,SNR=SNR,sigma=sigma,s=s)

# define AutoEncoder
loaded=np.load(file='../nn/training.npz')
X=loaded['X']
num_samp=X.shape[0]
tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
x_train,x_test=X[tr_idx],X[te_idx]
# define Auto-Encoder
latent_dim=441; half_depth=3
ae=AutoEncoder(x_train, x_test,latent_dim, half_depth)
try:
    folder = os.path.join(os.getcwd(),'analysis_f_SNR'+str(SNR))
    ae.model=load_model(os.path.join(folder,'ae_fullmodel.h5'))
    ae.encoder=load_model(os.path.join(folder,'ae_encoder.h5'))
    from tensorflow.keras import backend as K
    ae.decoder=K.function(inputs=ae.model.get_layer(name="encode_out").output,outputs=ae.model.output)
    print('AutoEncoder has been loaded.')
except Exception as err:
    print(err)
    print('Train AutoEncoder...\n')
    epochs=200
    import timeit
    t_start=timeit.default_timer()
    ae.train(epochs,batch_size=64,verbose=1)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training AutoEncoder: {}'.format(t_used))

algs=['VB','pCN','AE-pCN']
num_algs=len(algs)

import os
folder = os.path.join(os.getcwd(),'analysis_f_SNR'+str(SNR))
hdf5_files=[f for f in os.listdir(folder) if f.endswith('.h5')]
pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
npz_files=[f for f in os.listdir(folder) if f.endswith('.npz')]
num_samp=10000

_time=np.zeros(num_algs)
_mean=np.zeros((num_algs,3))
_std=np.zeros((num_algs,3))
_incl=np.zeros(num_algs)

# obtain MAP
MAP=df.Function(elliptic.pde.V,name="parameter")
try:
    f=df.HDF5File(elliptic.pde.mpi_comm, os.path.join('./result',"MAP_SNR"+str(SNR)+".h5"), "r")
    f.read(MAP,"parameter")
    f.close()
except:
    pass
MAP=MAP.vector().get_local()

seq=np.arange(num_algs).tolist()
seq.insert(0,seq.pop(1))
for i in seq:
    if i==0:
        # obtain vb estimate
        for f_i in npz_files:
            if algs[i]+'_' in f_i:
                try:
                    f=np.load(file=os.path.join(folder,f_i))
                    print(f_i+' has been read!')
                    _time[i]=f['t_used']
                    u_m=f['mean']; u_sd=np.abs(f['std'])
                    _mean[i]=[u_m.min(),np.median(u_m),u_m.max()]
                    _std[i]=[u_sd.min(),np.median(u_sd),u_sd.max()]
                    break
                except:
                    u_m=np.zeros(elliptic.prior.dim)
                    u_sd=np.zeros(elliptic.prior.dim)
                    pass
    elif i>0:
        # plot posterior estimate
        samp_f=df.Function(elliptic.pde.V,name="parameter")
        u_m=elliptic.prior.gen_vector()
        u_sd=elliptic.prior.gen_vector()
        if algs[i]=='pCN':
            num_read=0
            found=False
            for f_i in hdf5_files:
                if '_'+algs[i]+'_' in f_i:
                    try:
                        f=df.HDF5File(elliptic.pde.mpi_comm,os.path.join(folder,f_i),"r")
                        u_m.zero(); u_sd.zero()
                        for s in range(num_samp):
                            f.read(samp_f,'sample_{0}'.format(s))
                            u_m.axpy(1.,samp_f.vector())
                            u_sd.axpy(1.,samp_f.vector()*samp_f.vector())
                            num_read+=1
                        f.close()
                        print(f_i+' has been read!')
                        found=True
                        break
                    except:
                        pass
            if found:
                u_m=u_m.get_local()/num_read
                u_sd=np.sqrt(u_sd.get_local()/num_read-u_m**2)
            else:
                u_m=np.zeros(elliptic.prior.dim)
                u_sd=np.zeros(elliptic.prior.dim)
            MCMC_est=u_m
        elif algs[i]=='AE-pCN':
            found=False
            for f_i in pckl_files:
                if '_'+algs[i]+'_' in f_i:
                    try:
                        f=open(os.path.join(folder,f_i),'rb')
                        f_read=pickle.load(f)
                        samp=f_read[3]
                        f.close()
                        print(f_i+' has been read!')
                        found=True
                    except:
                        pass
            if found:
                samp_=ae.decode(samp)
                u_m.zero(); u_sd.zero()
                t_start=timeit.default_timer()
                for s in range(num_samp):
#                     samp_f.vector().set_local(elliptic.prior.v2u(elliptic.prior.gen_vector(samp_[s])))
                    samp_f.vector().set_local(elliptic.prior.C_act(samp_[s],.5))
                    u_m.axpy(1.,samp_f.vector())
                    u_sd.axpy(1.,samp_f.vector()*samp_f.vector())
                t_cnvt=timeit.default_timer()-t_start
                u_m=u_m.get_local()/num_samp
                u_sd=np.sqrt(u_sd.get_local()/num_samp-u_m**2)
            else:
                u_m=np.zeros(elliptic.prior.dim)
                u_sd=np.zeros(elliptic.prior.dim)
        for f_i in pckl_files:
            if '_'+algs[i]+'_' in f_i:
                try:
                    f=open(os.path.join(folder,f_i),'rb')
                    f_read=pickle.load(f)
                    _time[i]=f_read[5+('AE-'in algs[i])]
                    if algs[i]=='AE-pCN': _time[i]+=t_cnvt # adjust time for converting samples
                    f.close()
                    print(f_i+' has been read!')
                    break
                except:
                    pass
    _mean[i]=[u_m.min(),np.median(u_m),u_m.max()]
    _std[i]=[u_sd.min(),np.median(u_sd),u_sd.max()]
#     _incl[i]=np.mean((u_m-norm.ppf(1-.1/2)*u_sd<=MAP)&(MAP<=u_m+norm.ppf(1-.1/2)*u_sd))
    _incl[i]=np.mean((u_m-norm.ppf(1-.1/2)*u_sd<=MCMC_est)&(MCMC_est<=u_m+norm.ppf(1-.1/2)*u_sd))

_mean_str=[np.array2string(r,precision=2,separator=',').replace('[','').replace(']','') for r in _mean]
_std_str=[np.array2string(r,precision=2,separator=',').replace('[','').replace(']','') for r in _std]

sumry_np=np.array([algs,_time,_mean_str,_std_str,_incl]).T
sumry_header=('Method','time(s)','$\\mu$ (min,med,max)','$\\sigma$ (min,med,max)','90\% inclusion')

# save results
np.savetxt(os.path.join(folder,'summary.txt'),sumry_np,fmt="%s",delimiter=',',header=','.join(sumry_header))
try:
    import pandas as pd
    sumry_pd=pd.DataFrame(data=sumry_np,columns=sumry_header)
    sumry_pd.to_csv(os.path.join(folder,'summary.csv'),index=False,header=sumry_header)
except:
    pass