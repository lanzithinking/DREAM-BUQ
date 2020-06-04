"""
Plot estimates of uncertainty field u in Elliptic inverse problem.
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
npz_files=[f for f in os.listdir(folder) if f.endswith('.npz')]
pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
num_samp=10000
# plot ensembles
import matplotlib.pyplot as plt
import matplotlib as mp
from util import matplot4dolfin
matplot=matplot4dolfin()

num_rows=1
fig,axes = plt.subplots(nrows=num_rows,ncols=np.int(np.ceil(num_algs/num_rows)),sharex=True,sharey=True,figsize=(14,3.5))
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
#     if i==0:
#         # plot MAP
#         try:
#             f=df.HDF5File(elliptic.pde.mpi_comm, os.path.join('./result',"MAP_SNR"+str(SNR)+".h5"), "r")
#             MAP=df.Function(elliptic.pde.V,name="parameter")
#             f.read(MAP,"parameter")
#             f.close()
#             sub_fig=matplot.plot(MAP)
# #             sub_fig=df.plot(MAP)
#             ax.set_title('MAP')
#         except:
#             pass
    if i==0:
        # plot vb estimate
        found=False
        u_f=df.Function(elliptic.pde.V,name="parameter")
        for f_i in npz_files:
            if algs[i]+'_' in f_i:
                try:
                    f=np.load(file=os.path.join(folder,f_i))
                    print(f_i+' has been read!')
                    u_vb=f['mean']
                    found=True
                except:
                    pass
        if found:
            u_f.vector().set_local(u_vb)
            sub_fig=matplot.plot(u_f)
            ax.set_title(algs[i])
    elif i>0:
        # plot posterior estimate
        samp_f=df.Function(elliptic.pde.V,name="parameter")
        samp_v=elliptic.prior.gen_vector(); samp_v.zero()
        if algs[i]=='pCN':
            found=False
            num_read=0
            for f_i in hdf5_files:
                if '_'+algs[i]+'_' in f_i:
                    try:
                        f=df.HDF5File(elliptic.pde.mpi_comm,os.path.join(folder,f_i),"r")
                        samp_v.zero()
                        for s in range(num_samp):
                            f.read(samp_f,'sample_{0}'.format(s))
                            samp_v.axpy(1.,samp_f.vector())
                            num_read+=1
                        f.close()
                        print(f_i+' has been read!')
                        found=True
                    except:
                        pass
            if found:
                samp_mean=samp_v/num_read
                samp_f.vector().set_local(samp_mean)
                sub_fig=matplot.plot(samp_f)
                ax.set_title(algs[i])
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
                samp_v.zero()
#                 for s in range(num_samp):
#                     samp_f.vector().set_local(elliptic.prior.v2u(elliptic.prior.gen_vector(samp_[s])))
#                     samp_v.axpy(1.,samp_f.vector())
#                 samp_mean=samp_v/num_samp
#                 samp_v.set_local(elliptic.prior.v2u(elliptic.prior.gen_vector(np.mean(samp_,axis=0))))
                samp_v.set_local(elliptic.prior.C_act(np.mean(samp_,axis=0),.5))
                samp_mean=samp_v
                samp_f.vector().set_local(samp_mean)
                sub_fig=matplot.plot(samp_f)
                ax.set_title(algs[i])
    ax.set_aspect('auto')
plt.axis([0, 1, 0, 1])

# set color bar
cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
cbar=plt.colorbar(sub_fig, cax=cax,**kw)
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()
# fig.colorbar(sub_fig, ax=axes.ravel().tolist(), shrink=0.42)

# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(folder+'/point_estimates.png',bbox_inches='tight')
# plt.show()