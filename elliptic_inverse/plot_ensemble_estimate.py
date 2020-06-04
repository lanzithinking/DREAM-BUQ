"""
Test Ensemble Kalman Methods for Elliptic Inverse Problems

"""

# import modules
import numpy as np
import dolfin as df
from Elliptic import Elliptic
import sys
sys.path.append( "../" )
# from util.multivector import *
# from optimizer.EnK_dolfin import *

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

# # initialization
# J=10
# unknown=MultiVector(elliptic.prior.gen_vector(),J)
# for j in range(J): unknown[j].set_local(elliptic.prior.sample(whiten=False))
# # define parameters needed
# def G(u,IP=elliptic):
#     u_f=df.Function(IP.prior.V)
#     u_f.vector().zero()
#     u_f.vector().axpy(1.,u)
#     IP.pde.set_forms(unknown=u_f)
#     return IP.misfit._extr_soloc(IP.pde.soln_fwd()[0])
# 
# y=elliptic.misfit.obs
# data={'obs':y,'size':y.size,'cov':1./elliptic.misfit.prec*np.eye(y.size)}
# 
# # parameters
# stp_sz=[1,.01]
# nz_lvl=1
# err_thld=1e-1
algs=['EKI','EKS']
num_algs=len(algs)
max_iter=10

# #### EKI ####
# eki=EnK(unknown,G,data,elliptic.prior,stp_sz=stp_sz[0],nz_lvl=nz_lvl,err_thld=err_thld,alg=algs[0],reg=True)
# # run ensemble Kalman algorithm
# res_eki=eki.run(max_iter=max_iter)
    
# #### EKS ####
# eks=EnK(unknown,G,data,elliptic.prior,stp_sz=stp_sz[1],nz_lvl=nz_lvl,err_thld=err_thld,alg=algs[1],adpt=True)
# # run ensemble Kalman algorithm
# res_eks=eks.run(max_iter=max_iter)

import os
folder = os.path.join(os.getcwd(),'analysis_f_SNR'+str(SNR))
fnames=[f for f in os.listdir(folder) if f.endswith('.h5')]
# plot ensembles
import matplotlib.pyplot as plt
import matplotlib as mp
from util import matplot4dolfin
matplot=matplot4dolfin()

num_rows=1
fig,axes = plt.subplots(nrows=num_rows,ncols=np.int(np.ceil((1+num_algs)/num_rows)),sharex=True,sharey=True,figsize=(12,4))
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    if i==0:
        # plot MAP
        try:
            f=df.HDF5File(elliptic.pde.mpi_comm, os.path.join('./result',"MAP_SNR"+str(SNR)+".h5"), "r")
            MAP=df.Function(elliptic.pde.V,name="parameter")
            f.read(MAP,"parameter")
            f.close()
            sub_fig=matplot.plot(MAP)
#             sub_fig=df.plot(MAP)
            ax.set_title('MAP')
        except:
            pass
    elif 1<=i<=num_algs:
        # plot ensemble estimate
        found=False
        u_est=df.Function(elliptic.pde.V)
        for f_i in fnames:
            if algs[i-1]+'_uest_' in f_i:
                try:
                    f=df.HDF5File(elliptic.pde.mpi_comm,os.path.join(folder,f_i),"r")
#                     n={'0':res_eki[2],'1':res_eks[2]}[i-1]
                    n=max_iter-1
                    f.read(u_est,'iter{0}'.format(n))
                    f.close()
                    print(f_i+' has been read!')
                    found=True
                except:
                    pass
        if found:
            sub_fig=matplot.plot(u_est)
#             sub_fig=df.plot(u_est)
        ax.set_title(algs[i-1])
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
plt.savefig(folder+'/ensemble_estimates.png',bbox_inches='tight')
# plt.show()