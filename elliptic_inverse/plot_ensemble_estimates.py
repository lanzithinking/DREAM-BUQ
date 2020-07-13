"""
Test Ensemble Kalman Methods for Elliptic Inverse Problems

"""

# import modules
import numpy as np
import dolfin as df
from Elliptic import Elliptic
import sys
sys.path.append( "../" )
from util.multivector import *
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
ensbl_sz=500
# unknown=MultiVector(elliptic.prior.gen_vector(),ensbl_sz)
# for j in range(ensbl_sz): unknown[j].set_local(elliptic.prior.sample(whiten=False))
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

PLOT=True
if PLOT:
    import matplotlib.pyplot as plt
    plt.rcParams['image.cmap'] = 'jet'
    fig,axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12,6), facecolor='white')
    plt.ion()
import os
folder = os.path.join(os.getcwd(),'analysis_f_SNR'+str(SNR))
fnames=[f for f in os.listdir(folder) if f.endswith('.h5')]
num_ensbls=max_iter*ensbl_sz
prog=np.ceil(num_ensbls*(.1+np.arange(0,1,.1)))
# obtain estimates
mean_v=MultiVector(elliptic.prior.gen_vector(),num_algs)
std_v=MultiVector(elliptic.prior.gen_vector(),num_algs)
ensbl_f=df.Function(elliptic.pde.V)
if os.path.exists(os.path.join(folder,'enk_mean'+'_ensbl'+str(ensbl_sz)+'.h5')) and os.path.exists(os.path.join(folder,'enk_std'+'_ensbl'+str(ensbl_sz)+'.h5')):
    with df.HDF5File(elliptic.pde.mpi_comm,os.path.join(folder,'enk_mean'+'_ensbl'+str(ensbl_sz)+'.h5'),"r") as f:
        for a in range(num_algs):
            f.read(ensbl_f,algs[a])
            mean_v[a].set_local(ensbl_f.vector())
    with df.HDF5File(elliptic.pde.mpi_comm,os.path.join(folder,'enk_std'+'_ensbl'+str(ensbl_sz)+'.h5'),"r") as f:
        for a in range(num_algs):
            f.read(ensbl_f,algs[a])
            std_v[a].set_local(ensbl_f.vector())
else:
    for a in range(num_algs):
        ustd_fname=algs[a]+'_ustd'+'_ensbl'+str(ensbl_sz)+'_dim'+str(elliptic.prior.dim)
        u_std=df.HDF5File(elliptic.pde.mpi_comm,os.path.join(folder,ustd_fname+'.h5'),"w")
        print('Working on '+algs[a]+' algorithm...')
        # calculate ensemble estimates
        found=False
        ensbl_mean=elliptic.prior.gen_vector(); ensbl_mean.zero()
        ensbl_std=elliptic.prior.gen_vector(); ensbl_std.zero()
        num_read=0
        for f_i in fnames:
            if algs[a]+'_ensbl'+str(ensbl_sz)+'_' in f_i:
                try:
                    f=df.HDF5File(elliptic.pde.mpi_comm,os.path.join(folder,f_i),"r")
                    for n in range(max_iter):
                        ensbl_mean.zero(); ensbl_std.zero(); num_read=0
                        for j in range(ensbl_sz):
                            f.read(ensbl_f,'iter{0}_ensbl{1}'.format(n+1,j))
                            u=ensbl_f.vector()
                            ensbl_mean.axpy(1.,u)
                            ensbl_std.axpy(1.,u*u)
                            num_read+=1
                            s=n*ensbl_sz+j
                            if s+1 in prog:
                                print('{0:.0f}% ensembles have been retrieved.'.format(np.float(s+1)/num_ensbls*100))
                        ensbl_mean=ensbl_mean/num_read; ensbl_std=ensbl_std/num_read
                        ensbl_std_n=np.sqrt((ensbl_std - ensbl_mean*ensbl_mean).get_local())
                        ensbl_f.vector().set_local(ensbl_std_n)
                        u_std.write(ensbl_f,'iter{0}'.format(n))
                        if PLOT:
                            plt.clf()
                            ax=axes.flat[0]
                            plt.axes(ax)
                            ensbl_f.vector().set_local(ensbl_mean)
                            subfig=df.plot(ensbl_f)
                            plt.title(algs[a]+' Mean (iter='+str(n+1)+')')
                            cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
                            fig.colorbar(subfig, cax=cax)
                            
                            ax=axes.flat[1]
                            plt.axes(ax)
                            ensbl_f.vector().set_local(ensbl_std_n)
                            subfig=df.plot(ensbl_f)
                            plt.title(algs[a]+' STD (iter='+str(n+1)+')')
                            cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
                            fig.colorbar(subfig, cax=cax)
                            plt.draw()
                            plt.pause(1.0/10.0)
                    f.close()
                    print(f_i+' has been read!')
                    found=True; break
                except:
                    pass
        u_std.close()
        if found:
            mean_v[a].set_local(ensbl_mean)
            std_v[a].set_local(ensbl_std_n)
    # save
    with df.HDF5File(elliptic.pde.mpi_comm,os.path.join(folder,'enk_mean'+'_ensbl'+str(ensbl_sz)+'.h5'),"w") as f:
        for a in range(num_algs):
            ensbl_f.vector().set_local(mean_v[a])
            f.write(ensbl_f,algs[a])
    with df.HDF5File(elliptic.pde.mpi_comm,os.path.join(folder,'enk_std'+'_ensbl'+str(ensbl_sz)+'.h5'),"w") as f:
        for a in range(num_algs):
            ensbl_f.vector().set_local(std_v[a])
            f.write(ensbl_f,algs[a])

# plot
fnames=[f for f in os.listdir(folder) if f.endswith('.h5')]
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'
# ensemble mean
num_rows=1
fig,axes = plt.subplots(nrows=num_rows,ncols=np.int(np.ceil((1+num_algs)/num_rows)),sharex=True,sharey=True,figsize=(16,4))
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    if i==0:
        # plot MAP
        try:
            f=df.HDF5File(elliptic.pde.mpi_comm, os.path.join('./result',"MAP_SNR"+str(SNR)+".h5"), "r")
            MAP=df.Function(elliptic.pde.V,name="parameter")
            f.read(MAP,"parameter")
            f.close()
            sub_fig=df.plot(MAP)
            fig.colorbar(sub_fig,ax=ax)
            ax.set_title('MAP')
        except Exception as err:
            print(err)
            pass
    elif 1<=i<=num_algs:
        # plot ensemble estimate
        found=False
        u_est=df.Function(elliptic.pde.V)
        for f_i in fnames:
            if algs[i-1]+'_uest'+'_ensbl'+str(ensbl_sz) in f_i:
                try:
                    f=df.HDF5File(elliptic.pde.mpi_comm,os.path.join(folder,f_i),"r")
#                     n={'0':res_eki[2],'1':res_eks[2]}[i-1]
                    n=max_iter-1
                    f.read(u_est,'iter{0}'.format(n))
                    f.close()
                    print(f_i+' has been read!')
                    found=True
                except Exception as err:
                    print(err)
                    pass
        if found:
            sub_fig=df.plot(u_est)
            fig.colorbar(sub_fig,ax=ax)
        ax.set_title(algs[i-1])
    ax.set_aspect('auto')
plt.axis([0, 1, 0, 1])
plt.subplots_adjust(wspace=0.1, hspace=0)
# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(folder+'/ensemble_estimates_mean'+'_ensbl'+str(ensbl_sz)+'.png',bbox_inches='tight')
# plt.show()

# ensemble std
num_rows=1
fig,axes = plt.subplots(nrows=num_rows,ncols=np.int(np.ceil((1+num_algs)/num_rows)),sharex=True,sharey=True,figsize=(16,4))
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    if i==0:
        # plot uq
        try:
            f=df.HDF5File(elliptic.pde.mpi_comm, os.path.join(folder,"mcmc_std.h5"), "r")
            UQ=df.Function(elliptic.pde.V,name="parameter")
            f.read(UQ,"infHMC")
            f.close()
            sub_fig=df.plot(UQ)
            fig.colorbar(sub_fig,ax=ax)
            ax.set_title('$\infty$-HMC')
        except Exception as err:
            print(err)
            pass
    elif 1<=i<=num_algs:
        # plot ensemble estimate
        found=False
        u_std=df.Function(elliptic.pde.V)
        for f_i in fnames:
            if algs[i-1]+'_ustd'+'_ensbl'+str(ensbl_sz) in f_i:
                try:
                    f=df.HDF5File(elliptic.pde.mpi_comm,os.path.join(folder,f_i),"r")
#                     n={'0':res_eki[2],'1':res_eks[2]}[i-1]
                    n=max_iter-1
                    f.read(u_std,'iter{0}'.format(n))
                    f.close()
                    print(f_i+' has been read!')
                    found=True
                except Exception as err:
                    print(err)
                    pass
        if found:
            sub_fig=df.plot(u_std)
            fig.colorbar(sub_fig,ax=ax)
        ax.set_title(algs[i-1])
    ax.set_aspect('auto')
plt.axis([0, 1, 0, 1])
plt.subplots_adjust(wspace=0.1, hspace=0)
# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(folder+'/ensemble_estimates_std'+'_ensbl'+str(ensbl_sz)+'.png',bbox_inches='tight')
# plt.show()