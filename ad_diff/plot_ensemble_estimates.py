"""
Plot Ensemble Kalman Methods for Advection-Diffusion inverse problem

"""

# import modules
import numpy as np
import dolfin as df
from advdiff import advdiff
import sys
sys.path.append( "../" )
from util.multivector import *
# from optimizer.EnK_dolfin import *
from util.common_colorbar import common_colorbar

seed=2020
## define the inverse problem ##
meshsz = (51,51)
eldeg = 2
rel_noise = .5
nref = 1
adif = advdiff(mesh=meshsz, eldeg=eldeg, rel_noise=rel_noise, nref=nref, seed=seed)
adif.prior.V=adif.prior.Vh
adif.misfit.obs=np.array([dat.get_local() for dat in adif.misfit.d.data]).flatten()

# # initialization
ensbl_sz=500
# unknown=MultiVector(adif.prior.gen_vector(),args.ensemble_size)
# for j in range(args.ensemble_size): unknown[j].set_local(adif.prior.sample(whiten=False))
# # define parameters needed
# G = lambda u, IP=adif: np.array([dat.get_local() for dat in IP.misfit.get_observations(pde=IP.pde, init=u).data]).flatten()
# 
# y=np.array([dat.get_local() for dat in adif.misfit.d.data]).flatten()
# data={'obs':y,'size':y.size,'cov':adif.misfit.noise_variance*np.eye(y.size)}
# 
# # parameters
# stp_sz=[1,.01]
# nz_lvl=1
# err_thld=1e-1
algs=['EKI','EKS']
num_algs=len(algs)
max_iter=10

# #### EKI ####
# eki=EnK(unknown,G,data,adif.prior,stp_sz=stp_sz[0],nz_lvl=nz_lvl,err_thld=err_thld,alg=algs[0],reg=True)
# # run ensemble Kalman algorithm
# res_eki=eki.run(max_iter=max_iter)
    
# #### EKS ####
# eks=EnK(unknown,G,data,adif.prior,stp_sz=stp_sz[1],nz_lvl=nz_lvl,err_thld=err_thld,alg=algs[1],adpt=True)
# # run ensemble Kalman algorithm
# res_eks=eks.run(max_iter=max_iter)

PLOT=True
if PLOT:
    import matplotlib.pyplot as plt
    plt.rcParams['image.cmap'] = 'jet'
    fig,axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12,6), facecolor='white')
    plt.ion()
import os
folder = os.path.join(os.getcwd(),'analysis_eldeg'+str(eldeg))
# obtain estimates
mean_v=MultiVector(adif.prior.gen_vector(),num_algs)
std_v=MultiVector(adif.prior.gen_vector(),num_algs)
ensbl_f=df.Function(adif.prior.V)
if os.path.exists(os.path.join(folder,'enk_mean'+'_ensbl'+str(ensbl_sz)+'.h5')) and os.path.exists(os.path.join(folder,'enk_std'+'_ensbl'+str(ensbl_sz)+'.h5')):
    with df.HDF5File(adif.mpi_comm,os.path.join(folder,'enk_mean'+'_ensbl'+str(ensbl_sz)+'.h5'),"r") as f:
        for a in range(num_algs):
            f.read(ensbl_f,algs[a])
            mean_v[a].set_local(ensbl_f.vector())
    with df.HDF5File(adif.mpi_comm,os.path.join(folder,'enk_std'+'_ensbl'+str(ensbl_sz)+'.h5'),"r") as f:
        for a in range(num_algs):
            f.read(ensbl_f,algs[a])
            std_v[a].set_local(ensbl_f.vector())
else:
    fnames=[f for f in os.listdir(folder) if f.endswith('.h5')]
    num_ensbls=max_iter*ensbl_sz
    prog=np.ceil(num_ensbls*(.1+np.arange(0,1,.1)))
    for a in range(num_algs):
        ustd_fname=algs[a]+'_ustd'+'_ensbl'+str(ensbl_sz)+'_dim'+str(adif.prior.V.dim())
        u_std=df.HDF5File(adif.mpi_comm,os.path.join(folder,ustd_fname+'.h5'),"w")
        print('Working on '+algs[a]+' algorithm...')
        # calculate ensemble estimates
        found=False
        ensbl_mean=adif.prior.gen_vector(); ensbl_mean.zero()
        ensbl_std=adif.prior.gen_vector(); ensbl_std.zero()
        num_read=0
        for f_i in fnames:
            if algs[a]+'_ensbl'+str(ensbl_sz)+'_' in f_i:
                try:
                    f=df.HDF5File(adif.mpi_comm,os.path.join(folder,f_i),"r")
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
                        ensbl_std_n=np.sqrt(np.abs((ensbl_std - ensbl_mean*ensbl_mean).get_local()))
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
    with df.HDF5File(adif.mpi_comm,os.path.join(folder,'enk_mean'+'_ensbl'+str(ensbl_sz)+'.h5'),"w") as f:
        for a in range(num_algs):
            ensbl_f.vector().set_local(mean_v[a])
            f.write(ensbl_f,algs[a])
    with df.HDF5File(adif.mpi_comm,os.path.join(folder,'enk_std'+'_ensbl'+str(ensbl_sz)+'.h5'),"w") as f:
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
sub_figs = [None]*len(axes.flat)
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    if i==0:
        # plot MAP
        try:
            f=df.XDMFFile(adif.mpi_comm, os.path.join(os.getcwd(),'results/MAP.xdmf'))
            MAP=df.Function(adif.prior.V,name="MAP")
            f.read_checkpoint(MAP,'m',0)
            f.close()
            sub_figs[i]=df.plot(MAP)
            fig.colorbar(sub_figs[i],ax=ax)
            ax.set_title('MAP')
#             f=df.HDF5File(adif.mpi_comm, os.path.join(folder,"longrun_mcmc_mean.h5"), "r")
#             PtEst=df.Function(adif.prior.V,name="parameter")
#             f.read(PtEst,"infHMC")
#             f.close()
#             sub_figs[i]=df.plot(PtEst)
#             fig.colorbar(sub_figs[i],ax=ax)
#             ax.set_title('Long-run MCMC')
        except Exception as err:
            print(err)
            pass
    elif 1<=i<=num_algs:
        # plot ensemble estimate
        found=False
        u_est=df.Function(adif.prior.V)
        for f_i in fnames:
            if algs[i-1]+'_uest'+'_ensbl'+str(ensbl_sz) in f_i:
                try:
                    f=df.HDF5File(adif.mpi_comm,os.path.join(folder,f_i),"r")
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
            sub_figs[i]=df.plot(u_est)
            fig.colorbar(sub_figs[i],ax=ax)
        ax.set_title(algs[i-1])
    ax.set_aspect('auto')
plt.axis([0, 1, 0, 1])
# fig=common_colorbar(fig,axes,sub_figs)
plt.subplots_adjust(wspace=0.1, hspace=0)
# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(folder+'/ensemble_estimates_mean'+'_ensbl'+str(ensbl_sz)+'.png',bbox_inches='tight')
# plt.show()

# ensemble std
num_rows=1
fig,axes = plt.subplots(nrows=num_rows,ncols=np.int(np.ceil((1+num_algs)/num_rows)),sharex=True,sharey=True,figsize=(16,4))
sub_figs = [None]*len(axes.flat)
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    if i==0:
        # plot uq
        try:
            f=df.HDF5File(adif.mpi_comm, os.path.join(folder,"longrun_mcmc_std.h5"), "r")
            UQ=df.Function(adif.prior.V,name="parameter")
            f.read(UQ,"infHMC")
            f.close()
            sub_figs[i]=df.plot(UQ)
            fig.colorbar(sub_figs[i],ax=ax)
            ax.set_title('Long-run MCMC')
        except Exception as err:
            print(err)
            pass
    elif 1<=i<=num_algs:
        # plot ensemble estimate
        found=False
        u_std=df.Function(adif.prior.V)
        for f_i in fnames:
            if algs[i-1]+'_ustd'+'_ensbl'+str(ensbl_sz) in f_i:
                try:
                    f=df.HDF5File(adif.mpi_comm,os.path.join(folder,f_i),"r")
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
            sub_figs[i]=df.plot(u_std)
            fig.colorbar(sub_figs[i],ax=ax)
        ax.set_title(algs[i-1])
    ax.set_aspect('auto')
plt.axis([0, 1, 0, 1])
# fig=common_colorbar(fig,axes,sub_figs)
plt.subplots_adjust(wspace=0.1, hspace=0)
# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(folder+'/ensemble_estimates_std'+'_ensbl'+str(ensbl_sz)+'.png',bbox_inches='tight')
# plt.show()