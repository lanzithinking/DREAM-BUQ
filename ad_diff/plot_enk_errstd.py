"""
Plot error estimates of uncertainty field u in Advection-Diffusion inverse problem.
Shiwei Lan @ ASU, 2020
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
meshsz = (61,61)
eldeg = 1
gamma = 2.; delta = 10.
rel_noise = .5
nref = 1
adif = advdiff(mesh=meshsz, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed)
adif.prior.V=adif.prior.Vh
adif.misfit.obs=np.array([dat.get_local() for dat in adif.misfit.d.data]).flatten()

# # initialization
algs=['EKI','EKS']
num_algs=len(algs)
ensbszs=[50,100,200,500]
num_ensbsz=len(ensbszs)
num_ensbls=5000
max_iter=[np.int(num_ensbls/i) for i in ensbszs]


PLOT=False
if PLOT:
    import matplotlib.pyplot as plt
    plt.rcParams['image.cmap'] = 'jet'
    fig,axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12,6), facecolor='white')
    plt.ion()
import os,pickle
folder = os.path.join(os.getcwd(),'analysis_eldeg'+str(eldeg))
# obtain relative error estimates
relerr=[None,]*num_algs
relstd=[None,]*num_algs
ensbl_f=df.Function(elliptic.pde.V)
if os.path.exists(os.path.join(folder,'enk_relerr.pckl')) and os.path.exists(os.path.join(folder,'enk_relstd.pckl')):
    with open(os.path.join(folder,'enk_relerr.pckl'),"rb") as f:
        relerr=pickle.load(f)
    with open(os.path.join(folder,'enk_relstd.pckl'),"rb") as f:
        relstd=pickle.load(f)
else:
    # load MAP
    try:
#         f=df.XDMFFile(adif.mpi_comm, os.path.join(os.getcwd(),'properties/MAP.xdmf'))
#         MAP=df.Function(adif.prior.V,name="MAP")
#         f.read_checkpoint(MAP,'m',0)
#         f.close()
        f=df.HDF5File(elliptic.pde.mpi_comm, os.path.join(folder,"longrun_mcmc_mean.h5"), "r")
        PtEst=df.Function(elliptic.pde.V,name="parameter")
        f.read(PtEst,"infHMC")
        f.close()
    except Exception as err:
        print(err)
        pass
    # load std by long-run mcmc
    try:
        f=df.HDF5File(elliptic.pde.mpi_comm, os.path.join(folder,"longrun_mcmc_std.h5"), "r")
        UQ=df.Function(elliptic.pde.V,name="parameter")
        f.read(UQ,"infHMC")
        f.close()
    except Exception as err:
        print(err)
        pass
    # obtain estimates
    fnames=[f for f in os.listdir(folder) if f.endswith('.h5')]
    prog=np.ceil(num_ensbls*(.1+np.arange(0,1,.1)))
    for a in range(num_algs):
        relerr[a]=[None,]*num_ensbsz
        relstd[a]=[None,]*num_ensbsz
        for s in range(num_ensbsz):
            relerr[a][s]=np.zeros(max_iter[s])
            relstd[a][s]=np.zeros(max_iter[s])
            print('Working on '+algs[a]+' algorithm for ensemble size '+str(ensbszs[s])+'...')
            # calculate ensemble estimates
            found=False
            ensbl_mean=elliptic.prior.gen_vector(); ensbl_mean.zero()
            ensbl_std=elliptic.prior.gen_vector(); ensbl_std.zero()
            num_read=0
            for f_i in fnames:
                if algs[a]+'_ensbl'+str(ensbszs[s])+'_' in f_i:
                    try:
                        f=df.HDF5File(elliptic.pde.mpi_comm,os.path.join(folder,f_i),"r")
                        for n in range(max_iter[s]):
                            ensbl_mean.zero(); ensbl_std.zero(); num_read=0
                            for j in range(ensbszs[s]):
                                f.read(ensbl_f,'iter{0}_ensbl{1}'.format(n+1,j))
                                u=ensbl_f.vector()
                                ensbl_mean.axpy(1.,u)
                                ensbl_std.axpy(1.,u*u)
                                num_read+=1
                                p=n*ensbszs[s]+j
                                if p+1 in prog:
                                    print('{0:.0f}% ensembles have been retrieved.'.format(np.float(p+1)/num_ensbls*100))
                            ensbl_mean=ensbl_mean/num_read; ensbl_std=ensbl_std/num_read
                            ensbl_std_n=np.sqrt((ensbl_std - ensbl_mean*ensbl_mean).get_local())
#                             relerr[a][s][n]=(MAP.vector()-ensbl_mean).norm('l2')/MAP.vector().norm('l2')
                            relerr[a][s][n]=(PtEst.vector()-ensbl_mean).norm('l2')/PtEst.vector().norm('l2')
                            relstd[a][s][n]=np.mean(ensbl_std_n/UQ.vector().get_local())
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
#             if found:
    # save
    with open(os.path.join(folder,'enk_relerr.pckl'),"wb") as f:
        pickle.dump(relerr,f)
    with open(os.path.join(folder,'enk_relstd.pckl'),"wb") as f:
        pickle.dump(relstd,f)

# plot
fnames=[f for f in os.listdir(folder) if f.endswith('.pckl')]
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'
from itertools import cycle

fig,axes = plt.subplots(num=0,ncols=2,figsize=(12,5))
lines = ["-","-."]
linecycler0 = cycle(lines); linecycler1 = cycle(lines);

for a in range(num_algs):
#     axes[0].semilogy([relerr[a][s].min() for s in range(num_ensbsz)],next(linecycler0),linewidth=1.25)
    axes[0].semilogy([relerr[a][s][-1] for s in range(num_ensbsz)],next(linecycler0),linewidth=1.25)
    axes[0].set_xticks(range(num_ensbsz)); axes[0].set_xticklabels(ensbszs)
#     axes[1].semilogy([relstd[a][s].max() for s in range(num_ensbsz)],next(linecycler1),linewidth=1.25)
    axes[1].semilogy([relstd[a][s][-1] for s in range(num_ensbsz)],next(linecycler1),linewidth=1.25)
    axes[1].set_xticks(range(num_ensbsz)); axes[1].set_xticklabels(ensbszs)
    plt.axhline(y=1.0, color='k', linestyle='--')

plt.axes(axes[0])
plt.axis('tight')
plt.xlabel('ensemble size',fontsize=14); plt.ylabel('Relative Mean Error',fontsize=14)
# plt.legend(np.array(algs),fontsize=11,loc=3,ncol=num_algs,bbox_to_anchor=(0.,1.02,2.2,0.102),mode="expand", borderaxespad=0.)
plt.legend(np.array(algs),fontsize=11,loc='upper right',frameon=False)
plt.axes(axes[1])
# plt.axis([0,100,-1,1])
plt.axis('tight')
plt.xlabel('ensemble size',fontsize=14); plt.ylabel('Relative Standard Deviation',fontsize=14)
# plt.legend(np.array(algs),fontsize=11,loc='lower right',frameon=False)
# plt.axhline(y=1.0, color='k', linestyle='--')
# fig.tight_layout(rect=[0,0,1,.9])
plt.subplots_adjust(wspace=0.3, hspace=0.1)
plt.savefig(folder+'/enk_relerrstd.png',bbox_inches='tight')

# plt.show()
