"""
Plot ensemble estimates for EIT inverse problems
"""

# import modules
import numpy as np
from EIT import EIT
import sys,pickle
sys.path.append( "../" )
from util.common_colorbar import common_colorbar

# np.random.seed(2020)

## define the EIT inverse problem ##
n_el = 16
bbox = [[-1,-1],[1,1]]
meshsz = .05
el_dist, step = 1, 1
anomaly = [{'x': 0.4, 'y': 0.4, 'd': 0.2, 'perm': 10},
           {'x': -0.4, 'y': -0.4, 'd': 0.2, 'perm': 0.1}]
lamb=1e-3
eit=EIT(n_el=n_el,bbox=bbox,meshsz=meshsz,el_dist=el_dist,step=step,anomaly=anomaly,lamb=lamb)

# # initialization
ensbl_sz=100
# unknown=eit.prior['sample'](num_samp=ensbl_sz)
# # define parameters needed
# G=lambda u:eit.forward(u,n_jobs=5)
# y=eit.obs
# data={'obs':y,'size':y.size,'cov':np.eye(y.size)}
# 
# # parameters
# stp_sz=[1,.01]
# nz_lvl=1
# err_thld=1e-1
algs=['EKI','EKS']
num_algs=len(algs)
max_iter=50

# #### EKI ####
# eki=EnK(unknown,G,data,eit.prior,stp_sz=stp_sz[0],nz_lvl=nz_lvl,err_thld=err_thld,alg=algs[0],reg=True)
# # run ensemble Kalman algorithm
# res_eki=eki.run(max_iter=max_iter)
    
# #### EKS ####
# eks=EnK(unknown,G,data,eit.prior,stp_sz=stp_sz[1],nz_lvl=nz_lvl,err_thld=err_thld,alg=algs[1],adpt=True)
# # run ensemble Kalman algorithm
# res_eks=eks.run(max_iter=max_iter)

PLOT=False
if PLOT:
    import matplotlib.pyplot as plt
    plt.rcParams['image.cmap'] = 'jet'
    fig,axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12,6), facecolor='white')
    plt.ion()
import os
folder = os.path.join(os.getcwd(),'analysis')
# obtain estimates
mean_v=np.zeros((num_algs,eit.dim))
std_v=np.zeros((num_algs,eit.dim))
if os.path.exists(os.path.join(folder,'enk_est'+'_ensbl'+str(ensbl_sz)+'.pckl')):
    with open(os.path.join(folder,'enk_est'+'_ensbl'+str(ensbl_sz)+'.pckl'),"rb") as f:
        mean_v,std_v=pickle.load(f)
else:
    fnames=[f for f in os.listdir(folder) if f.endswith('.pckl')]
    prog=np.ceil(max_iter*(.1+np.arange(0,1,.1)))
    for a in range(num_algs):
        print('Working on '+algs[a]+' algorithm...')
        # calculate ensemble estimates
        found=False
        ensbl_mean=np.zeros(eit.dim); ensbl_std=np.zeros(eit.dim)
        for f_i in fnames:
            if algs[a]+'_ensbl'+str(ensbl_sz)+'_' in f_i:
                try:
                    f=open(os.path.join(folder,f_i),'rb')
                    loaded=pickle.load(f)
                    ensbl=loaded[3][:-1,:,:]
#                     ensbl=np.exp(loaded[3][:-1,:,:])
                    for n in range(loaded[-2]+1):
                        if n+1 in prog:
                            print('{0:.0f}% ensembles have been retrieved.'.format(np.float(n+1)/max_iter*100))
                        ensbl_mean=np.mean(ensbl[n],axis=0); ensbl_std=np.std(ensbl[n],axis=0)
                        if PLOT:
                            plt.clf()
                            ax=axes.flat[0]
                            plt.axes(ax)
                            subfig=eit.plot(ensbl_mean,ax=ax)
                            plt.title(algs[a]+' Mean (iter='+str(n+1)+')')
                            cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
                            fig.colorbar(subfig, cax=cax)
                            
                            ax=axes.flat[1]
                            plt.axes(ax)
                            subfig=eit.plot(ensbl_std,ax=ax)
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
        if found:
            mean_v[a]=ensbl_mean
            std_v[a]=ensbl_std
    # save
    with open(os.path.join(folder,'enk_est'+'_ensbl'+str(ensbl_sz)+'.pckl'),"wb") as f:
        pickle.dump([mean_v,std_v],f)

# plot
fnames=[f for f in os.listdir(folder) if f.endswith('.pckl')]
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'
# ensemble mean
num_rows=1
fig,axes = plt.subplots(nrows=num_rows,ncols=np.int(np.ceil((1+num_algs)/num_rows)),sharex=False,sharey=False,figsize=(16,4))
sub_figs = [None]*len(axes.flat)
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    if i==0:
        # plot MAP
        try:
            f=open(os.path.join('./result',str(eit.gdim)+'d_EIT_MAP_dim'+str(eit.dim)+'.pckl'),'rb')
            MAP=pickle.load(f)[0]
            f.close()
            sub_figs[i]=eit.plot(MAP,ax=ax)
            fig.colorbar(sub_figs[i],ax=ax)
            ax.set_title('MAP')
#             f=open(os.path.join(folder,"longrun_mcmc_mean.pckl"), "rb")
#             PtEst=pickle.load(f)[0]
#             f.close()
#             sub_figs[i]=eit.plot(PtEst,ax=ax)
#             fig.colorbar(sub_figs[i],ax=ax)
#             ax.set_title('Long-run MCMC')
        except Exception as err:
            print(err)
            pass
    elif 1<=i<=num_algs:
        # plot ensemble estimate
        sub_figs[i]=eit.plot(mean_v[i-1],ax=ax)
#         found=False
#         for f_i in fnames:
#             if algs[i-1]+'_ensbl'+str(ensbl_sz) in f_i:
#                 try:
#                     f=open(os.path.join(folder,f_i),"rb")
#                     u_est=pickle.load(f)[0]
# #                     n={'0':res_eki[2],'1':res_eks[2]}[i-1]
#                     n=max_iter-1
#                     u_est=u_est[n]
#                     f.close()
#                     print(f_i+' has been read!')
#                     found=True
#                 except Exception as err:
#                     print(err)
#                     pass
#         if found:
#             sub_figs[i]=eit.plot(u_est,ax=ax)
        fig.colorbar(sub_figs[i],ax=ax)
        ax.set_title(algs[i-1])
#     ax.set_aspect('equal')
#     plt.axis([-1, 1, -1, 1])
# fig=common_colorbar(fig,axes,sub_figs)
plt.subplots_adjust(wspace=0.2, hspace=0)
# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(folder+'/ensemble_estimates_mean'+'_ensbl'+str(ensbl_sz)+'.png',bbox_inches='tight')
# plt.show()

# ensemble std
num_rows=1
fig,axes = plt.subplots(nrows=num_rows,ncols=np.int(np.ceil((1+num_algs)/num_rows)),sharex=False,sharey=False,figsize=(16,4))
sub_figs = [None]*len(axes.flat)
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    if i==0:
        # plot uq
        try:
            f=open(os.path.join(folder,"longrun_mcmc_std.pckl"), "rb")
            UQ=pickle.load(f)[1]
            f.close()
            sub_figs[i]=eit.plot(UQ,ax=ax)
            fig.colorbar(sub_figs[i],ax=ax)
            ax.set_title('Long-run MCMC')
        except Exception as err:
            print(err)
            pass
    elif 1<=i<=num_algs:
        # plot ensemble estimate
        sub_figs[i]=eit.plot(std_v[i-1],ax=ax)
        fig.colorbar(sub_figs[i],ax=ax)
        ax.set_title(algs[i-1])
#     ax.set_aspect('equal')
#     plt.axis([-1, 1, -1, 1])
# fig=common_colorbar(fig,axes,sub_figs)
plt.subplots_adjust(wspace=0.2, hspace=0)
# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(folder+'/ensemble_estimates_std'+'_ensbl'+str(ensbl_sz)+'.png',bbox_inches='tight')
# plt.show()