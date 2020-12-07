"""
Plot KL divergence D_KL(posterior||prior) as function of time
Shiwei Lan @ ASU, 2020
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from itertools import cycle

# algorithms
algs=('pCN','infMALA','infHMC','epCN','einfMALA','einfHMC','DREAMpCN','DREAMinfMALA','DREAMinfHMC')
alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','e-pCN','e-$\infty$-MALA','e-$\infty$-HMC','DREAM-pCN','DREAM-$\infty$-MALA','DREAM-$\infty$-HMC')
num_algs=len(algs)
found = np.zeros(num_algs,dtype=np.bool)
# preparation for estimates
eldeg=1
folder = './analysis_eldeg'+str(eldeg)
fnames=[f for f in os.listdir(folder) if f.endswith('.pckl')]

max_iter=100
max_time=10
num_samp=5000

# plot data-misfit
fig,axes = plt.subplots(num=0,nrows=2,figsize=(12,8))
lines = ["-","-.",":"]
linecycler0 = cycle(lines); linecycler1 = cycle(lines);

for a in range(num_algs):
    for f_i in fnames:
        if '_'+algs[a]+'_' in f_i:
            try:
                f=open(os.path.join(folder,f_i),'rb')
                f_read=pickle.load(f,encoding='bytes')
                loglik=f_read[3]
                KL=np.cumsum(loglik)/range(1,1+len(loglik))
                time=f_read[-2]
#                 times=f_read[-1]
                f.close()
                print(f_i+' has been read!')
                found[a]=True
            except:
                pass
    if found[a]:
        axes[0].semilogy(range(max_iter),-KL[:max_iter],next(linecycler0),linewidth=1.25)
        spiter=time/num_samp
        nsamp_in=np.int(np.floor(max_time/spiter))
        axes[1].semilogy(np.linspace(0,max_time,num=nsamp_in),-KL[:nsamp_in],next(linecycler1),linewidth=1.25)
#         plt_idx=times<=max_time
#         axes[1].semilogy(times[plt_idx],-KL[plt_idx],next(linecycler1),linewidth=1.25)

plt.axes(axes[0])
plt.axis('tight')
plt.xlabel('iteration',fontsize=14); plt.ylabel('KL-divergence',fontsize=14)
plt.axes(axes[1])
# plt.axis([0,100,-1,1])
plt.axis('tight')
plt.xlabel('time (seconds)',fontsize=14); plt.ylabel('KL-divergence',fontsize=14)
# plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
plt.subplots_adjust(wspace=0, hspace=0.3)
# add legend
h2_pos=axes[1].get_position()
plt.legend(np.array(alg_names)[found],fontsize=11,loc=2,bbox_to_anchor=(1.01,2.33),labelspacing=3.8)
# fig.tight_layout(rect=[0,0,.85,1])
plt.savefig(folder+'/KL_t.png',bbox_inches='tight')

# plt.show()
