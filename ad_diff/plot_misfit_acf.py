"""
Plot autocorrelation of data-misfits
Shiwei Lan @ U of Warwick, 2016
----------------------------------
Modified for DREAM December 2020 @ ASU
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from itertools import cycle

def autocorr(x):
    """This one is closest to what plt.acorr does.
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    assert np.allclose(r, np.array(
        [(x[:n - k] * x[-(n - k):]).sum() for k in range(n)]))
    result = r / (variance * (np.arange(n, 0, -1)))
    return result

# algorithms
algs=('pCN','infMALA','infHMC','epCN','einfMALA','einfHMC','DREAMpCN','DREAMinfMALA','DREAMinfHMC')
alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','e-pCN','e-$\infty$-MALA','e-$\infty$-HMC','DREAM-pCN','DREAM-$\infty$-MALA','DREAM-$\infty$-HMC')
num_algs=len(algs)
found = np.zeros(num_algs,dtype=np.bool)

eldeg=2
folder = './analysis_eldeg'+str(eldeg)
fnames=[f for f in os.listdir(folder) if f.endswith('.pckl')]

# plot data-misfit
fig,axes = plt.subplots(num=0,ncols=2,figsize=(14,6))
lines = ["-","-.",":"]
linecycler0 = cycle(lines); linecycler1 = cycle(lines);

for a in range(num_algs):
    for f_i in fnames:
        if '_'+algs[a]+'_' in f_i:
            try:
                f=open(os.path.join(folder,f_i),'rb')
                f_read=pickle.load(f,encoding='bytes')
                loglik=f_read[3]
                f.close()
                print(f_i+' has been read!')
                found[a]=True
            except:
                pass
    if found[a]:
        # modify misifits to discern their traceplots
#         misfit=-loglik[500:]-a*3
        misfit=-loglik[500:]-a*10
        axes[0].plot(misfit,next(linecycler0),linewidth=1.25)
        # pd.tools.plotting.autocorrelation_plot(loglik[1000:], ax=axes[1],linestyle=next(linecycler))
        acorr_misfit=autocorr(misfit)
#         axes[1].plot(range(1,21),acorr_misfit[:20],next(linecycler1),linewidth=1.25)
#         plt.xticks(np.arange(2,21,2),np.arange(2,21,2))
        axes[1].plot(range(1,51),acorr_misfit[:50],next(linecycler1),linewidth=1.25)
        plt.xticks(np.arange(5,51,5),np.arange(5,51,5))
        
        plt.axhline(y=0.0, color='r', linestyle='-')

plt.axes(axes[0])
plt.axis('tight')
plt.xlabel('iteration',fontsize=14); plt.ylabel('data-misfit (offset)',fontsize=14)
plt.legend(np.array(alg_names)[found],fontsize=11,loc=3,ncol=3,bbox_to_anchor=(0.,1.02,2.2,0.102),mode="expand", borderaxespad=0.)
plt.axes(axes[1])
# plt.axis([0,100,-1,1])
plt.axis('tight')
plt.xlabel('lag',fontsize=14); plt.ylabel('auto-correlation',fontsize=14)
# fig.tight_layout(rect=[0,0,1,.9])
plt.savefig(folder+'/misfit_acf.png',bbox_inches='tight')

# plt.show()
