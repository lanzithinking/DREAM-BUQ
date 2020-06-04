"""
Get samples stored in hdf5 file
"""

import os
import numpy as np
import dolfin as df
# the inverse problem
from Elliptic import Elliptic

def restore_sample(mpi_comm,V,dir_name,f_name,num_samp):
    f=df.HDF5File(mpi_comm,os.path.join(dir_name,f_name),"r")
    samp_f=df.Function(V,name="parameter")
    samp=np.zeros((num_samp,V.dim()))
    prog=np.ceil(num_samp*(.1+np.arange(0,1,.1)))
    for s in range(num_samp):
        f.read(samp_f,'sample_{0}'.format(s))
        samp[s,]=samp_f.vector()
        if s+1 in prog:
            print('{0:.0f}% samples have been restored.'.format(np.float(s+1)/num_samp*100))
#     f_read=lambda s: restore_each_sample(f,samp_f,s)
#     samp=Parallel(n_jobs=4)(delayed(f_read)(i) for i in range(num_samp))
    f.close()
    return samp

## define the inverse elliptic problem ##
# parameters for PDE model
nx=40;ny=40;
# parameters for prior model
sigma=1.25;s=0.0625
# parameters for misfit model
SNR=100 # 100
# define the inverse problem
elliptic=Elliptic(nx=nx,ny=ny,SNR=SNR,sigma=sigma,s=s)

# obtain training/testing samples
folder='./analysis_f_SNR'+str(SNR)
fnames=[f for f in os.listdir(folder) if f.endswith('.h5')]
num_samp=2000
found=False
for f_i in fnames:
    if '_pCN_' in f_i:
        try:
            samp=restore_sample(elliptic.pde.mpi_comm,elliptic.pde.V,folder,f_i,num_samp)
            print(f_i+' has been read!')
            found=True; break
        except Exception as err:
            print(err)
            pass
if found:
    np.savez_compressed(file=os.path.join(folder,'training'),X=samp)
    # how to load
#     loaded=np.load(file=os.path.join(folder,'training.npz'))
#     X=loaded['X']