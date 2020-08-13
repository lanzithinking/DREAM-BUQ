"""
Main function to run ensemble methods for EIT inverse problem
Shiwei Lan @ ASU, 2020
"""

# modules
import os,argparse,pickle
import numpy as np

# the inverse problem
from EIT import EIT

# MCMC
import sys
sys.path.append( "../" )
from optimizer.EnK import *

np.set_printoptions(precision=3, suppress=True)
np.random.seed(2020)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=1)
    parser.add_argument('ensemble_size', nargs='?', type=int, default=100)
    parser.add_argument('max_iter', nargs='?', type=int, default=50)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[1.,.1]) # SNR10: [1,.01];SNR100: [1,.01]
    parser.add_argument('algs', nargs='?', type=str, default=('EKI','EKS'))
    args = parser.parse_args()

    ## define the EIT inverse problem ##
    n_el = 16
    bbox = [[-1,-1],[1,1]]
    meshsz = .05
    el_dist, step = 1, 1
    anomaly = [{'x': 0.4, 'y': 0.4, 'd': 0.2, 'perm': 10},
               {'x': -0.4, 'y': -0.4, 'd': 0.2, 'perm': 0.1}]
    eit=EIT(n_el=n_el,bbox=bbox,meshsz=meshsz,el_dist=el_dist,step=step,anomaly=anomaly,lamb=1)
    
    # initialization
    u0=eit.prior['sample'](num_samp=args.ensemble_size)
    G=lambda u:eit.forward(u,n_jobs=5)
    y=eit.obs
    data={'obs':y,'size':y.size,'cov':np.diag(eit.nz_var)}
    prior=eit.prior
    
    # EnK parameters
    nz_lvl=1
    err_thld=1e-1
    
    # run EnK to generate ensembles
    print("Preparing %s with step size %g ..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO]))
    ek=EnK(u0,G,data,prior,stp_sz=args.step_sizes[args.algNO],nz_lvl=nz_lvl,err_thld=err_thld,alg=args.algs[args.algNO],reg=True,adpt=True)
    ek_fun=ek.run
    ek_args=(args.max_iter,True)
    savepath,filename=ek_fun(*ek_args)
    
    # append extra information including the count of solving
    filename_=os.path.join(savepath,filename+'.pckl')
    filename=os.path.join(savepath,'EIT_'+filename+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
    pickle.dump([n_el,bbox,meshsz,el_dist,step,anomaly,y,args],f)
    f.close()

if __name__ == '__main__':
    main()
