"""
Main function to run (inf-)geometric MCMC for EIT inverse problem
Shiwei Lan @ Caltech, 2016
--------------------------
Modified Aug. 2020 @ ASU
"""

# modules
import os,argparse,pickle
import numpy as np

# the inverse problem
from EIT import EIT

# MCMC
import sys
sys.path.append( "../" )
from sampler.geoinfMC import geoinfMC

np.set_printoptions(precision=3, suppress=True)
np.random.seed(2020)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('num_samp', nargs='?', type=int, default=5000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=1000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.05,.1,.05,None,None])
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC'))
    args = parser.parse_args()

    ## define the EIT inverse problem ##
    n_el = 16
    bbox = [[-1,-1],[1,1]]
    meshsz = .04
    el_dist, step = 1, 1
    anomaly = [{'x': 0.4, 'y': 0.4, 'd': 0.2, 'perm': 10},
               {'x': -0.4, 'y': -0.4, 'd': 0.2, 'perm': 0.1}]
    nz_var=1e-2; lamb=1e-1; rho=.25
    eit=EIT(n_el=n_el,bbox=bbox,meshsz=meshsz,el_dist=el_dist,step=step,anomaly=anomaly,nz_var=nz_var,lamb=lamb,rho=rho)
    y=eit.obs
    
    # initialization
    u0=eit.prior['sample']()
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))
    
    inf_GMC=geoinfMC(u0,eit,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO],force_posperm=True)
    mc_fun=inf_GMC.sample
    mc_args=(args.num_samp,args.num_burnin)
    mc_fun(*mc_args)
    
    # append PDE information including the count of solving
    filename_=os.path.join(inf_GMC.savepath,inf_GMC.filename+'.pckl')
    filename=os.path.join(inf_GMC.savepath,'EIT_'+inf_GMC.filename+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
    soln_count=eit.soln_count
    pickle.dump([n_el,bbox,meshsz,el_dist,step,anomaly,y,soln_count,args],f)
    f.close()

if __name__ == '__main__':
    main()
