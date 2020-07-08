"""
Main function to run Elliptic PDE model (DILI; Cui et~al, 2016) to generate estimates
Shiwei Lan @ Caltech, 2016
--------------------------
Modified May 2020 @ ASU
"""

# modules
import os,argparse,pickle
import numpy as np
import dolfin as df

# the inverse problem
from Elliptic import Elliptic

# MCMC
import sys
sys.path.append( "../" )
from optimizer.EnK_dolfin import *
from util.multivector import *

np.set_printoptions(precision=3, suppress=True)
np.random.seed(2020)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('ensemble_size', nargs='?', type=int, default=100)
    parser.add_argument('max_iter', nargs='?', type=int, default=50)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[1.,.1]) # SNR10: [1,.01];SNR100: [1,.01]
    parser.add_argument('algs', nargs='?', type=str, default=('EKI','EKS'))
    args = parser.parse_args()

    ## define the inverse elliptic problem ##
    # parameters for PDE model
    nx=40;ny=40;
    # parameters for prior model
    sigma=1.25;s=0.0625
    # parameters for misfit model
    SNR=50 # 100
    # define the inverse problem
    elliptic=Elliptic(nx=nx,ny=ny,SNR=SNR,sigma=sigma,s=s)
    
    # initialization
    unknown=MultiVector(elliptic.prior.gen_vector(),args.ensemble_size)
    for j in range(args.ensemble_size): unknown[j].set_local(elliptic.prior.sample(whiten=False))
    # define parameters needed
    def G(u,IP=elliptic):
        u_f=df.Function(IP.prior.V)
        u_f.vector().zero()
        u_f.vector().axpy(1.,u)
        IP.pde.set_forms(unknown=u_f)
        return IP.misfit._extr_soloc(IP.pde.soln_fwd()[0])
    
    y=elliptic.misfit.obs
    data={'obs':y,'size':y.size,'cov':1./elliptic.misfit.prec*np.eye(y.size)}
    
    # EnK parameters
    nz_lvl=1
    err_thld=1e-1
    
    # run EnK to generate ensembles
    print("Preparing %s with step size %g ..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO]))
    ek=EnK(unknown,G,data,elliptic.prior,stp_sz=args.step_sizes[args.algNO],nz_lvl=nz_lvl,err_thld=err_thld,alg=args.algs[args.algNO],adpt=True)
    ek_fun=ek.run
    ek_args=(args.max_iter,True)
    savepath,filename=ek_fun(*ek_args)
    
    # append PDE information including the count of solving
    filename_=os.path.join(savepath,filename+'.pckl')
    filename=os.path.join(savepath,'Elliptic_'+filename+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
#     soln_count=[elliptic.soln_count,elliptic.pde.soln_count]
    soln_count=elliptic.pde.soln_count
    pickle.dump([nx,ny,sigma,s,SNR,soln_count,args],f)
    f.close()
#     # verify with load
#     f=open(filename,'rb')
#     mc_samp=pickle.load(f)
#     pde_info=pickle.load(f)
#     f.close
#     print(pde_cnt)

if __name__ == '__main__':
    main()
