"""
Main function to run Advection-Diffusion inverse problem to generate estimates
Shiwei Lan @ ASU, 2020
"""

# modules
import os,argparse,pickle
import numpy as np
import dolfin as df

# the inverse problem
from advdiff import advdiff

# MCMC
import sys
sys.path.append( "../" )
from optimizer.EnK_dolfin import *
from util.multivector import *

np.set_printoptions(precision=3, suppress=True)
seed=2020
np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('ensemble_size', nargs='?', type=int, default=100)
    parser.add_argument('max_iter', nargs='?', type=int, default=50)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[1.,.1]) # SNR10: [1,.01];SNR100: [1,.01]
    parser.add_argument('algs', nargs='?', type=str, default=('EKI','EKS'))
    args = parser.parse_args()

    ## define Advection-Diffusion inverse problem ##
#     mesh = df.Mesh('ad_10k.xml')
    meshsz = (51,51)
    rel_noise = .5
    nref = 1
    adif = advdiff(mesh=meshsz, rel_noise=rel_noise, nref=nref, seed=seed)
    adif.prior.V=adif.prior.Vh
    
    # initialization
    unknown=MultiVector(adif.prior.gen_vector(),args.ensemble_size)
    for j in range(args.ensemble_size): unknown[j].set_local(adif.prior.sample(whiten=False))
    # define parameters needed
    G = lambda u, IP=adif: np.array([dat.get_local() for dat in IP.misfit.get_observations(pde=IP.pde, init=u).data]).flatten()
    
    y=np.array([dat.get_local() for dat in adif.misfit.d.data]).flatten()
    data={'obs':y,'size':y.size,'cov':adif.misfit.noise_variance*np.eye(y.size)}
    
    # EnK parameters
    nz_lvl=1
    err_thld=1e-1
    
    # run EnK to generate ensembles
    print("Preparing %s with step size %g ..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO]))
    ek=EnK(unknown,G,data,adif.prior,stp_sz=args.step_sizes[args.algNO],nz_lvl=nz_lvl,err_thld=err_thld,alg=args.algs[args.algNO],adpt=True)
    ek_fun=ek.run
    ek_args=(args.max_iter,True)
    savepath,filename=ek_fun(*ek_args)
    
    # append PDE information including the count of solving
    filename_=os.path.join(savepath,filename+'.pckl')
    filename=os.path.join(savepath,'AdvDiff_'+filename+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
#     soln_count=[adif.soln_count,adif.pde.soln_count]
    soln_count=adif.pde.soln_count
    pickle.dump([meshsz,rel_noise,nref,soln_count,args],f)
    f.close()
#     # verify with load
#     f=open(filename,'rb')
#     mc_samp=pickle.load(f)
#     pde_info=pickle.load(f)
#     f.close
#     print(pde_cnt)

if __name__ == '__main__':
    main()
