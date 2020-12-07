"""
Main function to run Advection-Diffusion inverse problem to generate posterior samples
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
from sampler.geoinfMC_dolfin import geoinfMC

np.set_printoptions(precision=3, suppress=True)
seed=2020
np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=5000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=1000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.001,.005,.005,None,None])
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC'))
    args = parser.parse_args()

    ## define Advection-Diffusion inverse problem ##
#     mesh = df.Mesh('ad_10k.xml')
    meshsz = (61,61)
    eldeg = 1
    gamma = 2.; delta = 10.
    rel_noise = .5
    nref = 1
    adif = advdiff(mesh=meshsz, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed)
    adif.prior.V=adif.prior.Vh
    
    # initialization
#     unknown=adif.prior.gen_vector()
    unknown=adif.prior.sample(whiten=False)
#     MAP_file=os.path.join(os.getcwd(),'properties/MAP.xdmf')
#     if os.path.isfile(MAP_file):
#         unknown=df.Function(adif.prior.V, name='MAP')
#         f=df.XDMFFile(adif.mpi_comm,MAP_file)
#         f.read_checkpoint(unknown,'m',0)
#         f.close()
#         unknown = unknown.vector()
#     else:
#         unknown=adif.get_MAP(SAVE=True)
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))
    
    inf_GMC=geoinfMC(unknown,adif,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO])
    mc_fun=inf_GMC.sample
    mc_args=(args.num_samp,args.num_burnin)
    mc_fun(*mc_args)
    
    # append PDE information including the count of solving
    filename_=os.path.join(inf_GMC.savepath,inf_GMC.filename+'.pckl')
    filename=os.path.join(inf_GMC.savepath,'AdvDiff_'+inf_GMC.filename+'.pckl') # change filename
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
