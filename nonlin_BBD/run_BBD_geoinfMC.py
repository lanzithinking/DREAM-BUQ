"""
Main function to run (inf-)geometric MCMC for nonlinear BBD inverse problem
Shiwei Lan @ Caltech, 2016
--------------------------
Modified Aug. 2020 @ ASU
"""

# modules
import os,argparse,pickle
import numpy as np

# the inverse problem
from BBD import BBD

# MCMC
import sys
sys.path.append( "../" )
from sampler.geoinfMC import geoinfMC

np.set_printoptions(precision=3, suppress=True)
np.random.seed(2020)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=2)
    parser.add_argument('num_samp', nargs='?', type=int, default=100000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=10000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.01,.015,.02,None,None])
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC'))
    args = parser.parse_args()

    ## define the BBD inverse problem ##
    # set up
    try:
        with open('./result/BBD.pickle','rb') as f:
            [nz_var,pr_cov,A,true_input,y]=pickle.load(f)
        print('Data loaded!\n')
        kwargs={'true_input':true_input,'A':A,'y':y}
    except:
        print('No data found. Generate new data...\n')
        d=4; m=100
        nz_var=1; pr_cov=1
        true_input=np.random.randint(d,size=d)
        A=np.random.rand(m,d)
        kwargs={'true_input':true_input,'A':A}
    bbd=BBD(nz_var=nz_var,pr_cov=pr_cov,**kwargs)
    y=bbd.y
    bbd.prior={'mean':np.zeros(bbd.input_dim),'cov':np.diag(bbd.pr_cov) if np.ndim(bbd.pr_cov)==1 else bbd.pr_cov,'sample':bbd.sample}
    
    # initialization
    u0=bbd.sample()
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))
    
    inf_GMC=geoinfMC(u0,bbd,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO])
    mc_fun=inf_GMC.sample
    mc_args=(args.num_samp,args.num_burnin)
    mc_fun(*mc_args)
    
    # append PDE information including the count of solving
    filename_=os.path.join(inf_GMC.savepath,inf_GMC.filename+'.pckl')
    filename=os.path.join(inf_GMC.savepath,'BBD_'+inf_GMC.filename+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
    pickle.dump([nz_var,pr_cov,A,true_input,y,args],f)
    f.close()

if __name__ == '__main__':
    main()
