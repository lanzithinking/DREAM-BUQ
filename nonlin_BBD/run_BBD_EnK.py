"""
Main function to run ensemble methods for nonlinear BBD inverse problem
Shiwei Lan @ ASU, 2020
"""

# modules
import os,argparse,pickle
import numpy as np

# the inverse problem
from BBD import BBD

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
    
    # initialization
    u0=bbd.sample(num_samp=args.ensemble_size)
    G=bbd.forward
    y=bbd.y
    data={'obs':y,'size':y.size,'cov':np.diag(bbd.nz_var)}
    prior={'mean':np.zeros(bbd.input_dim),'cov':np.diag(bbd.pr_cov) if np.ndim(bbd.pr_cov)==1 else bbd.pr_cov,'sample':bbd.sample}
    
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
    filename=os.path.join(savepath,'BBD_'+filename+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
    pickle.dump([nz_var,pr_cov,A,true_input,y,args],f)
    f.close()

if __name__ == '__main__':
    main()
