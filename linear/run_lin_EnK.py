"""
Main function to run ensemble methods for linear-Gaussian inverse problem
Shiwei Lan @ ASU, 2020
"""

# modules
import os,argparse,pickle
import numpy as np

# the inverse problem
from LiN import LiN

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

    ## define the linear-Gaussian inverse problem ##
    # set up
    d=3; m=100
    try:
        with open('./result/lin.pickle','rb') as f:
            [nz_var,pr_cov,A,true_input,y]=pickle.load(f)
        print('Data loaded!\n')
        kwargs={'true_input':true_input,'A':A,'y':y}
    except:
        print('No data found. Generate new data...\n')
        nz_var=.1; pr_cov=1.
        true_input=np.arange(-np.floor(d/2),np.ceil(d/2))
        A=np.random.rand(m,d)
        kwargs={'true_input':true_input,'A':A}
    lin=LiN(d,m,nz_var=nz_var,pr_cov=pr_cov,**kwargs)
    
    # initialization
    u0=lin.sample(num_samp=args.ensemble_size)
    G=lin.forward
    y=lin.y
    data={'obs':y,'size':y.size,'cov':np.diag(lin.nz_var)}
    prior={'mean':np.zeros(lin.input_dim),'cov':np.diag(lin.pr_cov) if np.ndim(lin.pr_cov)==1 else lin.pr_cov,'sample':lin.sample}
    
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
    filename=os.path.join(savepath,'lin_'+filename+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
    pickle.dump([nz_var,pr_cov,A,true_input,y,args],f)
    f.close()

if __name__ == '__main__':
    main()
