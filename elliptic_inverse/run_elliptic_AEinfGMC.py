"""
Main function to run Elliptic PDE model (DILI; Cui et~al, 2016) to generate posterior samples
Shiwei Lan @ Caltech, 2016
--------------------------
Modified Sept 2019 @ ASU
"""

# modules
import os,argparse,pickle
import numpy as np
import dolfin as df
import tensorflow as tf
from tensorflow.keras.models import load_model

# the inverse problem
from Elliptic import Elliptic

# MCMC
import sys
sys.path.append( "../" )
from nn.autoencoder import AutoEncoder
from sampler.AEinfGMC_dolfin import AEinfGMC

np.set_printoptions(precision=3, suppress=True)
np.random.seed(2020)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('num_samp', nargs='?', type=int, default=10000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=1000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[2.,2.,1.3,6.,4.]) # SNR10: [.5,2,1.3,6.,4.];SNR100: [.01,.04,0.04,.52,.25]
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,4,1,4,1,4])
    parser.add_argument('algs', nargs='?', type=str, default=['AE_'+n for n in ('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC')])
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
    
    # define AutoEncoder
    loaded=np.load(file='./result/ae_training.npz')
    X=loaded['X']
    num_samp=X.shape[0]
    tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
    te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
    x_train,x_test=X[tr_idx],X[te_idx]
    # define Auto-Encoder
    latent_dim=441; half_depth=3
    ae=AutoEncoder(x_train, x_test,latent_dim, half_depth)
    try:
        folder = os.path.join(os.getcwd(),'analysis_f_SNR'+str(SNR))
        ae.model=load_model(os.path.join(folder,'ae_fullmodel.h5'))
        ae.encoder=load_model(os.path.join(folder,'ae_encoder.h5'))
        from tensorflow.keras import backend as K
        ae.decoder=K.function(inputs=ae.model.get_layer(name="encode_out").output,outputs=ae.model.output)
        print('AutoEncoder has been loaded.')
    except Exception as err:
        print(err)
        print('Train AutoEncoder...\n')
        epochs=200
        import timeit
        t_start=timeit.default_timer()
        ae.train(epochs,batch_size=64,verbose=1)
        t_used=timeit.default_timer()-t_start
        print('\nTime used for training AutoEncoder: {}'.format(t_used))
        ae.save('./result/')
    
    # initialization
#     unknown=elliptic.prior.sample(whiten=False)
#     unknown=elliptic.prior.gen_vector()
    unknown=np.random.randn(latent_dim)
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))
    
    from geom_latent import geom
    latent_geom=lambda q,geom_ord=[0],whitened=False:geom(q,elliptic,ae,geom_ord,whitened)
    AE_infGMC=AEinfGMC(unknown,elliptic,latent_geom,ae,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO],volcrK=False)
    mc_fun=AE_infGMC.sample
    mc_args=(args.num_samp,args.num_burnin)
    mc_fun(*mc_args)
    
    # append PDE information including the count of solving
    filename_=os.path.join(AE_infGMC.savepath,AE_infGMC.filename+'.pckl')
    filename=os.path.join(AE_infGMC.savepath,'Elliptic_'+AE_infGMC.filename+'.pckl') # change filename
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
