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
from nn.ae import AutoEncoder
from sampler.AEinfGMC_dolfin import AEinfGMC

# relevant geometry
from geom_latent_ae import geom

np.set_printoptions(precision=3, suppress=True)
np.random.seed(2020)
tf.random.set_seed(2020)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=3)
    parser.add_argument('num_samp', nargs='?', type=int, default=5000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=1000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.2,2.,1.5,3.,2.5]) # latent_dim441: [.2,2.,1.5,3.,2.5]
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
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
    # define the latent (coarser) inverse problem
    nx=10; ny=10
    obs,nzsd,loc=[getattr(elliptic.misfit,i) for i in ('obs','nzsd','loc')]
    elliptic_latent = Elliptic(nx=nx,ny=ny,SNR=SNR,obs=obs,nzsd=nzsd,loc=loc)
    
    # define AutoEncoder
    # training data algorithms
    algs=['EKI','EKS']
    num_algs=len(algs)
    alg_no=1
    # load data
    ensbl_sz = 500
    folder = './analysis_f_SNR'+str(SNR)
    loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_X.npz'))
    X=loaded['X']
    num_samp=X.shape[0]
#     tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
#     te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
#     x_train,x_test=X[tr_idx],X[te_idx]
    n_tr=np.int(num_samp*.75)
    x_train=X[:n_tr]
    x_test=X[n_tr:]
    # define Auto-Encoder
    half_depth=3; latent_dim=elliptic_latent.pde.V.dim()
    activation='linear'
#     activation=tf.keras.layers.LeakyReLU(alpha=0.01)
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
    ae=AutoEncoder(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim,
                   activation=activation, optimizer=optimizer)
    f_name=['ae_'+i+'_'+algs[alg_no]+str(ensbl_sz) for i in ('fullmodel','encoder','decoder')]
    try:
        ae.model=load_model(os.path.join(folder,f_name[0]+'.h5'),custom_objects={'loss':None})
        print(f_name[0]+' has been loaded!')
        ae.encoder=load_model(os.path.join(folder,f_name[1]+'.h5'),custom_objects={'loss':None})
        print(f_name[1]+' has been loaded!')
        ae.decoder=load_model(os.path.join(folder,f_name[2]+'.h5'),custom_objects={'loss':None})
        print(f_name[2]+' has been loaded!')
    except Exception as err:
        print(err)
        print('Train AutoEncoder...\n')
        epochs=200
        patience=0
        import timeit
        t_start=timeit.default_timer()
        ae.train(x_train,x_test=x_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
        t_used=timeit.default_timer()-t_start
        print('\nTime used for training AE: {}'.format(t_used))
        # save AE
        ae.model.save(os.path.join(folder,f_name[0]+'.h5'))
        ae.encoder.save(os.path.join(folder,f_name[1]+'.h5'))
        ae.decoder.save(os.path.join(folder,f_name[2]+'.h5'))
    
    # initialization
#     unknown=elliptic_latent.prior.sample(whiten=False)
    unknown=elliptic_latent.prior.gen_vector()
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))
    
    latent_geom=lambda q,geom_ord=[0],whitened=False,**kwargs:geom(q,elliptic,ae,geom_ord,whitened,**kwargs)
    AE_infGMC=AEinfGMC(unknown,elliptic_latent,latent_geom,ae,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO],volcrK=False,k=5,bip_lat=elliptic_latent) # uncomment for manifold algorithms
    mc_fun=AE_infGMC.sample
    mc_args=(args.num_samp,args.num_burnin)
    mc_fun(*mc_args)
    
    # append PDE information including the count of solving
    filename_=os.path.join(AE_infGMC.savepath,AE_infGMC.filename+'.pckl')
    filename=os.path.join(AE_infGMC.savepath,'Elliptic_'+AE_infGMC.filename+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
#     soln_count=[elliptic_latent.soln_count,elliptic_latent.pde.soln_count]
    soln_count=elliptic_latent.pde.soln_count
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
