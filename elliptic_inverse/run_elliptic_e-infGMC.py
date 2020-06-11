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
from nn.cnn import CNN
from sampler.einfGMC_dolfin import einfGMC

np.set_printoptions(precision=3, suppress=True)
np.random.seed(2020)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('num_samp', nargs='?', type=int, default=5000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=1000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.3,2.,1.3,6.,4.]) # SNR10: [.5,2,1.3,6.,4.];SNR100: [.01,.04,0.04,.52,.25]
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,4,1,4,1,4])
    parser.add_argument('algs', nargs='?', type=str, default=['e_'+n for n in ('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC')])
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
    # algorithms
    algs=['EKI','EKS']
    num_algs=len(algs)
    alg_no=1
    
    # define the emulator (CNN)
    # load data
    ensbl_sz = 100
    folder = './analysis_f_SNR'+str(SNR)
    loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training.npz'))
    X=loaded['X']
    Y=loaded['Y']
    # pre-processing: scale X to 0-1
#     X-=np.nanmin(X,axis=(1,2),keepdims=True) # try axis=(1,2,3)
#     X/=np.nanmax(X,axis=(1,2),keepdims=True)
    X=X[:,:,:,None]
    # split train/test
    num_samp=X.shape[0]
    n_tr=np.int(num_samp*.75)
    x_train,y_train=X[:n_tr],Y[:n_tr]
    x_test,y_test=X[n_tr:],Y[n_tr:]
    
    # define CNN
    num_filters=[16,32]
    activations={'conv':'relu','latent':tf.keras.layers.PReLU(),'output':'linear'}
    latent_dim=128
    droprate=.25
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    cnn=CNN(x_train.shape[1:], y_train.shape[1], num_filters=num_filters,
            latent_dim=latent_dim, activations=activations, droprate=droprate, optimizer=optimizer)
    try:
        cnn.model=load_model(os.path.join(folder,'cnn_'+algs[alg_no]+'.h5'))
        print('cnn_'+algs[alg_no]+'.h5'+' has been loaded!')
    except Exception as err:
        print(err)
        print('Train CNN...\n')
        epochs=100
        import timeit
        t_start=timeit.default_timer()
        cnn.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1)
        t_used=timeit.default_timer()-t_start
        print('\nTime used for training CNN: {}'.format(t_used))
        # save CNN
#         cnn.model.save('./result/cnn_model.h5')
        cnn.save(folder,'cnn_'+algs[alg_no])
    
    # initialization
#     unknown=elliptic.prior.sample(whiten=False)
    unknown=elliptic.prior.gen_vector()
#     unknown=np.random.randn(latent_dim)
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))
    
    from geom_emul import geom
    emul_geom=lambda q,geom_ord=[0],whitened=False:geom(q,elliptic,cnn,geom_ord,whitened)
    e_infGMC=einfGMC(unknown,elliptic,emul_geom,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO],volcrK=False)
    mc_fun=e_infGMC.sample
    mc_args=(args.num_samp,args.num_burnin)
    mc_fun(*mc_args)
    
    # append PDE information including the count of solving
    filename_=os.path.join(e_infGMC.savepath,e_infGMC.filename+'.pckl')
    filename=os.path.join(e_infGMC.savepath,'Elliptic_'+e_infGMC.filename+'.pckl') # change filename
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
