"""
Main function to run Elliptic PDE model (DILI; Cui et~al, 2016) to generate posterior samples
Shiwei Lan @ Caltech, 2016
--------------------------
Modified for DREAM June 2020 @ ASU
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
from nn.dnn import DNN
from nn.cnn import CNN
from nn.ae import AutoEncoder
from nn.cae import ConvAutoEncoder
from sampler.DREAM_dolfin import DREAM

# relevant geometry
import geom_emul
from geom_latent import *

np.set_printoptions(precision=3, suppress=True)
np.random.seed(2020)
tf.random.set_seed(2020)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('emuNO', nargs='?', type=int, default=1)
    parser.add_argument('aeNO', nargs='?', type=int, default=0)
    parser.add_argument('num_samp', nargs='?', type=int, default=5000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=1000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.3,2.5,1.5,None,None]) # AE [.3,2.5,1.5] # CAE [.06,.3]
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=['DREAM'+a for a in ('pCN','infMALA','infHMC','infmMALA','infmHMC')])
    parser.add_argument('emus', nargs='?', type=str, default=['dnn','cnn'])
    parser.add_argument('aes', nargs='?', type=str, default=['ae','cae'])
    args = parser.parse_args()
    
    ##------ define the inverse elliptic problem ------##
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
    
    ##------ define networks ------##
    # training data algorithms
    algs=['EKI','EKS']
    num_algs=len(algs)
    alg_no=1
    # load data
    ensbl_sz = 500
    folder = './analysis_f_SNR'+str(SNR)
    if not os.path.exists(folder): os.makedirs(folder)
    
    ##---- EMULATOR ----##
    # prepare for training data
    if args.emus[args.emuNO]=='dnn':
        loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
        X=loaded['X']; Y=loaded['Y']
    elif args.emus[args.emuNO]=='cnn':
        loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XimgY.npz'))
        X=loaded['X']; Y=loaded['Y']
        X=X[:,:,:,None]
    num_samp=X.shape[0]
#     n_tr=np.int(num_samp*.75)
#     x_train,y_train=X[:n_tr],Y[:n_tr]
#     x_test,y_test=X[n_tr:],Y[n_tr:]
    tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
    te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
    x_train,x_test=X[tr_idx],X[te_idx]
    y_train,y_test=Y[tr_idx],Y[te_idx]
    # define emulator
    if args.emus[args.emuNO]=='dnn':
        depth=3
        activations={'hidden':'softplus','output':'linear'}
        droprate=.4
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        emulator=DNN(x_train.shape[1], y_train.shape[1], depth=depth, droprate=droprate,
                     activations=activations, optimizer=optimizer)
    elif args.emus[args.emuNO]=='cnn':
        num_filters=[16,8,8]
        activations={'conv':'softplus','latent':'softmax','output':'linear'}
        latent_dim=256
        droprate=.5
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        emulator=CNN(x_train.shape[1:], y_train.shape[1], num_filters=num_filters, latent_dim=latent_dim, droprate=droprate,
                     activations=activations, optimizer=optimizer)
    f_name=args.emus[args.emuNO]+'_'+algs[alg_no]+str(ensbl_sz)
    # load emulator
    try:
        emulator.model=load_model(os.path.join(folder,f_name+'.h5'),custom_objects={'loss':None})
        print(f_name+' has been loaded!')
    except:
        try:
            emulator.model.load_weights(os.path.join(folder,f_name+'.h5'))
            print(f_name+' has been loaded!')
        except:
            print('\nNo emulator found. Training {}...\n'.format(args.emus[args.emuNO]))
            epochs=200
            patience=0
            emulator.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
            # save emulator
            try:
                emulator.model.save(os.path.join(folder,f_name+'.h5'))
            except:
                emulator.model.save_weights(os.path.join(folder,f_name+'.h5'))
    
    ##---- AUTOENCODER ----##
    # prepare for training data
    if args.aes[args.aeNO]=='ae':
        loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
        X=loaded['X']
    elif args.aes[args.aeNO]=='cae':
        loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XimgY.npz'))
        X=loaded['X']
        X=X[:,:-1,:-1,None]
    num_samp=X.shape[0]
#     n_tr=np.int(num_samp*.75)
#     x_train=X[:n_tr]
#     x_test=X[n_tr:]
    tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
    te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
    x_train,x_test=X[tr_idx],X[te_idx]
    # define autoencoder
    if args.aes[args.aeNO]=='ae':
        half_depth=3; latent_dim=elliptic_latent.pde.V.dim()
        activation='linear'
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
        autoencoder=AutoEncoder(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim,
                                activation=activation, optimizer=optimizer)
    elif args.aes[args.aeNO]=='cae':
        num_filters=[16,1]
        activations={'conv':tf.keras.layers.LeakyReLU(alpha=0.1),'latent':None}
        latent_dim=elliptic_latent.prior.dim
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        autoencoder=ConvAutoEncoder(x_train.shape[1:], num_filters=num_filters, latent_dim=latent_dim,
                                    activations=activations, optimizer=optimizer)
    f_name=[args.aes[args.aeNO]+'_'+i+'_'+algs[alg_no]+str(ensbl_sz) for i in ('fullmodel','encoder','decoder')]
    # load autoencoder
    try:
        autoencoder.model=load_model(os.path.join(folder,f_name[0]+'.h5'),custom_objects={'loss':None})
        print(f_name[0]+' has been loaded!')
        autoencoder.encoder=load_model(os.path.join(folder,f_name[1]+'.h5'),custom_objects={'loss':None})
        print(f_name[1]+' has been loaded!')
        autoencoder.decoder=load_model(os.path.join(folder,f_name[2]+'.h5'),custom_objects={'loss':None})
        print(f_name[2]+' has been loaded!')
    except:
        print('\nNo autoencoder found. Training {}...\n'.format(args.aes[args.aeNO]))
        epochs=200
        patience=0
        autoencoder.train(x_train,x_test=x_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
        # save autoencoder
        autoencoder.model.save(os.path.join(folder,f_name[0]+'.h5'))
        autoencoder.encoder.save(os.path.join(folder,f_name[1]+'.h5'))
        autoencoder.decoder.save(os.path.join(folder,f_name[2]+'.h5'))
    
    
    ##------ define MCMC ------##
    # initialization
#     unknown=elliptic_latent.prior.sample(whiten=False)
    unknown=elliptic_latent.prior.gen_vector()
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))
    
    emul_geom=lambda q,geom_ord=[0],whitened=False,**kwargs:geom_emul.geom(q,elliptic,emulator,geom_ord,whitened,**kwargs)
    latent_geom=lambda q,geom_ord=[0],whitened=False,**kwargs:geom(q,elliptic_latent.pde.V,elliptic.pde.V,autoencoder,geom_ord,whitened,emul_geom=emul_geom,**kwargs)
    dream=DREAM(unknown,elliptic_latent,latent_geom,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO],volcrK=False)#,k=5,bip_lat=elliptic_latent) # uncomment for manifold algorithms
    mc_fun=dream.sample
    mc_args=(args.num_samp,args.num_burnin)
    mc_fun(*mc_args)
    
    # append PDE information including the count of solving
    filename_=os.path.join(dream.savepath,dream.filename+'.pckl')
    filename=os.path.join(dream.savepath,'Elliptic_'+dream.filename+'_'+args.emus[args.emuNO]+'_'+args.aes[args.aeNO]+'.pckl') # change filename
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
