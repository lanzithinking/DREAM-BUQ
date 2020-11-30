"""
Main function to run DREAM for Advection-Diffusion inverse problem
Shiwei Lan @ ASU, 2020
"""

# modules
import os,argparse,pickle
import numpy as np
# on mac import dolfin then import tensorflow; reverse the order on linux -- it is a known issue on for FEniCS-2019.1 and TensorFlow-2.2
import dolfin as df
import tensorflow as tf
from tensorflow.keras.models import load_model

# the inverse problem
from advdiff import advdiff

# MCMC
import sys
sys.path.append( "../" )
from nn.dnn import DNN
from nn.cnn import CNN
from nn.ae import AutoEncoder
from nn.cae import ConvAutoEncoder
from nn.vae import VAE
from sampler.DREAM_dolfin import DREAM

# relevant geometry
import geom_emul
from geom_latent import *

# set to warn only once for the same warnings
tf.get_logger().setLevel('ERROR')
np.set_printoptions(precision=3, suppress=True)
seed=2020
np.random.seed(seed)
tf.random.set_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=1)
    parser.add_argument('emuNO', nargs='?', type=int, default=0)
    parser.add_argument('aeNO', nargs='?', type=int, default=0)
    parser.add_argument('num_samp', nargs='?', type=int, default=5000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=1000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[1e-2,1e-2,1e-2,None,None]) # AE [1e-2,1e-2,1e-2]
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=['DREAM'+a for a in ('pCN','infMALA','infHMC','infmMALA','infmHMC')])
    parser.add_argument('emus', nargs='?', type=str, default=['dnn','cnn'])
    parser.add_argument('aes', nargs='?', type=str, default=['ae','cae','vae'])
    args = parser.parse_args()
    
    ##------ define the inverse problem ------##
    ## define the Advection-Diffusion invese problem ##
#     mesh = df.Mesh('ad_10k.xml')
    meshsz = (51,51)
    rel_noise = .5
    nref = 1
    adif = advdiff(mesh=meshsz, rel_noise=rel_noise, nref=nref, seed=seed)
    adif.prior.V=adif.prior.Vh
    adif.misfit.obs=np.array([dat.get_local() for dat in adif.misfit.d.data]).flatten()
    # set up latent
    meshsz_latent = (11,11)
    adif_latent = advdiff(mesh=meshsz_latent, rel_noise=rel_noise, nref=nref, seed=seed)
    adif_latent.prior.V=adif_latent.prior.Vh
        
    ##------ define networks ------##
    # training data algorithms
    algs=['EKI','EKS']
    num_algs=len(algs)
    alg_no=1
    # load data
    ensbl_sz = 500
    folder = './train_NN'
#     if not os.path.exists(folder): os.makedirs(folder)
    
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
        depth=5
        activations={'hidden':tf.keras.layers.LeakyReLU(alpha=.01),'output':'linear'}
        droprate=0.25
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
        emulator=DNN(x_train.shape[1], y_train.shape[1], depth=depth, droprate=droprate,
                     activations=activations, optimizer=optimizer)
    elif args.emus[args.emuNO]=='cnn':
        num_filters=[16,8,8]
        activations={'conv':'softplus','latent':'softmax','output':'linear'}
        latent_dim=1024
        droprate=.25
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
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
    if 'c' in args.aes[args.aeNO]:
        loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XimgY.npz'))
        X=loaded['X']
        X=X[:,:-1,:-1,None]
    else:
        loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
        X=loaded['X']
    num_samp=X.shape[0]
#     n_tr=np.int(num_samp*.75)
#     x_train=X[:n_tr]
#     x_test=X[n_tr:]
    tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
    te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
    x_train,x_test=X[tr_idx],X[te_idx]
    # define autoencoder
    if args.aes[args.aeNO]=='ae':
        half_depth=3; latent_dim=adif_latent.prior.V.dim()
        droprate=0.
#         activation='linear'
        activation=tf.keras.layers.LeakyReLU(alpha=1.5)
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
        lambda_=0.
        autoencoder=AutoEncoder(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim, droprate=droprate,
                                activation=activation, optimizer=optimizer)
    elif args.aes[args.aeNO]=='cae':
        num_filters=[16,8]; latent_dim=adif_latent.prior.V.dim()
#         activations={'conv':tf.keras.layers.LeakyReLU(alpha=0.1),'latent':None} # [16,1]
        activations={'conv':'elu','latent':'linear'}
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        autoencoder=ConvAutoEncoder(x_train.shape[1:], num_filters=num_filters, latent_dim=latent_dim,
                                    activations=activations, optimizer=optimizer)
    elif args.aes[args.aeNO]=='vae':
        half_depth=5; latent_dim=adif_latent.prior.V.dim()
        repatr_out=False; beta=1.
        activation='elu'
#         activation=tf.keras.layers.LeakyReLU(alpha=0.01)
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
        autoencoder=VAE(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim, repatr_out=repatr_out,
                        activation=activation, optimizer=optimizer, beta=beta)
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
        noise=0.
        kwargs={'patience':patience}
        if args.aes[args.aeNO]=='ae' and noise: kwargs['noise']=noise
        autoencoder.train(x_train,x_test=x_test,epochs=epochs,batch_size=64,verbose=1,**kwargs)
        # save autoencoder
        autoencoder.model.save(os.path.join(folder,f_name[0]+'.h5'))
        autoencoder.encoder.save(os.path.join(folder,f_name[1]+'.h5'))
        autoencoder.decoder.save(os.path.join(folder,f_name[2]+'.h5'))
    
    
    ##------ define MCMC ------##
    # initialization
    u0=adif_latent.prior.sample(whiten=False)
    emul_geom=lambda q,geom_ord=[0],whitened=False,**kwargs:geom_emul.geom(q,adif,emulator,geom_ord,whitened,**kwargs)
    latent_geom=lambda q,geom_ord=[0],whitened=False,**kwargs:geom(q,adif_latent,adif,autoencoder,geom_ord,whitened,emul_geom=emul_geom,**kwargs)
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))
    
    dream=DREAM(u0,adif_latent,latent_geom,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO],whitened=False,log_wts=False)#,AE=autoencoder)#,k=5) # uncomment for manifold algorithms
    mc_fun=dream.sample
    mc_args=(args.num_samp,args.num_burnin)
    mc_fun(*mc_args)
    
    # append PDE information including the count of solving
    filename_=os.path.join(dream.savepath,dream.filename+'.pckl')
    filename=os.path.join(dream.savepath,'AdvDiff_'+dream.filename+'_'+args.emus[args.emuNO]+'_'+args.aes[args.aeNO]+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
    soln_count=adif_latent.soln_count
    pickle.dump([meshsz,meshsz_latent,rel_noise,nref,soln_count,args],f)
    f.close()

if __name__ == '__main__':
    main()
