"""
Main function to run Elliptic PDE model (DILI; Cui et~al, 2016) to generate posterior samples
Shiwei Lan @ Caltech, 2016
--------------------------
Modified for DREAM June 2020 @ ASU
"""

# modules
import os,argparse,pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# the inverse problem
from BBD import BBD

# MCMC
import sys
sys.path.append( "../" )
from nn.dnn import DNN
from nn.cnn import CNN
from nn.ae import AutoEncoder
from nn.cae import ConvAutoEncoder
from nn.vae import VAE
from sampler.DREAM import DREAM

# relevant geometry
import geom_emul
from geom_latent import *

# set to warn only once for the same warnings
tf.get_logger().setLevel('ERROR')
np.set_printoptions(precision=3, suppress=True)
np.random.seed(2020)
tf.random.set_seed(2020)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=2)
    parser.add_argument('emuNO', nargs='?', type=int, default=0)
    parser.add_argument('aeNO', nargs='?', type=int, default=0)
    parser.add_argument('num_samp', nargs='?', type=int, default=10000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=10000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.01,.005,.005,None,None]) # AE [.01,.005,.01]
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=['DREAM'+a for a in ('pCN','infMALA','infHMC','infmMALA','infmHMC')])
    parser.add_argument('emus', nargs='?', type=str, default=['dnn','cnn'])
    parser.add_argument('aes', nargs='?', type=str, default=['ae','cae','vae'])
    args = parser.parse_args()
    
    ##------ define the BBD inverse problem ------##
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
    # set up latent
    latent_dim=3
    class BBD_lat:
        def __init__(self,input_dim):
            self.input_dim=input_dim
        def sample(self,num_samp=1):
            samp=np.random.randn(num_samp,self.input_dim)
            return np.squeeze(samp)
    bbd_latent=BBD_lat(latent_dim)
    bbd_latent.prior={'mean':np.zeros(bbd_latent.input_dim),'cov':np.eye(bbd_latent.input_dim), 'sample':bbd_latent.sample}
#     bbd_latent=BBD(latent_dim,bbd.output_dim,nz_var=1,pr_cov=1)
#     bbd_latent.prior={'mean':np.zeros(bbd_latent.input_dim),'cov':np.diag(bbd_latent.pr_cov) if np.ndim(bbd_latent.pr_cov)==1 else bbd_latent.pr_cov,'sample':bbd_latent.sample}
    
    ##------ define networks ------##
    # training data algorithms
    algs=['EKI','EKS']
    num_algs=len(algs)
    alg_no=1
    # load data
    ensbl_sz = 100
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
        activations={'hidden':'softplus','output':'linear'}
        droprate=0.
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
            epochs=1000
            patience=10
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
        loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_X.npz'))
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
        half_depth=2; latent_dim=3
        droprate=0.
#         activation='linear'
        activation=tf.keras.layers.LeakyReLU(alpha=2.)
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
        lambda_=0.
        autoencoder=AutoEncoder(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim, droprate=droprate,
                                activation=activation, optimizer=optimizer)
    elif args.aes[args.aeNO]=='cae':
        num_filters=[16,8]; latent_dim=2
#         activations={'conv':tf.keras.layers.LeakyReLU(alpha=0.1),'latent':None} # [16,1]
        activations={'conv':'elu','latent':'linear'}
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        autoencoder=ConvAutoEncoder(x_train.shape[1:], num_filters=num_filters, latent_dim=latent_dim,
                                    activations=activations, optimizer=optimizer)
    elif args.aes[args.aeNO]=='vae':
        half_depth=5; latent_dim=2
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
        epochs=1000
        patience=10
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
    u0=bbd_latent.prior['sample']()
    emul_geom=lambda q,geom_ord=[0],whitened=False,**kwargs:geom_emul.geom(q,bbd,emulator,geom_ord,whitened,**kwargs)
    latent_geom=lambda q,geom_ord=[0],whitened=False,**kwargs:geom(q,autoencoder,geom_ord,whitened,emul_geom=emul_geom,**kwargs)
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))
    
    dream=DREAM(u0,bbd_latent,latent_geom,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO],whitened=False,vol_wts='adjust',AE=autoencoder)#,k=5,bip_lat=bbd_latent) # uncomment for manifold algorithms
    mc_fun=dream.sample
    mc_args=(args.num_samp,args.num_burnin)
    mc_fun(*mc_args)
    
    # append PDE information including the count of solving
    filename_=os.path.join(dream.savepath,dream.filename+'.pckl')
    filename=os.path.join(dream.savepath,'BBD_'+dream.filename+'_'+args.emus[args.emuNO]+'_'+args.aes[args.aeNO]+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
    pickle.dump([nz_var,pr_cov,A,true_input,y,args],f)
    f.close()

if __name__ == '__main__':
    main()
