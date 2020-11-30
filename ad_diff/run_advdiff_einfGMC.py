"""
Main function to run emulative (inf-)geometric MCMC for Advection-Diffusion inverse problem
Shiwei Lan @ ASU, 2020
"""

# modules
import os,argparse,pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# the inverse problem
from advdiff import advdiff

# MCMC
import sys
sys.path.append( "../" )
from nn.dnn import DNN
from nn.cnn import CNN
from sampler.einfGMC_dolfin import einfGMC

# relevant geometry
from geom_emul import geom

np.set_printoptions(precision=3, suppress=True)
seed=2020
np.random.seed(seed)
tf.random.set_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=1)
    parser.add_argument('emuNO', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=5000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=1000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[1e-2,1e-5,1e-5,None,None]) # DNN: 1e-3, 1e-5, 1e-5; CNN: 
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=['e'+a for a in ('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC')])
    parser.add_argument('emus', nargs='?', type=str, default=['dnn','cnn'])
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
    
    ##------ define networks ------##
    # training data algorithms
    algs=['EKI','EKS']
    num_algs=len(algs)
    alg_no=1
    # load data
    ensbl_sz = 500
    folder = './train_NN'
    
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
    
    # initialization
    u0=adif.prior.sample(whiten=False)
    emul_geom=lambda q,geom_ord=[0],whitened=False,**kwargs:geom(q,adif,emulator,geom_ord,whitened,**kwargs)
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))
    
    e_infGMC=einfGMC(u0,adif,emul_geom,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO])#,k=5) # uncomment for manifold algorithms
    mc_fun=e_infGMC.sample
    mc_args=(args.num_samp,args.num_burnin)
    mc_fun(*mc_args)
    
    # append PDE information including the count of solving
    filename_=os.path.join(e_infGMC.savepath,e_infGMC.filename+'.pckl')
    filename=os.path.join(e_infGMC.savepath,'AdvDiff_'+e_infGMC.filename+'_'+args.emus[args.emuNO]+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
    soln_count=adif.soln_count
    pickle.dump([meshsz,rel_noise,nref,soln_count,args],f)
    f.close()

if __name__ == '__main__':
    main()
