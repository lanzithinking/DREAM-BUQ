"""
prepare training data
Shiwei Lan @ ASU, August 2020
"""

import numpy as np

import os,sys
sys.path.append( "../" )
import pickle

TRAIN='X'

# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
# preparation for estimates
folder = './analysis'
pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
ensbl_sz=100
max_iter=50

SAVE=True
# prepare data
for a in range(num_algs):
    print('Working on '+algs[a]+' algorithm...')
    found=False
    # ensembles and forward outputs
    for f_i in pckl_files:
        if algs[a]+'_ensbl'+str(ensbl_sz)+'_' in f_i:
            try:
                f=open(os.path.join(folder,f_i),'rb')
                loaded=pickle.load(f)
                ensbl=loaded[3][:-1,:,:] if 'Y' in TRAIN else loaded[3][1:,:,:]
                ensbl=ensbl.reshape((-1,ensbl.shape[2]))
                fwdout=loaded[2].reshape((-1,loaded[2].shape[2]))
                f.close()
                print(f_i+' has been read!')
                found=True; break
            except:
                found=False
                pass
    if found and SAVE:
        savepath='./train_NN/'
        if 'Y' in TRAIN:
            np.savez_compressed(file=os.path.join(savepath,algs[a]+'_ensbl'+str(ensbl_sz)+'_training_'+TRAIN),X=ensbl,Y=fwdout)
        else:
            np.savez_compressed(file=os.path.join(savepath,algs[a]+'_ensbl'+str(ensbl_sz)+'_training_'+TRAIN),X=ensbl)
#         # how to load
#         loaded=np.load(file=os.path.join(savepath,algs[a]+'_ensbl'+str(ensbl_sz)+'_training_'+TRAIN+'.npz'))
#         X=loaded['X']
#         Y=loaded['Y']