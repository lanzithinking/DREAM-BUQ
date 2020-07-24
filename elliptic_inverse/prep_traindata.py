"""
Extract functions stored in hdf5 files and prepare them as training data
Shiwei Lan @ ASU, May 2020
"""

import dolfin as df
import numpy as np

import os,sys
sys.path.append( "../" )
from util.dolfin_gadget import vec2fun,fun2img
import pickle

TRAIN={0:'XimgY',1:'XY'}[1]
whiten=True

def retrieve_ensemble(bip,dir_name,f_name,ensbl_sz,max_iter,img_out=False,whiten=False):
    f=df.HDF5File(bip.pde.mpi_comm,os.path.join(dir_name,f_name),"r")
    ensbl_f=df.Function(bip.pde.V)
    num_ensbls=max_iter*ensbl_sz
    if img_out:
        gdim = bip.pde.V.mesh().geometry().dim()
        imsz = np.floor(bip.pde.V.dim()**(1./gdim)).astype('int')
        out_shape=(num_ensbls,np.int(bip.pde.V.dim()/imsz**(gdim-1)))+(imsz,)*(gdim-1)
    else:
        out_shape=(num_ensbls,bip.pde.V.dim())
    out=np.zeros(out_shape)
    prog=np.ceil(num_ensbls*(.1+np.arange(0,1,.1)))
    for n in range(max_iter):
        for j in range(ensbl_sz):
            f.read(ensbl_f,'iter{0}_ensbl{1}'.format(n+('Y' not in TRAIN),j))
            s=n*ensbl_sz+j
            if whiten:
                ensbl_v=bip.prior.u2v(ensbl_f.vector())
                out[s]=fun2img(vec2fun(ensbl_v,bip.pde.V)) if img_out else ensbl_v.get_local()
            else:
                out[s]=fun2img(ensbl_f) if img_out else ensbl_f.vector().get_local()
            if s+1 in prog:
                print('{0:.0f}% ensembles have been retrieved.'.format(np.float(s+1)/num_ensbls*100))
    f.close()
    return out

if __name__ == '__main__':
    from Elliptic import Elliptic
    # define the inverse problem
    np.random.seed(2020)
    nx=40; ny=40
    SNR=50
    elliptic = Elliptic(nx=nx,ny=ny,SNR=SNR)
    # algorithms
    algs=['EKI','EKS']
    num_algs=len(algs)
    # preparation for estimates
    folder = './analysis_f_SNR'+str(SNR)
    hdf5_files=[f for f in os.listdir(folder) if f.endswith('.h5')]
    pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
    ensbl_sz=500
    max_iter=10
    img_out=('img' in TRAIN)
    
    PLOT=False
    SAVE=True
    # prepare data
    for a in range(num_algs):
        print('Working on '+algs[a]+' algorithm...')
        found=False
        # ensembles
        for f_i in hdf5_files:
            if algs[a]+'_ensbl'+str(ensbl_sz)+'_' in f_i:
                try:
                    out=retrieve_ensemble(elliptic,folder,f_i,ensbl_sz,max_iter,img_out,whiten)
                    print(f_i+' has been read!')
                    found=True; break
                except:
                    pass
        if found and img_out and PLOT:
            import matplotlib.pyplot as plt
            plt.rcParams['image.cmap'] = 'jet'
            fig = plt.figure(figsize=(8,8), facecolor='white')
            ax = fig.add_subplot(111, frameon=False)
            plt.ion()
            plt.show(block=False)
            for t in range(out.shape[0]):
                plt.cla()
                plt.imshow(out[t],origin='lower',extent=[0,1,0,1])
                plt.title('Ensemble {}'.format(t))
                plt.show()
                plt.pause(1.0/100.0)
        # forward outputs
        if 'Y' in TRAIN:
            for f_i in pckl_files:
                if algs[a]+'_ensbl'+str(ensbl_sz)+'_' in f_i:
                    try:
                        f=open(os.path.join(folder,f_i),'rb')
                        loaded=pickle.load(f)
                        f.close()
                        print(f_i+' has been read!')
                        fwdout=loaded[1].reshape((-1,loaded[1].shape[2]))
                        break
                    except:
                        found=False
                        pass
        if found and SAVE:
            savepath='./train_NN/'
            ifwhiten='_whitened' if whiten else ''
            if 'Y' in TRAIN:
                np.savez_compressed(file=os.path.join(savepath,algs[a]+'_ensbl'+str(ensbl_sz)+'_training_'+TRAIN+ifwhiten),X=out,Y=fwdout)
            else:
                np.savez_compressed(file=os.path.join(savepath,algs[a]+'_ensbl'+str(ensbl_sz)+'_training_'+TRAIN+ifwhiten),X=out)
#             # how to load
#             loaded=np.load(file=os.path.join(savepath,algs[a]+'_ensbl'+str(ensbl_sz)+'_training_'+TRAIN+'.npz'))
#             X=loaded['X']
#             Y=loaded['Y']