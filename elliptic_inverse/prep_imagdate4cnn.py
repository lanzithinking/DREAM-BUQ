"""
Extract functions stored in hdf5 files and prepare them as images for CNN training
Shiwei Lan @ ASU, May 2020
"""

import dolfin as df
import numpy as np

import os,sys
sys.path.append( "../" )
from util.dolfin_gadget import fun2img
import pickle

# def retrieve_each_sample(f,samp_f,s):
#     f.read(samp_f,'sample_{0}'.format(s))
#     return samp_f.vector()

# def retrieve_sample(mpi_comm,V,dir_name,f_name,num_samp):
#     f=df.HDF5File(mpi_comm,os.path.join(dir_name,f_name),"r")
#     samp_f=df.Function(V,name="parameter")
#     samp=np.zeros((num_samp,V.dim()))
#     prog=np.ceil(num_samp*(.1+np.arange(0,1,.1)))
#     for s in xrange(num_samp):
#         f.read(samp_f,'sample_{0}'.format(s))
#         samp[s,]=samp_f.vector()
#         if s+1 in prog:
#             print('{0:.0f}% samples have been retrieved.'.format(np.float(s+1)/num_samp*100))
# #     f_read=lambda s: retrieve_each_sample(f,samp_f,s)
# #     samp=Parallel(n_jobs=4)(delayed(f_read)(i) for i in range(num_samp))
#     f.close()
#     return samp

def retrieve_ensemble(mpi_comm,V,dir_name,f_name,ensbl_sz,max_iter,im_shape):
    f=df.HDF5File(mpi_comm,os.path.join(dir_name,f_name),"r")
    ensbl_f=df.Function(V)
    num_ensbls=max_iter*ensbl_sz
    imag=np.zeros((num_ensbls,)+im_shape)
    prog=np.ceil(num_ensbls*(.1+np.arange(0,1,.1)))
    for n in range(max_iter):
        for j in range(ensbl_sz):
            f.read(ensbl_f,'iter{0}_ensbl{1}'.format(n+1,j))
            s=n*ensbl_sz+j
            imag[s,:,:]=fun2img(ensbl_f)
            if s+1 in prog:
                print('{0:.0f}% ensembles have been retrieved.'.format(np.float(s+1)/num_ensbls*100))
    f.close()
    return imag

if __name__ == '__main__':
    from Elliptic import Elliptic
    # define the inverse problem
    np.random.seed(2020)
    nx=40; ny=40
    SNR=100
    elliptic = Elliptic(nx=nx,ny=ny,SNR=SNR)
    # algorithms
    algs=['EKI','EKS']
    num_algs=len(algs)
    # preparation for estimates
    folder = './analysis_f_SNR'+str(SNR)
    hdf5_files=[f for f in os.listdir(folder) if f.endswith('.h5')]
    pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
    ensbl_sz=100
    max_iter=20
    im_shape=(nx+1,ny+1)
    
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
                    imag=retrieve_ensemble(elliptic.pde.mpi_comm,elliptic.pde.V,folder,f_i,ensbl_sz,max_iter,im_shape)
                    print(f_i+' has been read!')
                    found=True; break
                except:
                    pass
        if found and PLOT:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8,8), facecolor='white')
            ax = fig.add_subplot(111, frameon=False)
            plt.ion()
            plt.show(block=False)
            for t in range(imag.shape[0]):
                plt.cla()
                plt.imshow(imag[t],origin='lower',extent=[0,1,0,1])
                plt.title('Ensemble {}'.format(t))
                plt.show()
                plt.pause(1.0/100.0)
        # forward outputs
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
            np.savez_compressed(file=os.path.join(folder,algs[a]+'_ensbl'+str(ensbl_sz)+'_training'),X=imag,Y=fwdout)
#             # how to load
#             loaded=np.load(file=os.path.join(folder,algs[a]+'_ensbl'+str(ensbl_sz)+'_training.npz'))
#             X=loaded['X']
#             Y=loaded['Y']