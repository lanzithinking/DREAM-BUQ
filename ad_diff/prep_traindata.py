"""
Extract functions stored in hdf5 files and prepare them as training data
Shiwei Lan @ ASU, November 2020
"""

import dolfin as df
import numpy as np

import os, pickle

TRAIN={0:'XimgY',1:'XY'}[0]
whiten=False

def retrieve_ensemble(bip,dir_name,f_name,ensbl_sz,max_iter,img_out=False,whiten=False):
    f=df.HDF5File(bip.pde.mpi_comm,os.path.join(dir_name,f_name),"r")
    ensbl_f=df.Function(bip.prior.V)
    num_ensbls=max_iter*ensbl_sz
    if img_out:
        gdim = bip.prior.V.mesh().geometry().dim()
        imsz = bip.meshsz if hasattr(bip,'meshsz') else (np.floor((bip.prior.V.dim()/bip.prior.V.ufl_element().degree()**2)**(1./gdim)).astype('int'),)*gdim
#         out_shape=(num_ensbls,np.int((bip.prior.V.dim()/bip.prior.V.ufl_element().degree()**2)/imsz**(gdim-1)))+(imsz,)*(gdim-1)
        out_shape=(num_ensbls,)+imsz
    else:
        out_shape=(num_ensbls,bip.mesh.num_vertices())
    out=np.zeros(out_shape)
    prog=np.ceil(num_ensbls*(.1+np.arange(0,1,.1)))
    for n in range(max_iter):
        for j in range(ensbl_sz):
            f.read(ensbl_f,'iter{0}_ensbl{1}'.format(n+('Y' not in TRAIN),j))
            s=n*ensbl_sz+j
            ensbl_v=ensbl_f.vector()
            if whiten: ensbl_v=bip.prior.u2v(ensbl_v)
            out[s]=bip.vec2img(ensbl_v) if img_out else ensbl_f.compute_vertex_values(bip.mesh)
            if s+1 in prog:
                print('{0:.0f}% ensembles have been retrieved.'.format(np.float(s+1)/num_ensbls*100))
    f.close()
    return out

if __name__ == '__main__':
    from advdiff import advdiff
    # define the inverse problem
    seed=2020
    np.random.seed(seed)
#     mesh = df.Mesh('ad_10k.xml')
    mesh = (51,51)
    kappa = 1e-3
    rel_noise = .5
    nref = 1
    adif = advdiff(mesh=mesh, kappa=kappa, rel_noise=rel_noise, nref=nref, seed=seed)
    adif.prior.V=adif.prior.Vh
    # algorithms
    algs=['EKI','EKS']
    num_algs=len(algs)
    # preparation for estimates
    folder = './analysis'
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
                    out=retrieve_ensemble(adif,folder,f_i,ensbl_sz,max_iter,img_out,whiten)
                    print(f_i+' has been read!')
                    found=True; break
                except Exception as e:
                    print(e)
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
            if not os.path.exists(savepath): os.makedirs(savepath)
            ifwhiten='_whitened' if whiten else ''
            if 'Y' in TRAIN:
                np.savez_compressed(file=os.path.join(savepath,algs[a]+'_ensbl'+str(ensbl_sz)+'_training_'+TRAIN+ifwhiten),X=out,Y=fwdout)
            else:
                np.savez_compressed(file=os.path.join(savepath,algs[a]+'_ensbl'+str(ensbl_sz)+'_training_'+TRAIN+ifwhiten),X=out)
#             # how to load
#             loaded=np.load(file=os.path.join(savepath,algs[a]+'_ensbl'+str(ensbl_sz)+'_training_'+TRAIN+'.npz'))
#             X=loaded['X']
#             Y=loaded['Y']