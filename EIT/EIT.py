#!/usr/bin/env python
"""
3d electrical-impedance tomography (EIT) using pyEIT package
------------------------------------------------------------
Refer to https://github.com/liubenyuan/pyEIT
---------------------
Shiwei Lan @ ASU 2020
---------------------
created Aug. 10, 2020
"""
from __future__ import division, absolute_import, print_function
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2020, The NN-MCMC project"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
from scipy import sparse as sps
import pyeit.mesh as mesh
from pyeit.mesh import quality
import pyeit.mesh.plot as mplot
# from pyeit.eit.fem import Forward
from fem_ import Forward
from pyeit.eit.interp2d import *
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.jac as jac
import sys
sys.path.append( '../' )
from util.sparse_geeklet import sparse_cholesky
import os,pickle
try:
    from joblib import Parallel, delayed
    N_JOB=-2 # all but one
except:
    print('WARNING: no parallel environment found.')

class EIT:
    """
    electrical-impedance tomography (EIT)
    """
    def __init__(self,n_el=16,bbox=None,meshsz=0.1,el_dist=7,step=1,anomaly=None,nz_var=1.,lamb=1.,**kwargs):
        self.n_el=n_el
        self.bbox=bbox
        if self.bbox is None: self.bbox=[[-1,-1,-1], [1,1,1]]
        self.meshsz=meshsz
        self.el_dist,self.step=el_dist,step
        self.anomaly=anomaly
        if self.anomaly is None: self.anomaly=[{'x': 0.40, 'y': 0.40, 'z': 0.0, 'd': 0.30, 'perm': 100.0}]
        self.nz_var=nz_var
        # set up pde
        self.set_pde()
        self.gdim=self.pts.shape[1]
        self.dim=self.tri.shape[0]
        print('Physical PDE model is defined.\n')
        # set up prior
        self.lamb=lamb
        pr_mean=kwargs.pop('pr_mean',np.zeros(self.dim))
        pr_cov=kwargs.pop('pr_cov',sps.eye(self.dim)/self.lamb)
        self.prior={'mean':pr_mean,'cov':pr_cov}
        self.prior['sample']=lambda num_samp=1:self.sample(num_samp=num_samp)
        print('Prior model is specified.\n')
        # set up misfit
        self.set_misfit(**kwargs)
        print('Likelihood model is obtained.\n')
    
    def set_pde(self):
        # construct mesh
        self.mesh_obj,self.el_pos=mesh.create(self.n_el, bbox=self.bbox, h0=self.meshsz)
        # extract node, element
        self.pts=self.mesh_obj['node']
        self.tri=self.mesh_obj['element']
        # initialize forward solver using the unstructured mesh object and the positions of electrodes
        self.fwd=Forward(self.mesh_obj, self.el_pos)
        # boundary condition
        self.ex_mat=eit_scan_lines(self.n_el,self.el_dist)
    
    def _soln_fwd(self,ex_line,perm=None,**kwargs):
        """
        Compute the potential distribution
        """
        if perm is None: perm=self.true_perm
        parser=kwargs.pop('parser','std')
        f,_=self.fwd.solve(ex_line,self.step,perm=perm, parser=parser, **kwargs)
        return f
    
    def solve(self,perm=None,**kwargs):
        """
        EIT simulation, generate perturbation matrix and forward output v
        """
        if perm is None: perm=self.true_perm
        parser=kwargs.pop('parser','std')
        fs=self.fwd.solve_eit(self.ex_mat,self.step,perm=perm, parser=parser, **kwargs)
        return fs
    
    def forward(self,input,n_jobs=N_JOB):
#         import timeit
        try:
            solve_i=lambda u: self.solve(perm=u,skip_jac=True).v
#             t_start=timeit.default_timer()
            output=Parallel(n_jobs=n_jobs)(delayed(solve_i)(u) for u in input)
#             print('Time consumed: {}'.format(timeit.default_timer()-t_start))
        except:
#             t_start=timeit.default_timer()
            output=np.array([self.solve(perm=u,skip_jac=True).v for u in input])
#             print('Time consumed: {}'.format(timeit.default_timer()-t_start))
#         print(np.allclose(output,output1))
        return output
    
    def get_obs(self,**kwargs):
        folder=kwargs.pop('folder','./result')
        fname=str(self.gdim)+'d_EIT_dim'+str(self.dim)
        try:
            with open(os.path.join(folder,fname+'.pckl'),'rb') as f:
                [self.true_perm,obs]=pickle.load(f)[:2]
            print('Data loaded!')
        except:
            print('No data found. Generate new data...')
            mesh_new=mesh.set_perm(self.mesh_obj, anomaly=self.anomaly, background=1.0)
            self.true_perm=mesh_new['perm']
            fs=self.solve(self.true_perm,skip_jac=True,**kwargs)
            obs=fs.v
            if not os.path.exists(folder): os.makedirs(folder)
            with open(os.path.join(folder,fname+'.pckl'),'wb') as f:
                pickle.dump([self.true_perm,obs,self.n_el,self.bbox,self.meshsz,self.el_dist,self.step,self.anomaly,self.lamb],f)
        return obs
    
    def set_misfit(self,obs=None,**kwargs):
        self.obs=obs
        if self.obs is None: self.obs=self.get_obs(**kwargs)
        if np.size(self.nz_var)<np.size(self.obs): self.nz_var=np.resize(self.nz_var,np.size(self.obs))
    
    def get_geom(self,unknown,geom_ord=[0],whitened=False,**kwargs):
        loglik=None; gradlik=None; metact=None; rtmetact=None; eigs=None
        
        if whitened:
            unknown=self.prior['cov'].dot(unknown)
        
        if any(s>=0 for s in geom_ord):
            fs=self.solve(unknown,skip_jac=not any(s>0 for s in geom_ord))
            loglik = -0.5*np.sum((self.obs-fs.v)**2)
            if whitened:
#                 cholC = np.linalg.cholesky(self.prior['cov'])
                L,P=sparse_cholesky(self.prior['cov']); cholC=P.dot(L)
                gradlik = cholC.T.dot(gradlik)
        
        if any(s>=1 for s in geom_ord):
            jac=-fs.jac # pyeit.eit.fem returns jacobian of residual: d(v-f)/dsigma = -df/dsigma
            gradlik = np.dot(jac.T,self.obs-fs.v)
        
        if any(s>=1.5 for s in geom_ord):
            _get_metact_misfit=lambda u_actedon: jac.T.dot(jac.dot(u_actedon)) # GNH
            _get_rtmetact_misfit=lambda u_actedon: jac.T.dot(u_actedon)
            metact = _get_metact_misfit
            rtmetact = _get_rtmetact_misfit
            if whitened:
                metact = lambda u: cholC.T.dot(_get_metact_misfit(cholC.dot(u))) # ppGNH
                rtmetact = lambda u: cholC.T.dot(_get_rtmetact_misfit(u))
        
        if any(s>1 for s in geom_ord) and len(kwargs)!=0:
            if whitened:
                # generalized eigen-decomposition (_C^(1/2) F _C^(1/2), M), i.e. _C^(1/2) F _C^(1/2) = M V D V', V' M V = I
                eigs = geigen_RA(metact, lambda u: u, lambda u: u, dim=self.dim,**kwargs)
            else:
                # generalized eigen-decomposition (F, _C^(-1)), i.e. F = _C^(-1) U D U^(-1), U' _C^(-1) U = I, V = _C^(-1/2) U
                eigs = geigen_RA(metact,lambda u: np.linalg.solve(self.prior['cov'],u),lambda u: self.prior['cov'].dot(u),dim=self.dim,**kwargs)
            if any(s>1.5 for s in geom_ord):
                # adjust the gradient
                # update low-rank approximate Gaussian posterior
                self.post_Ga = Gaussian_apx_posterior(self.prior,eigs=eigs)
                Hu= self.post_Ga['Hlr'].dot(unknown)
                gradlik+=Hu
        
        if len(kwargs)==0:
            return loglik,gradlik,metact,rtmetact
        else:
            return loglik,gradlik,metact,eigs
    
    def get_MAP(self,lamb_decay=1.,lamb_min=1e-5,maxiter=20,verbose=True,**kwargs):
        map=jac.JAC(self.mesh_obj,self.el_pos,self.ex_mat,self.step,perm=1.0,parser='std')
        map.setup(p=kwargs.pop('p',0.25),lamb=kwargs.pop('lamb',self.lamb),method=kwargs.pop('method','lm'))
        ds=map.gn(self.obs,lamb_decay=lamb_decay,lamb_min=lamb_min,maxiter=maxiter,verbose=verbose,**kwargs)
        return ds
    
    def sample(self,num_samp=1,type='prior'):
        """
        Generate sample
        """
        samp=None
        if type=='prior':
            samp=np.random.randn(num_samp,self.dim)
            if (self.prior['cov']!=sps.eye(self.dim)).nnz!=0:
                L,P=sparse_cholesky(self.prior['cov'])
                samp=P.dot(L.dot(samp.T)).T
            if any(self.prior['mean']): samp+=self.prior['mean']
        return np.squeeze(samp)
    
    def plot(self,perm=None,**kwargs):
        import matplotlib.pylab as plt
        if perm is None: perm=self.true_perm
        if 'ax' in kwargs:
            ax=kwargs.pop('ax')
            if self.gdim==2:
                im=ax.tripcolor(self.pts[:,0],self.pts[:,1],self.tri,np.real(perm),shading='flat')
            elif self.gdim==3:
                plt.axes(ax)
                im=mplot.tetplot(self.pts,self.tri,vertex_color=sim2pts(self.pts,self.tri,np.real(perm)),alpha=1.0)
            return im
        else:
            if self.gdim==2:
                plt.tripcolor(self.pts[:,0],self.pts[:,1],self.tri,np.real(perm),shading='flat')
            elif self.gdim==3:
                mplot.tetplot(self.pts,self.tri,vertex_color=sim2pts(self.pts,self.tri,np.real(perm)),alpha=1.0)
            plt.show()
    
    def vec2img(self,perm,imsz=None,**kwargs):
        """
        Convert vector over mesh to image as a matrix
        ---------------------------------------------
        (2D only)
        """
        if imsz is None: imsz = np.ceil(np.sqrt(np.size(perm))).astype('int')
#         mask = kwargs.pop('mask',None)
#         wt_v2i = kwargs.pop('wt_v2i',None)
        if not all(hasattr(self, att) for att in ['mask','wt_v2i']):
            xg, yg, self.mask = meshgrid(self.pts, n=imsz, gc=kwargs.pop('gc',True), **kwargs)
#             im = np.ones_like(mask)
            # mapping from values on xy to values on xyi
            xy = np.mean(self.pts[self.tri], axis=1)
            xyi = np.vstack((xg.flatten(), yg.flatten())).T
            # self.wt_v2i = weight_idw(xy, xyi)
            self.wt_v2i = weight_sigmod(xy, xyi, ratio=.01, s=100)
        im = np.dot(self.wt_v2i.T, perm)
        # im = weight_linear_rbf(xy, xyi, perm)
        im[self.mask] = 0.
        # reshape to grid size
        im = im.reshape((imsz,)*2)
        return im
    
    def img2vec(self,im,**kwargs):
        """
        Convert image matrix to vector value over mesh
        ----------------------------------------------
        (2D only)
        """
        im = im.ravel()
#         mask = kwargs.pop('mask',None)
#         wt_i2v = kwargs.pop('wt_i2v',None)
        if not all(hasattr(self, att) for att in ['mask','wt_i2v']):
            imsz = im.shape[0]
            xg, yg, self.mask = meshgrid(self.pts, n=imsz, gc=kwargs.pop('gc',True), **kwargs)
            # mapping from values on xyi to values on xy
            xy = np.mean(self.pts[self.tri], axis=1)
            xyi = np.vstack((xg.flatten(), yg.flatten())).T
            # self.wt_i2v = weight_idw(xyi, xy)
            self.wt_i2v = weight_sigmod(xyi, xy, ratio=.01, s=100)
        im[self.mask] = 0.
        perm = np.dot(self.wt_i2v.T, im)
        return perm
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.rcParams['image.cmap'] = 'jet'
    import sys
    sys.path.append( "../" )
    from util.common_colorbar import common_colorbar
    np.random.seed(2020)
    
    # define inverse problem
    n_el = 16
    bbox = [[-1,-1],[1,1]]
    meshsz = .05
#     meshsz = .2
    el_dist, step = 1, 1
#     el_dist, step = 7, 1
    anomaly = [{'x': 0.4, 'y': 0.4, 'd': 0.2, 'perm': 10},
               {'x': -0.4, 'y': -0.4, 'd': 0.2, 'perm': 0.1}]
#     anomaly = None
    eit=EIT(n_el=n_el,bbox=bbox,meshsz=meshsz,el_dist=el_dist,step=step,anomaly=anomaly,lamb=1)
    
    
    # check gradient
#     u=eit.true_perm
    u=eit.prior['sample']()
    f,g=eit.get_geom(u,geom_ord=[0,1])[:2]
    v=eit.prior['sample']()
    h=1e-8
    gv_fd=(eit.get_geom(u+h*v)[0]-f)/h
    reldif=abs(gv_fd-g.dot(v.T))/np.linalg.norm(v)
    print('Relative difference between finite difference and exacted results: {}'.format(reldif))
    
    
    # check image conversion
#     demo()
    if eit.gdim==2:
        fig,axes = plt.subplots(nrows=1,ncols=3,sharex=True,sharey=False,figsize=(16,5))
        ax=axes.flat[0]
#         ax.set_aspect('equal')
        subfig=eit.plot(ax=ax)
        ax.set_title(r'Original')
        cax = fig.add_axes([ax.get_position().x1+0.002,ax.get_position().y0,0.01,ax.get_position().height])
        fig.colorbar(subfig, cax=cax)
        
        ax=axes.flat[1]
#         ax.set_aspect('equal')
        im=eit.vec2img(eit.true_perm)
#         subfig=ax.imshow(im,origin='lower')
        ax.triplot(eit.pts[:, 0], eit.pts[:, 1], eit.tri, alpha=0.5)
        xg, yg, mask = meshgrid(eit.pts,n=im.shape[0])
        subfig=ax.pcolor(xg, yg, im, edgecolors=None, linewidth=0, alpha=0.8)
        ax.set_title(r'Converted Image')
        cax = fig.add_axes([ax.get_position().x1+0.005,ax.get_position().y0,0.01,ax.get_position().height])
        fig.colorbar(subfig, cax=cax)
        
        ax=axes.flat[2]
#         ax.set_aspect('equal')
        perm_rec=eit.img2vec(im)
        subfig=eit.plot(perm_rec,ax=ax)
        ax.set_title(r'Recovered')
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.01,ax.get_position().height])
        fig.colorbar(subfig, cax=cax)
        
        plt.subplots_adjust(wspace=0.4, hspace=0)
        plt.savefig(os.path.join('./result',str(eit.gdim)+'d_image_conversion_dim'+str(eit.dim)+'.png'),bbox_inches='tight')
    
    
#     # obtain MAP as reconstruction of permittivity
#     ds=eit.get_MAP(lamb_decay=0.1,lamb=1e-3, method='kotre',maxiter=100)
# #     ds=eit.get_MAP(lamb_decay=0.2,lamb=1e-2, method='kotre')
#     with open(os.path.join('./result',str(eit.gdim)+'d_MAP_dim'+str(eit.dim)+'.pckl'),'wb') as f:
#         pickle.dump([ds,n_el,bbox,meshsz,el_dist,step,anomaly],f)
#     
#     # plot MAP results
#     if eit.gdim==2:
#         fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=False,figsize=(12,5))
#         sub_figs=[None]*2
#         sub_figs[0]=eit.plot(ax=axes.flat[0])
#         axes.flat[0].axis('equal')
#         axes.flat[0].set_title(r'True Conductivities')
#         sub_figs[1]=eit.plot(perm=ds,ax=axes.flat[1])
#         axes.flat[1].axis('equal')
#         axes.flat[1].set_title(r'Reconstructed Conductivities (MAP)')
#         from util.common_colorbar import common_colorbar
#         fig=common_colorbar(fig,axes,sub_figs)
#     #     plt.subplots_adjust(wspace=0.2, hspace=0)
#         # save plots
#         # fig.tight_layout(h_pad=1)
#         plt.savefig(os.path.join('./result',str(eit.gdim)+'d_reconstruction_dim'+str(eit.dim)+'.png'),bbox_inches='tight')
#         # plt.show()
#     else:
#         eit.plot()
#         eit.plot(perm=ds)