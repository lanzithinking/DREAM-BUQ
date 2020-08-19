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
__version__ = "0.6"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
from scipy import sparse as sps
import scipy.spatial.distance as spd
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
from util.sparse_geeklet import csr_trim0,sparse_cholesky
import os,pickle
try:
    from joblib import Parallel, delayed
    N_JOB=-2 # all CPUs but one
except:
    N_JOB=1
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
        self.lamb=lamb
        # set up pde
        self.set_pde()
        self.dim,self.gdim=self.pts.shape
#         self.M=self.fwd.get_mass()
        print('Physical PDE model is defined.\n')
        # set up prior
        pr_mean=kwargs.pop('pr_mean',np.zeros(self.dim))
#         pr_cov=kwargs.pop('pr_cov',sps.eye(self.dim)/self.lamb)
        pr_cov=kwargs.pop('pr_cov',self.get_ker(sigma=kwargs.pop('sigma',1),rho=kwargs.pop('rho',.05))/self.lamb)
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
         # count PDE solving times
        self.soln_count = np.zeros(2)
    
    def _soln_fwd(self,ex_line,perm=None,**kwargs):
        """
        Compute the potential distribution
        """
        if perm is None: perm=self.true_perm
        parser=kwargs.pop('parser','std')
        f,_=self.fwd.solve(ex_line,perm=perm, **kwargs)
        return np.real(f)
    
    def solve(self,perm=None,**kwargs):
        """
        EIT simulation, generate perturbation matrix and forward output v
        """
        if perm is None: perm=self.true_perm
        parser=kwargs.pop('parser','std')
        skip_jac=kwargs.pop('skip_jac',False)
        fs=self.fwd.solve_eit(self.ex_mat,self.step,perm=perm, parser=parser, skip_jac=skip_jac, **kwargs)
#         if not all(np.isreal(fs.v)): print('Oh! Non-real output!')
        self.soln_count[0]+=1; self.soln_count[1]+=not skip_jac
        return fs
    
    def _pts2sim(self,pts_values):
#         el_value = pts2sim(self.tri, pts_values)
        el_value = np.mean(pts_values[...,self.tri], axis=np.ndim(pts_values))
        return el_value
    
    def _sim2pts(self,sim_values):
        return sim2pts(self.pts, self.tri, sim_values)
    
    def forward(self,input,n_jobs=N_JOB):
#         import timeit
        par_run=np.ndim(input)>1 and input.shape[0]>1
        if input.shape[-1]!=self.fwd.n_tri: input=self._pts2sim(input)
        if par_run:
            try:
                solve_i=lambda u: self.solve(perm=u,skip_jac=True).v
                n_jobs=min(n_jobs,input.shape[0])
    #             t_start=timeit.default_timer()
                output=Parallel(n_jobs=n_jobs)(delayed(solve_i)(u) for u in input)
    #             print('Time consumed: {}'.format(timeit.default_timer()-t_start))
            except:
                par_run=False
                print('Parallel run failed. Running in series...')
                pass
        if not par_run:
#             t_start=timeit.default_timer()
            output=np.array([self.solve(perm=u,skip_jac=True).v for u in input])
#             print('Time consumed: {}'.format(timeit.default_timer()-t_start))
#         print(np.allclose(output,output1))
        return output
    
    def get_ker(self,**kwargs):
        """
        Get the kernel matrix K with K_ij = k(x_i,x_j).
        """
#         folder=kwargs.pop('folder','./result')
#         fname=str(self.gdim)+'d_EIT_meshpdist_dim'+str(self.dim)
#         try:
#             pDist=np.load(os.path.join(folder,fname+'.npz'))['ker_dist']
#             print('Pairwise distance '+fname+' loaded!')
#         except:
        pDist=spd.pdist(self.pts)
#             if not os.path.exists(folder): os.makedirs(folder)
#             np.savez_compressed(file=os.path.join(folder,fname),ker_dist=pDist)
        sigma=kwargs.pop('sigma',1.)
        rho=kwargs.pop('rho',.05)
        K = sps.csr_matrix(sigma**2*np.exp(-spd.squareform(pDist)/(2*rho)))
        csr_trim0(K,1e-8)
        return K
    
    def get_obs(self,**kwargs):
        folder=kwargs.pop('folder','./result')
        fname=str(self.gdim)+'d_EIT_dim'+str(self.dim)
        try:
            with open(os.path.join(folder,fname+'.pckl'),'rb') as f:
                [self.true_perm,obs]=pickle.load(f)[:2]
            print('Data '+fname+' loaded!')
        except:
            print('No data found. Generate new data...')
            mesh_new=mesh.set_perm(self.mesh_obj, anomaly=self.anomaly, background=1.0)
            self.true_perm=mesh_new['perm']
            fs=self.solve(self.true_perm,skip_jac=True,**kwargs)
            obs=fs.v
#             if np.size(self.nz_var)<np.size(obs): self.nz_var=np.resize(self.nz_var,np.size(obs))
#             obs+=np.sqrt(self.nz_var)*np.random.randn(np.size(obs)) # voltage must be positive!
            if not os.path.exists(folder): os.makedirs(folder)
            with open(os.path.join(folder,fname+'.pckl'),'wb') as f:
                pickle.dump([self.true_perm,obs,self.n_el,self.bbox,self.meshsz,self.el_dist,self.step,self.anomaly,self.lamb],f)
        return obs
    
    def set_misfit(self,obs=None,**kwargs):
        self.obs=obs
        if self.obs is None: self.obs=self.get_obs(**kwargs)
        if np.size(self.nz_var)<np.size(self.obs): self.nz_var=np.resize(self.nz_var,np.size(self.obs))
    
    def _d_pts2sim(self):
        n_vertices = self.tri.shape[1]
        indptr = np.arange(self.tri.size+1,step=n_vertices,dtype='int')
        return sps.csr_matrix((np.ones(self.tri.size)/n_vertices,self.tri.ravel(),indptr),shape=(self.fwd.n_tri,self.dim))
    
    def get_geom(self,unknown,geom_ord=[0],whitened=False,**kwargs):
        loglik=None; gradlik=None; metact=None; rtmetact=None; eigs=None
        
        if whitened:
            unknown=self.prior['cov'].dot(unknown)
        
        force_posperm=kwargs.pop('force_posperm',False)
        if len(unknown)==self.dim:
            perm=self._pts2sim(unknown) # unknown is a vector of nodal values
        elif len(unknown)==self.fwd.n_tri:
            perm=unknown
        if force_posperm:
            sign_unknown=np.sign(unknown)
            perm=np.abs(perm)
        
        if any(s>=0 for s in geom_ord):
            fs=self.solve(perm=perm,skip_jac=not any(s>0 for s in geom_ord))
            loglik = -0.5*np.sum((self.obs-fs.v)**2/self.nz_var)
        
        if any(s>=1 for s in geom_ord):
            jacT=-self._d_pts2sim().T.dot(fs.jac.T) # pyeit.eit.fem returns jacobian of residual: d(v-f)/dsigma = -df/dsigma
            gradlik = np.dot(jacT,(self.obs-fs.v)/self.nz_var)
            if force_posperm: gradlik*=sign_unknown
#             gradlik = self.M.dot(gradlik)
            if whitened:
#                 cholC = np.linalg.cholesky(self.prior['cov'])
                if not all(key in self.prior for key in ['L','P']):
                    L,P=sparse_cholesky(self.prior['cov'])
                    self.prior['L']=L; self.prior['P']=P
                cholC=self.prior['P'].dot(self.prior['L'])
                gradlik = cholC.T.dot(gradlik)
        
        if any(s>=1.5 for s in geom_ord):
            _get_metact_misfit=lambda u_actedon: jacT.dot(jacT.T.dot(self._pts2sim(u_actedon))/self.nz_var) # GNH
            _get_rtmetact_misfit=lambda u_actedon: jacT.dot(u_actedon/sqrt(self.nz_var))
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
            if not all(key in self.prior for key in ['L','P']):
                L,P=sparse_cholesky(self.prior['cov'])
                self.prior['L']=L; self.prior['P']=P
            samp=self.prior['P'].dot(self.prior['L'].dot(samp.T)).T
            if any(self.prior['mean']): samp+=self.prior['mean']
        return np.squeeze(samp)
    
    def plot(self,input=None,**kwargs):
        import matplotlib.pylab as plt
        if input is None: input=self.true_perm
        if 'ax' in kwargs:
            ax=kwargs.pop('ax')
            if self.gdim==2:
                if len(input)==self.dim: input=self._pts2sim(input)
                im=ax.tripcolor(self.pts[:,0],self.pts[:,1],self.tri,np.real(input),shading='flat')
            elif self.gdim==3:
                if len(input)==self.fwd.n_tri: input=self._sim2pts(input)
                plt.axes(ax)
                im=mplot.tetplot(self.pts,self.tri,vertex_color=np.real(input),alpha=1.0)
            return im
        else:
            if self.gdim==2:
                if len(input)==self.dim: input=self._pts2sim(input)
                plt.tripcolor(self.pts[:,0],self.pts[:,1],self.tri,np.real(input),shading='flat')
            elif self.gdim==3:
                if len(input)==self.fwd.n_tri: input=self._sim2pts(input)
                mplot.tetplot(self.pts,self.tri,vertex_color=np.real(input),alpha=1.0)
            plt.show()
    
    def vec2img(self,input,imsz=None,**kwargs):
        """
        Convert vector over mesh to image as a matrix
        ---------------------------------------------
        (2D only)
        """
        if imsz is None: imsz = np.ceil(np.sqrt(input.shape[-1])).astype('int')
#         mask = kwargs.pop('mask',None)
#         wt_v2i = kwargs.pop('wt_v2i',None)
        if not all(hasattr(self, att) for att in ['mask','wt_v2i']):
            xg, yg, self.mask = meshgrid(self.pts, n=imsz, gc=kwargs.pop('gc',True), **kwargs)
#             im = np.ones_like(mask)
            # mapping from values on xy to values on xyi
            if input.shape[-1]==self.dim:
                xy = self.pts
            elif input.shape[-1]==self.fwd.n_tri:
                xy = np.mean(self.pts[self.tri], axis=1)
            xyi = np.vstack((xg.flatten(), yg.flatten())).T
            # self.wt_v2i = weight_idw(xy, xyi)
            self.wt_v2i = weight_sigmod(xy, xyi, ratio=.01, s=100)
        im = input.dot(self.wt_v2i)
        # im = weight_linear_rbf(xy, xyi, input)
        im[...,self.mask] = 0.
        # reshape to grid size
        im = im.reshape((-1,)+(imsz,)*2 if np.ndim(input)==2 else (imsz,)*2)
        return im
    
    def img2vec(self,im,out_opt='node',**kwargs):
        """
        Convert image matrix to vector value over mesh
        ----------------------------------------------
        (2D only)
        """
        imsz = im.shape[1]
        im = np.squeeze(im.reshape((-1,imsz**2)))
#         mask = kwargs.pop('mask',None)
#         wt_i2v = kwargs.pop('wt_i2v',None)
        if not all(hasattr(self, att) for att in ['mask','wt_i2v']):
            xg, yg, self.mask = meshgrid(self.pts, n=imsz, gc=kwargs.pop('gc',True), **kwargs)
            # mapping from values on xyi to values on xy
            if out_opt=='node':
                xy = self.pts
            elif out_opt=='cell':
                xy = np.mean(self.pts[self.tri], axis=1)
            xyi = np.vstack((xg.flatten(), yg.flatten())).T
            # self.wt_i2v = weight_idw(xyi, xy)
            self.wt_i2v = weight_sigmod(xyi, xy, ratio=.01, s=100)
        im[...,self.mask] = 0.
        output = im.dot(self.wt_i2v)
        return output
    
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
    meshsz = .04
#     meshsz = .2
    el_dist, step = 1, 1
#     el_dist, step = 7, 1
    anomaly = [{'x': 0.4, 'y': 0.4, 'd': 0.2, 'perm': 10},
               {'x': -0.4, 'y': -0.4, 'd': 0.2, 'perm': 0.1}]
#     anomaly = None
    nz_var=1e-2; lamb=1e-1; rho=.25
    eit=EIT(n_el=n_el,bbox=bbox,meshsz=meshsz,el_dist=el_dist,step=step,anomaly=anomaly,nz_var=nz_var,lamb=lamb,rho=rho)
    
    
    # check gradient
#     u=eit.true_perm
    u=eit.prior['sample']()
    f,g=eit.get_geom(u,geom_ord=[0,1],force_posperm=True)[:2]
    v=eit.prior['sample']()
    h=1e-9
    gv_fd=(eit.get_geom(u+h*v,force_posperm=True)[0]-f)/h
    reldif=abs(gv_fd-g.dot(v.T))/np.linalg.norm(v)
    print('Relative difference between finite difference and exacted results: {}'.format(reldif))
    
    
#     # check image conversion
# #     demo()
#     if eit.gdim==2:
#         fig,axes = plt.subplots(nrows=1,ncols=3,sharex=True,sharey=False,figsize=(16,5))
#         ax=axes.flat[0]
# #         ax.set_aspect('equal')
#         subfig=eit.plot(ax=ax)
#         ax.set_title(r'Original')
#         cax = fig.add_axes([ax.get_position().x1+0.002,ax.get_position().y0,0.01,ax.get_position().height])
#         fig.colorbar(subfig, cax=cax)
#           
#         ax=axes.flat[1]
# #         ax.set_aspect('equal')
#         im=eit.vec2img(eit.true_perm)
# #         subfig=ax.imshow(im,origin='lower')
#         ax.triplot(eit.pts[:, 0], eit.pts[:, 1], eit.tri, alpha=0.5)
#         xg, yg, mask = meshgrid(eit.pts,n=im.shape[0])
#         subfig=ax.pcolor(xg, yg, im, edgecolors=None, linewidth=0, alpha=0.8)
#         ax.set_title(r'Converted Image')
#         cax = fig.add_axes([ax.get_position().x1+0.005,ax.get_position().y0,0.01,ax.get_position().height])
#         fig.colorbar(subfig, cax=cax)
#           
#         ax=axes.flat[2]
# #         ax.set_aspect('equal')
#         perm_rec=eit.img2vec(im)
#         subfig=eit.plot(perm_rec,ax=ax)
#         ax.set_title(r'Recovered')
#         cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.01,ax.get_position().height])
#         fig.colorbar(subfig, cax=cax)
#           
#         plt.subplots_adjust(wspace=0.4, hspace=0)
#         plt.savefig(os.path.join('./result',str(eit.gdim)+'d_image_conversion_dim'+str(eit.dim)+'.png'),bbox_inches='tight')
    
    
    # obtain MAP as reconstruction of permittivity
    try:
        with open(os.path.join('./result',str(eit.gdim)+'d_EIT_MAP_dim'+str(eit.dim)+'.pckl'),'rb') as f:
            ds=pickle.load(f)[0]
    except:
        ds=eit.get_MAP(lamb_decay=0.1,lamb=1e-3, method='kotre',maxiter=100)
#         ds=eit.get_MAP(lamb_decay=0.2,lamb=1e-2, method='kotre', maxiter=100)
        with open(os.path.join('./result',str(eit.gdim)+'d_EIT_MAP_dim'+str(eit.dim)+'.pckl'),'wb') as f:
            pickle.dump([ds,n_el,bbox,meshsz,el_dist,step,anomaly],f)
     
    # plot MAP results
    if eit.gdim==2:
        fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=False,figsize=(12,5))
        sub_figs=[None]*2
        sub_figs[0]=eit.plot(ax=axes.flat[0])
        axes.flat[0].axis('equal')
        axes.flat[0].set_title(r'True Conductivities')
        sub_figs[1]=eit.plot(input=ds,ax=axes.flat[1])
        axes.flat[1].axis('equal')
        axes.flat[1].set_title(r'Reconstructed Conductivities (MAP)')
        from util.common_colorbar import common_colorbar
        fig=common_colorbar(fig,axes,sub_figs)
    #     plt.subplots_adjust(wspace=0.2, hspace=0)
        # save plots
        # fig.tight_layout(h_pad=1)
        plt.savefig(os.path.join('./result',str(eit.gdim)+'d_EIT_MAP_dim'+str(eit.dim)+'.png'),bbox_inches='tight')
        # plt.show()
    else:
        eit.plot()
        eit.plot(input=ds)