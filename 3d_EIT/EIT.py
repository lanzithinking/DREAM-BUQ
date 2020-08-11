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
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
from scipy import sparse as sps
import pyeit.mesh as mesh
from pyeit.mesh import quality
import pyeit.mesh.plot as mplot
from pyeit.eit.fem import Forward
from pyeit.eit.interp2d import sim2pts
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.jac as jac
import sys
sys.path.append( '../' )
from util.sparse_geeklet import sparse_cholesky
import os,pickle

class EIT:
    """
    electrical-impedance tomography (EIT)
    """
    def __init__(self,n_el=16,bbox=None,meshsz=0.1,el_dist=7,step=1,anomaly=None,lamb=1.,**kwargs):
        self.n_el=n_el
        self.bbox=bbox
        if self.bbox is None: self.bbox=[[-1,-1,-1], [1,1,1]]
        self.meshsz=meshsz
        self.el_dist,self.step=el_dist,step
        self.anomaly=anomaly
        if self.anomaly is None: self.anomaly=[{'x': 0.40, 'y': 0.40, 'z': 0.0, 'd': 0.30, 'perm': 100.0}]
        # set up pde
        self.set_pde()
        self.gdim=self.pts.shape[1]
        self.dim=self.tri.shape[0]
        print('\nPhysical PDE model is defined.')
        # set up prior
        self.lamb=lamb
        pr_mean=kwargs.pop('pr_mean',np.zeros(self.dim))
        pr_cov=kwargs.pop('pr_cov',sps.eye(self.dim)/self.lamb)
        def pr_samp(n):
            samp=np.random.randn(n,self.dim)
            if (pr_cov!=sps.eye(self.dim)).nnz!=0:
                L,P=sparse_cholesky(pr_cov)
                samp=P.dot(L.dot(samp.T)).T
            return samp
        self.prior={'pr_mean':pr_mean,'pr_cov':pr_cov,'sample':lambda n=1: np.squeeze(pr_samp(n))}
        print('\nPrior model is specified.')
        # set up misfit
        self.set_misfit(**kwargs)
        print('\nLikelihood model is obtained.')
    
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
    
    def soln_fwd(self,ex_line,perm=None,**kwargs):
        """
        Compute the potential distribution
        """
        if perm is None: perm=self.true_perm
        parser=kwargs.pop('parser','std')
        f,_=self.fwd.solve(ex_line,self.step,perm=perm, parser=parser)
        return f
    
    def solve(self,perm=None,**kwargs):
        """
        EIT simulation, generate perturbation matrix and forward output v
        """
        if perm is None: perm=self.true_perm
        parser=kwargs.pop('parser','std')
        fs=self.fwd.solve_eit(self.ex_mat,self.step,perm=perm, parser=parser)
        return fs
    
    def get_obs(self,**kwargs):
        folder=kwargs.pop('folder','./result')
        fname=str(self.gdim)+'d_EIT_dim'+str(self.dim)
        try:
            with open(os.path.join(folder,fname+'.pckl'),'rb') as f:
                [self.true_perm,obs]=pickle.load(f)
            print('Data loaded!\n')
        except:
            print('No data found. Generate new data...\n')
            mesh_new=mesh.set_perm(self.mesh_obj, anomaly=self.anomaly, background=1.0)
            self.true_perm=mesh_new['perm']
            fs=self.solve(self.true_perm,**kwargs)
            obs=fs.v
            with open(os.path.join(folder,fname+'.pckl'),'wb') as f:
                pickle.dump([self.true_perm,obs],f)
        return obs
    
    def set_misfit(self,obs=None,**kwargs):
        self.obs=obs
        if self.obs is None: self.obs=self.get_obs(**kwargs)
    
    def get_geom(self,unknown,geom_ord=[0],whitened=False,**kwargs):
        loglik=None; gradlik=None; metact=None; rtmetact=None; eigs=None
        
        if whitened:
            unknown=self.prior['cov'].dot(unknown)
        
        if any(s>=0 for s in geom_ord):
            fs=self.solve(unknown)
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
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os,sys
    sys.path.append( "../" )
    from util.common_colorbar import common_colorbar
    np.random.seed(2020)
    
    # define inverse problem
    n_el = 16
    bbox = [[-1,-1],[1,1]]
    meshsz = .1
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
    h=1e-12
    gv_fd=(eit.get_geom(u+h*v)[0]-f)/h
    reldif=abs(gv_fd-g.dot(v.T))/np.linalg.norm(v)
    print('Relative difference between finite difference and exacted results: {}'.format(reldif))
    
    # obtain MAP as reconstruction of permittivity
    ds=eit.get_MAP(lamb_decay=0.1,lamb=1e-3, method='kotre')
    
    # plot results
    fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=False,figsize=(12,5))
    sub_figs=[None]*2
    sub_figs[0]=eit.plot(ax=axes.flat[0])
    axes.flat[0].axis('equal')
    axes.flat[0].set_title(r'True Conductivities')
    sub_figs[1]=eit.plot(perm=ds,ax=axes.flat[1])
    axes.flat[1].axis('equal')
    axes.flat[1].set_title(r'Reconstructed Conductivities (MAP)')
    from util.common_colorbar import common_colorbar
    fig=common_colorbar(fig,axes,sub_figs)
#     plt.subplots_adjust(wspace=0.2, hspace=0)
    # save plots
    # fig.tight_layout(h_pad=1)
    plt.savefig(os.path.join('./result/'+str(eit.gdim)+'d_reconstruction_dim'+str(eit.dim)+'.png'),bbox_inches='tight')
    # plt.show()