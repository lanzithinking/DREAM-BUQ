'''
Prior of the Advection-Diffusion problem written in FEniCS-2019.1.0 and hIPPYlib-3.0
https://hippylib.github.io/tutorials_v3.0.0/4_AdvectionDiffusionBayesian/
-------------------------------------------------------------------------
Project of Bayesian SpatioTemporal analysis for Inverse Problems (B-STIP)
Shiwei Lan @ ASU, Sept. 2020
--------------------------------------------------------------------------
Created on Sep 23, 2020
'''
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2020, The Bayesian STIP project"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import dolfin as dl
import ufl
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *
from hippylib.modeling.prior import SqrtPrecisionPDE_Prior

import warnings
warnings.simplefilter('once')

class BiLaplacian(SqrtPrecisionPDE_Prior):
    """
    Gaussian prior N(mu,C), assembled(C^(-1)) = R = A * M^(-1) * A
    C = A^{-1} M A^-1
    """
    def __init__(self, Vh, gamma = 1., delta = 8., **kwargs):
        """
        Initialize the BiLaplacian prior
        """
        # function space
        self.Vh = Vh
        self.mpi_comm = self.Vh.mesh().mpi_comm()
        self.rank = dl.MPI.rank(self.mpi_comm)
        # prior parameters
        self.gamma = gamma
        self.delta = delta
        assert delta != 0., "Intrinsic Gaussian Prior are not supported"
        Theta = kwargs.pop('Theta',None)
        robin_bc = kwargs.pop('robin_bc',True)
        def sqrt_precision_varf_handler(trial, test): 
            if Theta == None:
                varfL = ufl.inner(ufl.grad(trial), ufl.grad(test))*ufl.dx
            else:
                varfL = ufl.inner( Theta*ufl.grad(trial), ufl.grad(test))*ufl.dx
            varfM = ufl.inner(trial,test)*ufl.dx
            varf_robin = ufl.inner(trial,test)*ufl.ds
            if robin_bc:
                robin_coeff = gamma*np.sqrt(delta/gamma)/1.42
            else:
                robin_coeff = 0.
            return dl.Constant(gamma)*varfL + dl.Constant(delta)*varfM + dl.Constant(robin_coeff)*varf_robin
        # define mean
        mean = kwargs.pop('mean', dl.interpolate(dl.Constant(0.25), self.Vh).vector())
        super().__init__(self.Vh, sqrt_precision_varf_handler, mean)
        if self.rank == 0:
            print( "Prior regularization: (delta - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(self.delta, self.gamma,2) )
    
    def gen_vector(self, v=None):
        """
        generic function to generate a vector in the PARAMETER space
        """
        if v is None:
            vec = dl.Vector(self.mpi_comm)
            self.init_vector(vec, 0)
        else:
            if isinstance(v, dl.Vector):
                vec = dl.Vector(v)
            elif isinstance(v, np.ndarray):
                vec = self.gen_vector()
                vec.set_local(v)
            else:
                df.warning('Unknown type.')
                vec=None
        return vec
    
    def cost(self, x):
        """
        negative log-prior
        """
        Rdx = dl.Vector()
        self.init_vector(Rdx,0)
        dx = x[PARAMETER] - self.mean
        self.applyR(dx, Rdx)
        reg = .5*Rdx.inner(dx)
        return reg
    
    def grad(self, x, out):
        """
        gradient of negative log-prior
        """
        out.zero()
        dx = x[PARAMETER] - self.mean
        self.applyR(dx, out)
    
    def applyR(self, dm, out):
        """
        apply C^{-1}: out = C{-1} dm
        """
        self.R.mult(dm,out)
    
    def sample(self, whiten=False, add_mean=False):
        """
        Generate a prior sample
        """
        noise = dl.Vector()
        self.init_vector(noise,"noise")
        parRandom.normal(1., noise)
        rhs = self.sqrtM*noise
        u_vec = dl.Vector()
        self.init_vector(u_vec, 0)
        if whiten:
            self.Msolver.solve(u_vec, rhs)
        else:
            self.Asolver.solve(u_vec, rhs)
        
        if add_mean:
            u_vec.axpy(1., self.mean)
        return u_vec
    
    def logpdf(self, u, whiten=False, add_mean=False, grad=False):
        """
        Compute the logarithm of prior density (and its gradient) of function u.
        """
        u_m=dl.Vector(u)
        if add_mean:
            u_m.axpy(-1.,self.u2v(self.mean) if whiten else self.mean)
        
        Pu_m=dl.Vector()
        self.init_vector(Pu_m,0)
        if whiten:
            self.M.mult(u_m,Pu_m)
        else:
            self.applyR(u_m,Pu_m)
        
        logpri=-0.5*Pu_m.inner(u_m)
        if grad:
            gradpri=-Pu_m
            return logpri,gradpri
        else:
            return logpri
    
    def C_act(self, u_actedon, comp=1, transp=False):
        """
        Compute operation of C^comp on vector u: u --> C^comp * u
        """
        if comp==0:
            return u_actedon
        else:
            Cu=dl.Vector()
            self.init_vector(Cu, np.ceil((np.sign(comp)+1)/2).astype('int'))
            if comp==1: self.Rsolver.solve(Cu, u_actedon)
            elif comp==-1: self.applyR(u_actedon, Cu)
            elif comp==0.5:
                if not transp:
                    self.Asolver.solve(Cu, self.M*u_actedon)
                else:
                    self.Asolver.solve(self.R.help1,u_actedon)
                    self.M.mult(self.R.help1, Cu)
            elif comp==-0.5:
                if transp:
                    self.Msolver.solve(Cu, self.A*u_actedon)
                else:
                    self.Msolver.solve(self.Rsolver.help1,u_actedon)
                    self.A.mult(self.Rsolver.help1, Cu)
            else: 
                warnings.warn('Action not defined!')
                Cu=None
        return Cu
    
    def u2v(self, u, u_ref=None):
        """
        v:=C^(-1/2) (u-u_ref)
        """
        v = dl.Vector()
        self.init_vector(v,1)
        b = u if u_ref is None else u - u_ref
        self.Msolver.solve(v, self.A*b)
        return v
    
    def v2u(self, v, u_ref=None):
        """
        u = u_ref + C^(1/2) v
        """
        u = dl.Vector()
        self.init_vector(u,1)
        self.Asolver.solve(u, self.M*v)
        if u_ref is not None:
            u.axpy(1., u_ref)
        return u
    
if __name__=='__main__':
    from pde import *
    np.random.seed(2020)
    # obtain function space
#     mesh = dl.Mesh('ad_10k.xml')
#     Vh = dl.FunctionSpace(mesh, "Lagrange", 2)
    meshsz = (61,61)
    eldeg = 1
    pde = TimeDependentAD(mesh=meshsz, eldeg=eldeg)
    Vh = pde.Vh[STATE]
    # define prior
    gamma = 2.; delta = 10.
    prior = BiLaplacian(Vh, gamma=gamma, delta=delta)
    
    # tests
    whiten=False
    u=prior.sample(whiten=whiten)
    logpri,gradpri=prior.logpdf(u, whiten=whiten, grad=True)
    print('The logarithm of prior density at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(logpri,gradpri.norm('l2')))
    fig=dl.plot(vector2Function(u,prior.Vh))
    plt.colorbar(fig); plt.show()
    v=prior.u2v(u)
    logpri_wt,gradpri_wt=prior.logpdf(v, whiten=True, grad=True)
    print('The logarithm of prior density at whitened v is %0.4f, and the L2 norm of its gradient is %0.4f' %(logpri_wt,gradpri_wt.norm('l2')))
    fig=dl.plot(vector2Function(v,prior.Vh))
    plt.colorbar(fig); plt.show()
    
    whiten=True
#     v=prior.sample(whiten=whiten)
    logpri,gradpri=prior.logpdf(v, whiten=whiten, grad=True)
    print('The logarithm of prior density at whitened v is %0.4f, and the L2 norm of its gradient is %0.4f' %(logpri,gradpri.norm('l2')))
    fig=dl.plot(vector2Function(v,prior.Vh))
    plt.colorbar(fig); plt.show()
    u=prior.v2u(v)
    logpri_wt,gradpri_wt=prior.logpdf(u, whiten=False, grad=True)
    print('The logarithm of prior density at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(logpri_wt,gradpri_wt.norm('l2')))
    fig=dl.plot(vector2Function(u,prior.Vh))
    plt.colorbar(fig); plt.show()