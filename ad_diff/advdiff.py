'''
Advection-Diffusion inverse problem
Created on Sep 23, 2020
-------------------------------------------------------------------------
https://hippylib.github.io/tutorials_v3.0.0/4_AdvectionDiffusionBayesian/
------------------
@author: shiweilan
'''

import dolfin as dl
import ufl
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *

# sys.path.append( "../" )
# from util import *
from pde import *
from prior import *
from misfit import *
# from posterior import *
from whiten import *
from randomizedEigensolver_ext import *

class advdiff(TimeDependentAD,SpaceTimePointwiseStateObservation):
    def __init__(self, mesh=None, gamma = 1., delta = 8., simulation_times=None, observation_times=None, targets=None, rel_noise=0.01, **kwargs):
        """
        Initialize the inverse advection-diffusion problem by defining the physical PDE model, the prior model and the misfit (likelihood) model.
        """
        # get the mesh
        self.mesh = dl.Mesh('ad_10k.xml') if mesh is None else mesh
        self.mpi_comm = self.mesh.mpi_comm()
        self.rank = dl.MPI.rank(self.mpi_comm)
        self.nproc = dl.MPI.size(self.mpi_comm)
        # parameters for prior 
        self.gamma = gamma
        self.delta = delta
        # parameters for misfit
        if simulation_times is None or observation_times is None:
            t_init         = 0.
            t_final        = 4.
            dt             = .1
        if simulation_times is None:
            self.simulation_times = np.arange(t_init, t_final+.5*dt, dt)
        else:
            self.simulation_times = simulation_times
        if observation_times is None:
            t_1            = 1.
            observation_dt = .2
            self.observation_times = np.arange(t_1, t_final+.5*dt, observation_dt)
        else:
            self.observation_times = observation_times
        self.targets = np.loadtxt('targets.txt') if targets is None else targets
        self.rel_noise = rel_noise
        
        # define the inverse problem with pde, prior, and misfit
        seed = kwargs.pop('seed',2020)
        self.setup(seed,**kwargs)
        # Part of model public API
        self.gauss_newton_approx = False
        # initialize the joint state vector
        self.x = self.generate_vector()
        
    def setup(self,seed=2020,**kwargs):
        # set (common) random seed
        Random.seed=seed
        if self.nproc > 1:
            Random.split(self.rank, self.nproc, 1000000)
        np.random.seed(seed)
        sep = "\n"+"#"*80+"\n"
        # set pde
        if self.rank == 0: print(sep, 'Define the physical PDE model.', sep)
        self.pde = TimeDependentAD(self.mesh, self.simulation_times, **kwargs)
        # set prior
        if self.rank == 0: print(sep, 'Specify the prior model.', sep)
        self.prior = BiLaplacian(Vh=self.pde.Vh[PARAMETER], gamma=self.gamma, delta=self.delta, **kwargs)
        self.whtprior = wht_prior(self.prior)
        # set misfit
        if self.rank == 0: print(sep, 'Obtain the likelihood model.', sep)
        self.misfit = SpaceTimePointwiseStateObservation(Vh=self.pde.Vh[STATE], observation_times=self.observation_times, targets=self.targets, pde=self.pde, rel_noise=self.rel_noise, **kwargs)
#         # set low-rank approximate Gaussian posterior
#         if self.rank == 0: print(sep, 'Set the approximate posterior model.', sep)
#         self.post_Ga = Gaussian_apx_posterior(self.prior, eigs='hold')
    
    def generate_vector(self, component = "ALL"):
        if component == "ALL":
            u = TimeDependentVector(self.simulation_times)
            u.initialize(self.pde.M, 0)
            m = dl.Vector()
            self.prior.init_vector(m,0)
            p = TimeDependentVector(self.simulation_times)
            p.initialize(self.pde.M, 0)
            return [u, m, p]
        elif component == STATE:
            u = TimeDependentVector(self.simulation_times)
            u.initialize(self.pde.M, 0)
            return u
        elif component == PARAMETER:
            m = dl.Vector()
            self.prior.init_vector(m,0)
            return m
        elif component == ADJOINT:
            p = TimeDependentVector(self.simulation_times)
            p.initialize(self.pde.M, 0)
            return p
        else:
            raise
    
    def init_parameter(self, m):
        self.prior.init_vector(m,0)
    
    def cost(self, x):
        reg = self.prior.cost(x)
        misfit = self.misfit.cost(x)
        return [reg+misfit, reg, misfit]
    
    def evalGradientParameter(self,x, mg, misfit_only=False):
        self.prior.init_vector(mg,1)
        if misfit_only == False:
            dm = x[PARAMETER] - self.prior.mean
            self.prior.R.mult(dm, mg)
        else:
            mg.zero()
        
        p0 = dl.Vector()
        self.pde.M.init_vector(p0,0)
        x[ADJOINT].retrieve(p0, self.simulation_times[1])
        
        mg.axpy(-1., self.pde.Mt_stab*p0)
        
        g = dl.Vector()
        self.prior.M.init_vector(g,1)
        
        self.prior.Msolver.solve(g,mg)
        
        grad_norm = g.inner(mg)
        
        return grad_norm
    
    def setPointForHessianEvaluations(self, x, gauss_newton_approx=False):
        """
        Specify the point x = [u,a,p] at which the Hessian operator (or the Gauss-Newton approximation)
        need to be evaluated.
        
        Nothing to do since the problem is linear
        """
        self.gauss_newton_approx = gauss_newton_approx
        return
    
    def exportState(self, x, filename, varname):
        out_file = dl.XDMFFile(self.mpi_comm, filename)
        out_file.parameters["functions_share_mesh"] = True
        out_file.parameters["rewrite_function_mesh"] = False
        ufunc = dl.Function(self.pde.Vh[STATE], name=varname)
        t = self.simulation_times[0]
        out_file.write_checkpoint(vector2Function(x[PARAMETER], self.pde.Vh[STATE], name=varname), 'm', t, dl.XDMFFile.Encoding.HDF5, False)
        for t in self.simulation_times[1:]:
            x[STATE].retrieve(ufunc.vector(), t)
            out_file.write_checkpoint(ufunc, 'u', t, dl.XDMFFile.Encoding.HDF5, True)
        out_file.close()
    
    def _get_misfit(self, parameter=None):
        """
        Compute the misfit for given parameter.
        """
        if parameter is None:
            parameter=self.prior.mean
        self.x[PARAMETER] = parameter
        self.pde.solveFwd(self.x[STATE], self.x)
        misfit = self.misfit.cost(self.x)
        return misfit
    
    def _get_grad(self, parameter=None, MF_only=True):
        """
        Compute the gradient of misfit (default), or the gradient of negative log-posterior for given parameter.
        """
        if parameter is None:
            parameter=self.prior.mean
        self.x[PARAMETER] = parameter
        self.pde.solveAdj(self.x[ADJOINT], self.x, self.misfit)
        grad = self.generate_vector(PARAMETER)
        gradnorm = self.evalGradientParameter(self.x, grad, misfit_only=MF_only)
        return grad
    
    def _get_HessApply(self, parameter=None, MF_only=True):
        """
        Compute the Hessian apply (action) for given parameter,
        default to the Gauss-Newton approximation.
        """
        if parameter is None:
            parameter=self.prior.mean
        self.x[PARAMETER] = parameter
        # set point for Hessian evaluation
        self.setPointForHessianEvaluations(self.x)
        self_ = self
        self_.M = self.pde.M
        self_.M_stab = self.pde.M_stab
        self_.Mt_stab = self.pde.Mt_stab
        self_.solver = self.pde.solver
        self_.solvert = self.pde.solvert
        self_.soln_count = self.pde.soln_count
        self_.B = self.misfit.B
        self_.u_snapshot = self.misfit.u_snapshot
        self_.Bu_snapshot = self.misfit.Bu_snapshot
        self_.noise_variance = self.misfit.noise_variance
        if not MF_only:
            self_.R = self.prior.R
            self_.applyR = self.prior.applyR
        HessApply = ReducedHessian(self_, misfit_only=MF_only)
        return HessApply
    
    def get_geom(self,parameter=None,geom_ord=[0],whitened=False,log_level=dl.LogLevel.ERROR,**kwargs):
        """
        Get necessary geometric quantities including log-likelihood (0), adjusted gradient (1), 
        Hessian apply (1.5), and its eigen-decomposition using randomized algorithm (2).
        """
        if parameter is None:
            parameter=self.prior.mean
        loglik=None; agrad=None; HessApply=None; eigs=None;
        # set log level: DBG(10), TRACE(13), PROGRESS(16), INFO(20,default), WARNING(30), ERROR(40), or CRITICAL(50)
        dl.set_log_level(log_level)
        
        # un-whiten if necessary
        if whitened:
            parameter=self.whtprior.v2u(parameter)
        
        # get log-likelihood
        if any(s>=0 for s in geom_ord):
            loglik = -self._get_misfit(parameter)
        
        # get gradient
        if any(s>=1 for s in geom_ord):
            agrad = -self._get_grad(parameter)
            if whitened:
                agrad_ = agrad.copy(); agrad.zero()
                self.whtprior.C_act(agrad_,agrad,comp=0.5,transp=True)
        
        # get Hessian Apply
        if any(s>=1.5 for s in geom_ord):
            HessApply = self._get_HessApply(parameter,kwargs.pop('MF_only',True)) # Hmisfit if MF is true
            if whitened:
                HessApply = wht_Hessian(self.whtprior,HessApply)
            if np.max(geom_ord)<=1.5:
                # adjust the gradient
                Hu = self.generate_vector(PARAMETER)
                HessApply.mult(parameter,Hu)
                agrad.axpy(1.,Hu)
                if not kwargs.pop('MF_only',True):
                    Ru = self.generate_vector(PARAMETER)
#                     self.prior.R.mult(parameter,Ru)
                    self.prior.grad(parameter,Ru)
                    agrad.axpy(-1.,Ru)
        
        # get estimated eigen-decomposition for the Hessian (or Gauss-Newton)
        if any(s>1 for s in geom_ord):
            k=kwargs['k'] if 'k' in kwargs else kwargs['incr_k'] if 'incr_k' in kwargs  else 80
            p=kwargs['p'] if 'p' in kwargs else 20
            if len(kwargs)==0:
                kwargs['k']=k
#             if self.rank == 0:
#                 print('Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.'.format(k,p))
            Omega = MultiVector(parameter, k+p)
            parRandom.normal(1., Omega)
            if whitened:
                eigs = singlePassGx(HessApply, self.prior.M, self.prior.Msolver, Omega, **kwargs)
            else:
                eigs = doublePassG(HessApply, self.prior.R, self.prior.Rsolver, Omega, **kwargs)
            if any(s>1.5 for s in geom_ord):
                # adjust the gradient using low-rank approximation
                self.post_Ga = GaussianLRPosterior(getattr(self,{True:'wht',False:''}[whitened]+'prior'), eigs[0], eigs[1])
                Hu = self.generate_vector(PARAMETER)
                self.post_Ga.Hlr.mult(parameter,Hu) # post_Ga.Hlr=posterior precision
                agrad.axpy(1.,Hu)
                Ru = self.generate_vector(PARAMETER)
                getattr(self,{True:'wht',False:''}[whitened]+'prior').grad(parameter,Ru)
                agrad.axpy(-1.,Ru)
        
        return loglik,agrad,HessApply,eigs
    
    def get_eigs(self,parameter=None,whitened=False,**kwargs):
        """
        Get the eigen-decomposition of Hessian action directly using randomized algorithm.
        """
        if parameter is None:
            parameter=self.prior.mean
        
        # un-whiten if necessary
        if whitened:
            parameter=self.whtprior.v2u(parameter)
        
        # solve the forward problem
        self.x[PARAMETER] = parameter
        self.pde.solveFwd(self.x[STATE], self.x)
        # solve the adjoint problem
        self.pde.solveAdj(self.x[ADJOINT], self.x, self.misfit)
        # get Hessian Apply
        HessApply = self._get_HessApply(parameter,kwargs.pop('MF_only',True)) # Hmisfit if MF is true
        if whitened:
            HessApply = wht_Hessian(self.whtprior,HessApply)
        # get estimated eigen-decomposition for the Hessian (or Gauss-Newton)
        k=kwargs['k'] if 'k' in kwargs else kwargs['incr_k'] if 'incr_k' in kwargs  else 80
        p=kwargs['p'] if 'p' in kwargs else 20
        if len(kwargs)==0:
            kwargs['k']=k
#         if self.rank == 0:
#             print('Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.'.format(k,p))
        Omega = MultiVector(parameter, k+p)
        parRandom.normal(1., Omega)
        if whitened:
            eigs = singlePassGx(HessApply, self.prior.M, self.prior.Msolver, Omega, **kwargs)
        else:
            eigs = doublePassG(HessApply, self.prior.R, self.prior.Rsolver, Omega, **kwargs)
        
        return eigs
    
    def get_MAP(self,rand_init=False,preconditioner='posterior',SAVE=True):
        """
        Get the maximum a posterior (MAP).
        """
        import time
        if self.rank == 0:
            sep = "\n"+"#"*80+"\n"
            print( sep, "Find the MAP point", sep)
        # setup solver
#         m0 = self.prior.mean.copy()
#         if rand_init:
#             noise = dl.Vector()
#             self.prior.init_vector(noise,"noise")
#             Random.normal(noise, 1., True)
#             self.prior.sample(noise,m0)
        m0 = self.prior.sample(add_mean=True) if rand_init else self.prior.mean.copy()
        solver = CGSolverSteihaug()
        H = self._get_HessApply(m0,MF_only=False)
        solver.set_operator(H)
        if preconditioner=='posterior':
            eigs = self.get_eigs(m0)
            self.post_Ga = GaussianLRPosterior(self.prior, eigs[0], eigs[1])
            P = self.post_Ga.Hlr
        elif preconditioner=='prior':
            P = self.prior.Rsolver
        solver.set_preconditioner( P )
        solver.parameters["rel_tolerance"] = 1e-6
        solver.parameters["abs_tolerance"] = 1e-9
        solver.parameters["max_iter"]      = 25
        if self.rank == 0:
            solver.parameters["print_level"] = 0 
        else:
            solver.parameters["print_level"] = -1
        mg0 = self._get_grad(m0)
        # solve for MAP
        start = time.time()
        solver.solve(m0, -mg0)
        end = time.time()
        self.x[PARAMETER] = m0
        self.pde.solveFwd(self.x[STATE],self.x)
        MAP = self.x
        if self.rank == 0:
            print('\nTime used is %.4f' % (end-start))
        
        if SAVE:
            fld_name='results'
            self._check_folder(fld_name)
            self.exportState(self.x, os.path.join(fld_name,"MAP.xdmf"), "MAP")
        
        if self.rank == 0:
            if solver.converged:
                print('\nConverged in ', solver.iter, ' iterations.')
            else:
                print('\nNot Converged')
            
            print('Termination reason: ', solver.reason[solver.reasonid])
            print('Final norm: ', solver.final_norm)
        
        return MAP[PARAMETER]
    
    def _check_folder(self,fld_name='result'):
        """
        Check the existence of folder for storing result and create one if not
        """
        import os
        if not hasattr(self, 'savepath'):
            cwd=os.getcwd()
            self.savepath=os.path.join(cwd,fld_name)
        if not os.path.exists(self.savepath):
            print('Save path does not exist; created one.')
            os.makedirs(self.savepath)
    
    def test(self,h=1e-4):
        """
        Demo to check results with the adjoint method against the finite difference method.
        """
        # random sample parameter
        parameter = self.prior.sample(add_mean=True)
#         true_init = dl.Expression('min(0.5,exp(-100*(pow(x[0]-0.35,2) +  pow(x[1]-0.7,2))))', element=self.pde.Vh[STATE].ufl_element())
#         parameter = dl.interpolate(true_init, self.pde.Vh[STATE]).vector()
        
        MF_only = True
        import time
        # obtain the geometric quantities
        if self.rank == 0:
            print('\n\nObtaining geometric quantities with Adjoint method...')
        start = time.time()
        loglik,grad,_,_ = self.get_geom(parameter,geom_ord=[0,1],whitened=False)
        HessApply = self._get_HessApply(parameter,MF_only=MF_only)
        end = time.time()
        if self.rank == 0:
            print('Time used is %.4f' % (end-start))
        
        # check with finite difference
        if self.rank == 0:
            print('\n\nTesting against Finite Difference method...')
        start = time.time()
        # random direction
        v = self.prior.sample()
        ## gradient
        if self.rank == 0:
            print('\nChecking gradient:')
        parameter_p = parameter.copy()
        parameter_p.axpy(h,v)
        loglik_p = -self._get_misfit(parameter_p)
#         parameter_m = parameter.copy()
#         parameter_m.axpy(-h,v)
#         loglik_m = -self._get_misfit(parameter_m)
        dloglikv_fd = (loglik_p-loglik)/h
        dloglikv = grad.inner(v)
#         rdiff_gradv = np.abs(dloglikv_fd-dloglikv)/np.linalg.norm(v)
        rdiff_gradv = np.abs(dloglikv_fd-dloglikv)/v.norm('l2')
        if self.rank == 0:
            print('Relative difference of gradients in a random direction between adjoint and finite difference: %.10f' % rdiff_gradv)

        # random direction
        w = self.prior.sample()
        ## metric-action
        if self.rank == 0:
            print('\nChecking Metric-action:')
        parameter_p = parameter.copy()
        parameter_p.axpy(h,w)
        self.x[PARAMETER] = parameter_p
        self.pde.solveFwd(self.x[STATE], self.x)
        gradv_p = -self._get_grad(parameter_p,MF_only=MF_only).inner(v)
#         parameter_m = parameter.copy()
#         parameter_m.axpy(-h,w)
#         self.x[PARAMETER] = parameter_m
#         self.pde.solveFwd(self.x[STATE], self.x)
#         gradv_m = -self._get_grad(parameter_m,MF_only=MF_only).inner(v)
        gradv = grad.inner(v)
        dgradvw_fd = (gradv_p-gradv)/h
        dgradvw = -HessApply.inner(v,w)
#             rdiff_Hessvw = np.abs(dgradvw_fd-dgradvw)/np.linalg.norm(v)/np.linalg.norm(w)
        rdiff_Hessvw = np.abs(dgradvw_fd-dgradvw)/v.norm('l2')/w.norm('l2')
        end = time.time()
        if self.rank == 0:
            print('Relative difference of Hessians in two random directions between adjoint and finite difference: %.10f' % rdiff_Hessvw)
            print('Time used is %.4f' % (end-start))
    
if __name__ == '__main__':
    # set up random seed
    seed=2020
    np.random.seed(seed)
    # define Bayesian inverse problem
    mesh = dl.Mesh('ad_10k.xml')
    Re = 100.
    rel_noise = .1
    nref = 1
    adif = advdiff(mesh=mesh, Re=Re, rel_noise=rel_noise, nref=nref, seed=seed)
    # test
    adif.test(1e-8)
    # obtain MAP
    map_v = adif.get_MAP(rand_init=True)
    fig=dl.plot(vector2Function(map_v,adif.pde.Vh[PARAMETER]))
    plt.colorbar(fig)
#     plt.show()
    plt.savefig(os.path.join(os.getcwd(),'results/map.png'),bbox_inches='tight')