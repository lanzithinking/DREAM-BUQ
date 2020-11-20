'''
Misfit of advection-diffusion problem
Created on Sep 23, 2020

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
from pde import TimeDependentAD

sys.path.append( "../" )
from util.common_colorbar import common_colorbar

class SpaceTimePointwiseStateObservation(Misfit):
    def __init__(self, Vh, observation_times=None, targets=None, rel_noise = 0.01, d = None, **kwargs):
        # function space
        self.Vh = Vh
        self.mpi_comm = self.Vh.mesh().mpi_comm()
        self.rank = dl.MPI.rank(self.mpi_comm)
        # observation times
        if observation_times is None:
            t_init         = 0.
            t_final        = 4.
            t_1            = 1.
            dt             = .1
            observation_dt = .2
            self.observation_times = np.arange(t_1, t_final+.5*dt, observation_dt)
        else:
            self.observation_times = observation_times
        # observation locations
        self.targets = np.loadtxt('targets.txt') if targets is None else targets
        self.rel_noise = rel_noise
        # obtain observations
        if d is None:
            d=self.get_observations(pde=kwargs.pop('pde',None), nref=kwargs.pop('nref',0), init=kwargs.pop('pde',None))
            if self.rank == 0:
                sep = "\n"+"#"*80+"\n"
                print( sep, "Generate synthetic observations at {0} locations for {1} time points".format(self.targets.shape[0], len(self.observation_times)), sep )
        # reset observation container for reference
        self.prep_container()
        self.d.axpy(1., d)
        
    def prep_container(self, Vh=None):
        if Vh is None:
            Vh = self.Vh
        # storage for observations
        self.B = assemblePointwiseObservation(Vh, self.targets)
        self.d = TimeDependentVector(self.observation_times)
        self.d.initialize(self.B, 0)
        ## TEMP Vars
        self.u_snapshot = dl.Vector()
        self.Bu_snapshot = dl.Vector()
        self.d_snapshot  = dl.Vector()
        self.B.init_vector(self.u_snapshot, 1)
        self.B.init_vector(self.Bu_snapshot, 0)
        self.B.init_vector(self.d_snapshot, 0)
    
    def get_observations(self, pde=None, nref=0, init=None):
        # pde for observations
        if pde is None:
            mesh = self.Vh.mesh()
            for i in range(nref): mesh = dl.refine(mesh)
            pde = TimeDependentAD(mesh)
        elif nref>0:
            mesh = pde.mesh
            for i in range(nref): mesh = dl.refine(mesh)
            pde = TimeDependentAD(mesh)
        # initial condition
        if init is None:
            true_init = dl.Expression('min(0.5,exp(-100*(pow(x[0]-0.35,2) +  pow(x[1]-0.7,2))))', element=pde.Vh[STATE].ufl_element())
            init = dl.interpolate(true_init, pde.Vh[STATE]).vector()
        # prepare container for observations
        self.prep_container(pde.Vh[STATE])
        
        utrue = pde.generate_vector(STATE)
        x = [utrue, init, None]
        pde.solveFwd(x[STATE], x)
        self.observe(x, self.d)
        MAX = self.d.norm("linf", "linf")
        noise_std_dev = self.rel_noise * MAX
        parRandom.normal_perturb(noise_std_dev,self.d)
        self.noise_variance = noise_std_dev*noise_std_dev
        return self.d.copy()
    
    def observe(self, x, obs):
        obs.zero()
        
        for t in self.observation_times:
            x[STATE].retrieve(self.u_snapshot, t)
            self.B.mult(self.u_snapshot, self.Bu_snapshot)
            obs.store(self.Bu_snapshot, t)
            
    def cost(self, x):
        c = 0
        for t in self.observation_times:
            x[STATE].retrieve(self.u_snapshot, t)
            self.B.mult(self.u_snapshot, self.Bu_snapshot)
            self.d.retrieve(self.d_snapshot, t)
            self.Bu_snapshot.axpy(-1., self.d_snapshot)
            c += self.Bu_snapshot.inner(self.Bu_snapshot)
            
        return c/(2.*self.noise_variance)
    
    def grad(self, i, x, out):
        out.zero()
        if i == STATE:
            for t in self.observation_times:
                x[STATE].retrieve(self.u_snapshot, t)
                self.B.mult(self.u_snapshot, self.Bu_snapshot)
                self.d.retrieve(self.d_snapshot, t)
                self.Bu_snapshot.axpy(-1., self.d_snapshot)
                self.Bu_snapshot *= 1./self.noise_variance
                self.B.transpmult(self.Bu_snapshot, self.u_snapshot) 
                out.store(self.u_snapshot, t)           
        else:
            pass
    
    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        pass
    
    def apply_ij(self, i,j, direction, out):
        out.zero()
        if i == STATE and j == STATE:
            for t in self.observation_times:
                direction.retrieve(self.u_snapshot, t)
                self.B.mult(self.u_snapshot, self.Bu_snapshot)
                self.Bu_snapshot *= 1./self.noise_variance
                self.B.transpmult(self.Bu_snapshot, self.u_snapshot) 
                out.store(self.u_snapshot, t)
        else:
            pass    
    
    def applyWuu(self, du, out):
        out.zero()
        self.apply_ij(STATE, STATE, du, out)
    
    def applyWum(self, dm, out):
        out.zero()
    
    def applyWmu(self, du, out):
        out.zero()
    
    def applyWmm(self, dm, out):
        out.zero()
    
    def plot_data(self, times, figsz=(12,5)):
        n=len(times)
        nrow=np.floor(np.sqrt(n)).astype('int')
        ncol=np.ceil(np.sqrt(n)).astype('int')
        fig,axes=plt.subplots(nrows=nrow,ncols=ncol,sharex=True,sharey=True,figsize=figsz)
        sub_figs = [None]*len(axes.flat)
        for i in range(n):
            plt.axes(axes.flat[i])
            dl.plot(self.Vh.mesh())
            sub_figs[i]=plt.scatter(self.targets[:,0],self.targets[:,1], c=self.d.data[np.where(np.isclose(self.d.times,times[i]))[0][0]])
            plt.title('Time: {:.1f} s'.format(times[i],))
        fig=common_colorbar(fig,axes,sub_figs)
        return fig
    
if __name__ == '__main__':
    np.random.seed(2020)
#     # define pde
#     pde = TimeDependentAD()
    # obtain function space
    mesh = dl.Mesh('ad_10k.xml')
    Vh = dl.FunctionSpace(mesh, "Lagrange", 2)
    # set observation times
    t_init         = 0.
    t_final        = 4.
    t_1            = 1.
    dt             = .1
    observation_dt = .2
    observation_times = np.arange(t_1, t_final+.5*dt, observation_dt)
    # set observation locations
    targets = np.loadtxt('targets.txt')
    # define misfit
    nref = 1
    misfit = SpaceTimePointwiseStateObservation(Vh, observation_times, targets, nref=nref)
#     # optional: refine mesh to obtain (new) observations
#     rf_mesh = dl.refine(pde.mesh)
#     rf_pde = TimeDependentAD(mesh=rf_mesh)
#     rf_obs = SpaceTimePointwiseStateObservation(rf_pde.Vh[STATE], observation_times, targets, pde=rf_pde).d.copy()
#     misfit.d.zero()
#     misfit.d.axpy(1.,rf_obs)
    # plot observations
    plt_times=[1.,2.,3.,4.]
    fig = misfit.plot_data(plt_times, (10,9))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(os.path.join(os.getcwd(),'result/obs.png'),bbox_inches='tight')
    