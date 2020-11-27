'''
PDE of advection-diffusion problem
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

sys.path.append( "../" )
from util.common_colorbar import common_colorbar

class codomain(dl.SubDomain):
    def __init__(self, offset = 0, *args, **kwargs):
        self.offset = offset
        dl.SubDomain.__init__(self, *args, **kwargs)
    def inside(self, x, on_boundary):
        return self.sub1(x) or self.sub2(x)
    def sub1(self, x):
        return x[0]>=0.25+self.offset and x[0]<=0.5-self.offset and x[1]>=0.15+self.offset and x[1]<=0.4-self.offset
    def sub2(self, x):
        return x[0]>=0.6+self.offset and x[0]<=0.75-self.offset and x[1]>=0.6+self.offset and x[1]<=0.85-self.offset

class TimeDependentAD:
    def __init__(self, mesh=None, simulation_times=None, gls_stab=True, **kwargs):
        # get mesh
        if mesh is None:
            self.mesh = dl.Mesh('ad_10k.xml')
        elif isinstance(mesh, tuple):
            self.meshsz = mesh
            self.mesh = self.generate_mesh(*self.meshsz)
        else:
            self.mesh = mesh
        self.mpi_comm = self.mesh.mpi_comm()
        self.rank = dl.MPI.rank(self.mpi_comm)
        # set FEM
        self.set_FEM()
        # get simulation times
        if simulation_times is None:
            t_init         = 0.
            t_final        = 4.
            dt             = .1
            self.simulation_times = np.arange(t_init, t_final+.5*dt, dt)
        else:
            self.simulation_times = simulation_times
        # get the diffusion coefficient
        self.kappa = kwargs.pop('kappa',.001)
        # get the Reynolds number
        self.Re = kwargs.pop('Re',1e2)
        # set weak forms
        self.gls_stab = gls_stab
        self.set_forms()
        
        # count PDE solving times
        self.soln_count = np.zeros(4)
        # 0-3: number of solving (forward,adjoint,2ndforward,2ndadjoint) equations respectively
    
    def generate_mesh(self, nx=52, ny=52):
        sqmesh = dl.UnitSquareMesh(nx=nx-1, ny=ny-1)
        submf = dl.MeshFunction('size_t', sqmesh, sqmesh.topology().dim())
        submf.set_all(1)
        codomain().mark(submf,0)
        mesh = dl.SubMesh(sqmesh, submf, 1)
        return mesh
    
    def set_FEM(self):
        """
        Define finite element space of advection-diffusion PDE.
        """
        Vh_STATE = dl.FunctionSpace(self.mesh, "Lagrange", 2)
        ndofs = Vh_STATE.dim()
        if self.rank == 0: print( "Number of dofs: {0}".format( ndofs ) )
        self.Vh = [Vh_STATE,Vh_STATE,Vh_STATE]
        
    def computeVelocityField(self):
        Xh = dl.VectorFunctionSpace(self.mesh,'Lagrange', 2)
        Wh = dl.FunctionSpace(self.mesh, 'Lagrange', 1)
        
        mixed_element = dl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
        XW = dl.FunctionSpace(self.mesh, mixed_element)
        
        Re = dl.Constant(self.Re)
        
        def v_boundary(x,on_boundary):
            return on_boundary
        
        def q_boundary(x,on_boundary):
            return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS
        
        g = dl.Expression(('0.0','(x[0] < 1e-14) - (x[0] > 1 - 1e-14)'), element=Xh.ufl_element())
        bc1 = dl.DirichletBC(XW.sub(0), g, v_boundary)
        bc2 = dl.DirichletBC(XW.sub(1), dl.Constant(0), q_boundary, 'pointwise')
        bcs = [bc1, bc2]
        
        vq = dl.Function(XW)
        (v,q) = ufl.split(vq)
        (v_test, q_test) = dl.TestFunctions (XW)
        
        def strain(v):
            return ufl.sym(ufl.grad(v))
        
        F = ( (2./Re)*ufl.inner(strain(v),strain(v_test))+ ufl.inner (ufl.nabla_grad(v)*v, v_test)
               - (q * ufl.div(v_test)) + ( ufl.div(v) * q_test) ) * ufl.dx
        
        dl.solve(F == 0, vq, bcs, solver_parameters={"newton_solver":
                                                    {"relative_tolerance":1e-4, "maximum_iterations":100,
                                                     "linear_solver":"default"}})
        
        return v
    
    def set_forms(self):
        # Assume constant timestepping
        dt = self.simulation_times[1] - self.simulation_times[0]
        
        self.wind_velocity = self.computeVelocityField()
        
        u = dl.TrialFunction(self.Vh[STATE])
        v = dl.TestFunction(self.Vh[STATE])
        
        kappa = dl.Constant(self.kappa)
        dt_expr = dl.Constant(dt)
        
        r_trial = u + dt_expr*( -ufl.div(kappa*ufl.grad(u))+ ufl.inner(self.wind_velocity, ufl.grad(u)) )
        r_test  = v + dt_expr*( -ufl.div(kappa*ufl.grad(v))+ ufl.inner(self.wind_velocity, ufl.grad(v)) )
        
        h = dl.CellDiameter(self.mesh)
        vnorm = ufl.sqrt(ufl.inner(self.wind_velocity, self.wind_velocity))
        if self.gls_stab:
            tau = ufl.min_value((h*h)/(dl.Constant(2.)*kappa), h/vnorm )
        else:
            tau = dl.Constant(0.)
        
        self.M = dl.assemble( ufl.inner(u,v)*ufl.dx )
        self.M_stab = dl.assemble( ufl.inner(u, v+tau*r_test)*ufl.dx )
        self.Mt_stab = dl.assemble( ufl.inner(u+tau*r_trial,v)*ufl.dx )
        Nvarf  = (ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) + ufl.inner(self.wind_velocity, ufl.grad(u))*v )*ufl.dx
        Ntvarf  = (ufl.inner(kappa *ufl.grad(v), ufl.grad(u)) + ufl.inner(self.wind_velocity, ufl.grad(v))*u )*ufl.dx
        self.N  = dl.assemble( Nvarf )
        self.Nt = dl.assemble(Ntvarf)
        stab = dl.assemble( tau*ufl.inner(r_trial, r_test)*ufl.dx)
        self.L = self.M + dt*self.N + stab
        self.Lt = self.M + dt*self.Nt + stab
        
        self.solver  = PETScLUSolver( self.mpi_comm )
        self.solver.set_operator( dl.as_backend_type(self.L) )
        self.solvert = PETScLUSolver( self.mpi_comm ) 
        self.solvert.set_operator(dl.as_backend_type(self.Lt) )
    
    def generate_vector(self, component = STATE):
        if component == STATE:
            u = TimeDependentVector(self.simulation_times)
            u.initialize(self.M, 0)
            return u
        elif component == ADJOINT:
            p = TimeDependentVector(self.simulation_times)
            p.initialize(self.M, 0)
            return p
        else:
            raise
    
    def solveFwd(self, out, x):
        out.zero()
        uold = x[PARAMETER]
        out.store(uold,0) # store the initial condition
        u = dl.Vector()
        rhs = dl.Vector()
        self.M.init_vector(rhs, 0)
        self.M.init_vector(u, 0)
        for t in self.simulation_times[1::]:
            self.M_stab.mult(uold, rhs)
            self.solver.solve(u, rhs)
            out.store(u,t)
            uold = u
        self.soln_count[0] += 1
    
    def solveAdj(self, out, x, misfit):
        
        grad_state = TimeDependentVector(self.simulation_times)
        grad_state.initialize(self.M, 0)
        misfit.grad(STATE, x, grad_state)
        
        out.zero()
        
        pold = dl.Vector()
        self.M.init_vector(pold,0)
        
        p = dl.Vector()
        self.M.init_vector(p,0)
        
        rhs = dl.Vector()
        self.M.init_vector(rhs,0)
        
        grad_state_snap = dl.Vector()
        self.M.init_vector(grad_state_snap,0)
        
        for t in self.simulation_times[::-1]:
            self.Mt_stab.mult(pold,rhs)
            grad_state.retrieve(grad_state_snap, t)
            rhs.axpy(-1., grad_state_snap)
            self.solvert.solve(p, rhs)
            pold = p
            out.store(p, t)
        self.soln_count[1] += 1
    
    def solveFwdIncremental(self, sol, rhs):
        sol.zero()
        uold = dl.Vector()
        u = dl.Vector()
        Muold = dl.Vector()
        myrhs = dl.Vector()
        self.M.init_vector(uold, 0)
        self.M.init_vector(u, 0)
        self.M.init_vector(Muold, 0)
        self.M.init_vector(myrhs, 0)
        
        for t in self.simulation_times[1::]:
            self.M_stab.mult(uold, Muold)
            rhs.retrieve(myrhs, t)
            myrhs.axpy(1., Muold)
            self.solver.solve(u, myrhs)
            sol.store(u,t)
            uold = u
        self.soln_count[2] += 1
    
    def solveAdjIncremental(self, sol, rhs):
        sol.zero()
        pold = dl.Vector()
        p = dl.Vector()
        Mpold = dl.Vector()
        myrhs = dl.Vector()
        self.M.init_vector(pold, 0)
        self.M.init_vector(p, 0)
        self.M.init_vector(Mpold, 0)
        self.M.init_vector(myrhs, 0)
        
        for t in self.simulation_times[::-1]:
            self.Mt_stab.mult(pold,Mpold)
            rhs.retrieve(myrhs, t)
            Mpold.axpy(1., myrhs)
            self.solvert.solve(p, Mpold)
            pold = p
            sol.store(p, t)  
        self.soln_count[3] += 1
    
    def applyC(self, dm, out):
        out.zero()
        myout = dl.Vector()
        self.M.init_vector(myout, 0)
        self.M_stab.mult(dm,myout)
        myout *= -1.
        t = self.simulation_times[1]
        out.store(myout,t)
        
        myout.zero()
        for t in self.simulation_times[2:]:
            out.store(myout,t)
    
    def applyCt(self, dp, out):
        t = self.simulation_times[1]
        dp0 = dl.Vector()
        self.M.init_vector(dp0,0)
        dp.retrieve(dp0, t)
        dp0 *= -1.
        self.Mt_stab.mult(dp0, out)
    
    def plot_soln(self, x, times, figsz=(12,5)):
        n=len(times)
        nrow=np.floor(np.sqrt(n)).astype('int')
        ncol=np.ceil(np.sqrt(n)).astype('int')
        fig,axes=plt.subplots(nrows=nrow,ncols=ncol,sharex=True,sharey=True,figsize=figsz)
        sub_figs = [None]*len(axes.flat)
        for i in range(n):
            plt.axes(axes.flat[i])
            sub_figs[i]=dl.plot(vector2Function(x.data[list(x.times).index(times[i])],self.Vh[STATE]))
            plt.title('Time: {:.1f} s'.format(times[i],))
        fig=common_colorbar(fig,axes,sub_figs)
        return fig
    
if __name__ == '__main__':
    np.random.seed(2020)
    # get mesh
#     mesh = dl.Mesh('ad_10k.xml')
    mesh = (51,51)
#     from mshr import Rectangle, generate_mesh
#     domain = Rectangle(dl.Point(0,0),dl.Point(1,1)) - Rectangle(dl.Point(.25,.15),dl.Point(.5,.4)) - Rectangle(dl.Point(.6,.6),dl.Point(.75,.85))
#     mesh = generate_mesh(domain, 20)
#     dl.plot(mesh); plt.show()
    # set solution times
    t_init         = 0.
    t_final        = 4.
    t_1            = 1.
    dt             = .1
    observation_dt = .2
    simulation_times = np.arange(t_init, t_final+.5*dt, dt)
    # define PDE
    kappa = 1e-3
    ad_diff = TimeDependentAD(mesh, simulation_times, kappa=kappa)
    # plot mesh
#     dl.plot(mesh)
    # set parameters of PDE
    Vh = ad_diff.Vh[STATE]
    ic_expr = dl.Expression('min(0.5,exp(-100*(pow(x[0]-0.35,2) +  pow(x[1]-0.7,2))))', element=Vh.ufl_element())
    true_initial_condition = dl.interpolate(ic_expr, Vh).vector()
    utrue = ad_diff.generate_vector(STATE)
    x = [utrue, true_initial_condition, None]
    ad_diff.solveFwd(x[STATE], x)
    # plot solution
    plt_times=[0,0.4,1.,2.,3.,4.]
    fig = ad_diff.plot_soln(x[STATE], plt_times, (12,8))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(os.path.join(os.getcwd(),'results/solns.png'),bbox_inches='tight')