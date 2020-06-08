#!/usr/bin/env python
"""
(Regularizing) Ensemble Kalman Methods
Shiwei Lan @ASU, 2020
------------------------------------------------------------------
EKI: Algorithm 1 of 'Ensemble Kalman methods for inverse problems'
by Marco A Iglesias, Kody J H Law and Andrew M Stuart, Inverse Problems, Volume 29, Number 4, 2013
(Algorithm 1 of 'A regularizing iterative ensemble Kalman method for PDE-constrained inverse problems'
by Marco A Iglesias, Inverse Problems, Volume 32, Number 2, 2016)
EKS: Algorithm of 'Interacting Langevin Diffusions: Gradient Structure And Ensemble Kalman Sampler'
by Alfredo Garbuno-Inigo, Franca Hoffmann, Wuchen Li, and Andrew M. Stuart, SIAM Journal on Applied Dynamical Systems, Volume 19, Issue 1, February 4, 2020
-------------------------------
Created May 4, 2020
-------------------------------
This is specialized for FEniCS.
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2020"
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com"

import numpy as np
import timeit,time
import dolfin as df
import sys
sys.path.append( "../" )
from util.multivector import *

class _ImpLinOp(df.LinearOperator):
    def __init__(self,prior,u,alpha):
        self.prior=prior
        self.u=u
        self.alpha=alpha
        
        df.LinearOperator.__init__(self,self.gen_vector(),self.gen_vector())
        
    def gen_vector(self,v=None):
        return self.prior.gen_vector(v)
    
    def init_vector(self,x,dim):
        self.prior.init_vector(x,dim)
    
    def mult(self,x,y):
        y.zero()
        tmp=self.u.dot(self.prior.C_act(x,-1))
        tmp-=np.mean(tmp,axis=0)
        self.u.reduce(y,self.alpha/(tmp.shape[0]-1)*tmp)
        y.axpy(1.,x)
    
    def inner(self,x,y):
        Ox = df.Vector()
        self.init_vector(Ox,0)
        self.mult(x, Ox)
        return Ox.inner(self.M*y)
    
    def get_solver(self,**kwargs):
#         solver=df.PETScLUSolver(self.prior.mpi_comm,self._as_petscmat(),'mumps' if df.has_lu_solver_method('mumps') else 'default')
# #         solver.set_operator(self._as_petscmat())
# #         solver.parameters['reuse_factorization']=True
# #         solver.parameters['symmetric']=True
        
        preconditioner=kwargs.pop('preconditioner',None)
        if preconditioner:
            solver = df.PETScKrylovSolver("default", "default")
            solver.set_operators(self,preconditioner)
        else:
            solver = df.PETScKrylovSolver("cg", "none") # very slow without proper preconditioner
            solver.set_operator(self)
        solver.parameters["maximum_iterations"] = 1000
        solver.parameters["relative_tolerance"] = 1e-7
        solver.parameters["error_on_nonconvergence"] = True
        solver.parameters["nonzero_initial_guess"] = False
        
        return solver
    
    def _as_petscmat(self):
        if df.has_petsc4py():
            from petsc4py import PETSc
            mat = PETSc.Mat().createPython(df.as_backend_type(self.prior.M).mat().getSizes(), comm = self.prior.mpi_comm)
#             mat = PETSc.Mat().createPython(self.prior.dim, comm = self.prior.mpi_comm)
            mat.setPythonContext(self)
            return df.PETScMatrix(mat)
        else:
            df.warning('Petsc4py not installed: cannot generate PETScMatrix with specified size!')
            pass

class EnK(object):
    def __init__(self,u,G,data,prior=None,stp_sz=None,nz_lvl=1,alg='EKI',**kwargs):
        '''
        Ensemble Kalman Methods (EKI & EKS)
        -----------------------------------
        u: J ensemble states in D dimensional space
        G: forward mapping
        data: data including observations y and likelihood definition (covariance Gamma)
            - obs: observations
            - size: size of data
            - cov: covariance of data(noise)
        prior: prior including prior definition (covariance C)
            - mean: prior mean
            - cov: prior covariance
            - sample: generate prior sample
        stp_sz: step size for discretized Kalman dynamics
        nz_lvl: level of perturbed noise to data, 0 means no perturbation
        reg: indicator of whether to implement regularization in optimization (EKI)
        adpt: indicator of whether to implement adaptation in stepsize (EKS)
        alg: algorithm option for choice of 'EKI' or 'EKS'
        optional:
        err_thld: threshold factor for stopping criterion
        reg_thld: threshold factor for finding regularizing parameter
        adpt_par: parameter in time-step adaptation to avoid overfloating
        '''
        # ensemble states
        self.u=u
        self.D=self.u[0].size() # state dimension (D)
        self.J=self.u.nvec() # number of ensembles (J)
        # forward mapping
        self.G=G
        # data and prior
        self.data=data
        self.prior=prior
        
        # algorithm specific parameters
        self.h=stp_sz
        self.r=nz_lvl
        self.tau=kwargs.pop('err_thld',.1) # default to be .1
        self.reg=kwargs.pop('reg',False) # default to be false
        if self.reg:
            self.rho=kwargs.pop('reg_thld',0.5) # default to be 0.5
            self.tau=max(self.tau,1/self.rho) # tau >=1/rho
        self.adpt=kwargs.pop('adpt',True) # default to be true
        if self.adpt:
            self.eps=kwargs.pop('adpt_par',self.tau)
        self.alg=alg
    
    # update ensemble sates
    def update(self):
        '''
        One step update of ensemble Kalman methods
        '''
        # prediction step
        p=np.array([self.G(self.u[j]) for j in range(self.J)]) # (J,m) where m is the data (observation) dimension
        p_m=np.mean(p,axis=0,keepdims=True)
        
        # discrepancy principle
        eta=np.random.multivariate_normal(np.zeros(self.data['size']),self.r*self.data['cov']) if self.alg=='EKI' and self.r>=0 else 0
        y_eta=self.data['obs']+eta; # perturb data by noise eta
        err=np.sqrt((y_eta-p_m).dot(np.linalg.solve(self.data['cov'],(y_eta-p_m).T)))
        
        # analysis step
        p_tld=p-p_m # (J,m)
        C_pp=p_tld.T.dot(p_tld)/(self.J-1) # (m,m)
        C_up=MultiVector(self.u[0],self.data['size']) # C_up=0_{D x m}
        MvDSmatMult(self.u,p_tld/(self.J-1),C_up) # (D,m), C_up=u*p_tld/(J-1)
        alpha={'EKI':1./self.h,'EKS':self.h}[self.alg]
        while self.reg and self.alg=='EKI':
            alpha*=2
            err_alpha=np.linalg.solve(C_pp+alpha*self.data['cov'],(y_eta-p_m).T)
            if alpha*np.sqrt(err_alpha.T.dot(self.data['cov'].dot(err_alpha)))>=self.rho*err: break
        
#         d=np.linalg.solve((self.alg=='EKI')*C_pp+alpha*self.data['cov'],(y_eta-p).T) # (m,J)
        d=np.linalg.solve(C_pp+alpha*self.data['cov'],(y_eta-p).T) # (m,J), C_pp is present to stabilize the inverse
        
        if self.alg=='EKI':
            MvDSmatMult(C_up,d,self.u,True) # self.u+=C_up*d
        elif self.alg=='EKS':
            if self.adpt: alpha/=np.sqrt(np.sum(d*C_pp.dot(d))*(self.J-1))*alpha+self.eps
            print(alpha)
            
            u_=MultiVector(self.u) # copy u to u_
#             MvDSmatMult(C_up,d/alpha*self.h,self.u,True) # u+=C_up*d
            MvDSmatMult(C_up,d,self.u,True)
            implinop=_ImpLinOp(self.prior,u_,alpha)
            solver=implinop.get_solver()
            u_j=self.prior.gen_vector()
            implinop.init_vector(u_j,1)
            for j in range(self.J):
                solver.solve(u_j,self.u[j])
                self.u[j].set_local(u_j)
            
            noise=np.sqrt(2*alpha/(self.J-1))*np.random.normal(size=(self.J,)*2)
            noise-=np.mean(noise,axis=0)
            MvDSmatMult(u_,noise,self.u,True) # u+= u_*noise
        
        return err,p
    
    # run EnK
    def run(self,max_iter=100,SAVE=False):
        '''
        Run ensemble Kalman methods to collect ensembles estimates/samples
        '''
        print('\nRunning '+self.alg+' now...\n')
        if self.h is None: self.h=1./max_iter
        # allocate space to store results
        errs=np.zeros(max_iter)
        fwdouts=np.zeros((max_iter,self.J,self.data['size']))
        
        import os
        fpath=os.path.join(os.getcwd(),'result')
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
        ensbl_fname=self.alg+'_ensbl'+str(self.J)+'_dim'+str(self.D)+'_'+ctime
#         ensbls=df.File(os.path.join(fpath,ensbl_fname+".xdmf"))
        ensbls=df.HDF5File(self.prior.mpi_comm,os.path.join(fpath,ensbl_fname+".h5"),"w")
        # record the initial ensemble
        u_f=df.Function(self.prior.V)
        for j in range(self.J):
            u_f.vector().zero()
            u_f.vector().axpy(1.,self.u[j])
            ensbls.write(u_f,'iter{0}_ensbl{1}'.format(0,j))
        uest_fname=self.alg+'_uest'+'_dim'+str(self.D)+'_'+ctime
        u_est=df.HDF5File(self.prior.mpi_comm,os.path.join(fpath,uest_fname+".h5"),"w")
        # start the timer
        tic=timeit.default_timer()
        r=self.r if self.r>0 else 1
        for n in range(max_iter):
            # update the Kalman filter
            errs[n],fwdouts[n]=self.update()
            # record the ensemble
            for j in range(self.J):
                u_f.vector().zero()
                u_f.vector().axpy(1.,self.u[j])
                ensbls.write(u_f,'iter{0}_ensbl{1}'.format(n+1,j))
            # estimate unknown parameters
            u_f.vector().zero()
            self.u.reduce(u_f.vector(),np.ones(self.J)/self.J)
            u_est.write(u_f,'iter{0}'.format(n))
#             p_n=self.G(u_f.vector()); err_n=np.sqrt((self.data['obs']-p_n).dot(np.linalg.solve(self.data['cov'],(self.data['obs']-p_n).T))) # compute post error
            if self.D<=10:
                print('Estimated unknown parameters: '+(min(self.D,10)*"%.4f ") % tuple(u_f.vector()[:min(self.D,10)]) )
            else:
                print('Estimated unknown parameters: min %.4f, med %.4f, max %.4f ' % (u_f.vector().min(), np.median(u_f.vector()), u_f.vector().max()) )
            print(self.alg+' at iteration %d, with error %.8f.\n' % (n+1,errs[n]) )
#             print(self.alg+' at iteration %d, with error %.8f.\n' % (n+1,err_n) )
            # terminate if discrepancy principle satisfied
            if errs[n]<=self.tau*r: break
        # stop timer
        ensbls.close(); u_est.close()
        toc=timeit.default_timer()
        t_used=toc-tic
        print('EnK terminates at iteration %d, with error %.4f, using time %.4f.' % (n+1,errs[n],t_used) )
        
        return_list=errs,fwdouts,n,t_used
        if SAVE:
            return self.save(return_list)
        else:
            return return_list
    
    # save results to file
    def save(self,dump_list):
        import os,errno
        import pickle
        # create folder
        cwd=os.getcwd()
        savepath=os.path.join(cwd,'result')
        try:
            os.makedirs(savepath)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
            else:
                raise
        # name file
        ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
        filename=self.alg+'_ensbl'+str(self.J)+'_dim'+str(self.D)+'_'+ctime
        # dump data
        f=open(os.path.join(savepath,filename+'.pckl'),'wb')
        pickle.dump(dump_list,f)
        f.close()
        
        return savepath,filename
    
#         # load data
#         f=open(os.path.join(savepath,filename+'.pckl'),'rb')
#         loaded=pickle.load(f)
#         f.close()
    
# test
if __name__=='__main__':
    np.random.seed(2020)
    # import modules
    sys.path.append('../elliptic_inverse/')
    from elliptic_inverse.Elliptic import Elliptic
    
    ## define the inverse elliptic problem ##
    # parameters for PDE model
    nx=40;ny=40;
    # parameters for prior model
    sigma=1.25;s=0.0625
    # parameters for misfit model
    SNR=50 # 100
    # define the inverse problem
    elliptic=Elliptic(nx=nx,ny=ny,SNR=SNR,sigma=sigma,s=s)
    
    # initialization
    J=10
    unknown=MultiVector(elliptic.prior.gen_vector(),J)
    for j in range(J): unknown[j].set_local(elliptic.prior.sample(whiten=False))
    # define parameters needed
    def G(u,IP=elliptic):
        u_f=df.Function(IP.prior.V)
        u_f.vector().zero()
        u_f.vector().axpy(1.,u)
        IP.pde.set_forms(unknown=u_f)
        return IP.misfit._extr_soloc(IP.pde.soln_fwd()[0])
    
    y=elliptic.misfit.obs
    data={'obs':y,'size':y.size,'cov':1./elliptic.misfit.prec*np.eye(y.size)}
    
    # parameters
    stp_sz=[.1,.01]
    nz_lvl=1
    err_thld=1e-1
    algs=['EKI','EKS']
    num_algs=len(algs)
    max_iter=20
    
    #### EKI ####
    eki=EnK(unknown,G,data,elliptic.prior,stp_sz=stp_sz[0],nz_lvl=nz_lvl,err_thld=err_thld,alg=algs[0],reg=True)
    # run ensemble Kalman algorithm
    res_eki=eki.run(max_iter=max_iter)
    
    #### EKS ####
    eks=EnK(unknown,G,data,elliptic.prior,stp_sz=stp_sz[1],nz_lvl=nz_lvl,err_thld=err_thld,alg=algs[1],adpt=True)
    # run ensemble Kalman algorithm
#     max_iter=100
    res_eks=eks.run(max_iter=max_iter)
    
    import os
    folder = os.path.join(os.getcwd(),'result')
    fnames=[f for f in os.listdir(folder) if f.endswith('.h5')]
    # plot ensembles
    import matplotlib.pyplot as plt
    import matplotlib as mp
    from util import matplot4dolfin
    matplot=matplot4dolfin()
    
    num_rows=1
    fig,axes = plt.subplots(nrows=num_rows,ncols=np.int(np.ceil((1+num_algs)/num_rows)),sharex=True,sharey=True,figsize=(10,4))
    for i,ax in enumerate(axes.flat):
        plt.axes(ax)
        if i==0:
            # plot MAP
            try:
                f=df.HDF5File(elliptic.pde.mpi_comm, os.path.join('./result',"MAP_SNR"+str(SNR)+".h5"), "r")
                MAP=df.Function(elliptic.pde.V,name="parameter")
                f.read(MAP,"parameter")
                f.close()
                sub_fig=matplot.plot(MAP)
                ax.set_title('MAP')
            except:
                pass
        elif 1<=i<=num_algs:
            # plot ensemble estimate
            found=False
            u_est=df.Function(elliptic.pde.V)
            for f_i in fnames:
                if algs[i-1]+'_uest_' in f_i:
                    try:
                        f=df.HDF5File(elliptic.pde.mpi_comm,os.path.join(folder,f_i),"r")
                        n={'0':res_eki[2],'1':res_eks[2]}[i-1]
                        f.read(u_est,'iter{0}'.format(n))
                        f.close()
                        print(f_i+' has been read!')
                        found=True
                    except:
                        pass
            if found:
                sub_fig=matplot.plot(u_est)
#                 sub_fig=df.plot(u_est)
                ax.set_title(algs[i-1])
        ax.set_aspect('auto')
    plt.axis([0, 1, 0, 1])
    
    # set color bar
    cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(sub_fig, cax=cax, **kw)

    # save plots
#     fig.tight_layout()
#     plt.savefig(folder+'/ensemble_estimates.png',bbox_inches='tight')
    plt.show()