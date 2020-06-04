#!/usr/bin/env python
"""
(Regularizing) Ensemble Kalman Methods
Shiwei Lan @CalTech, 2017; @ASU, 2020
------------------------------------------------------------------
EKI: Algorithm 1 of 'Ensemble Kalman methods for inverse problems'
by Marco A Iglesias, Kody J H Law and Andrew M Stuart, Inverse Problems, Volume 29, Number 4, 2013
(Algorithm 1 of 'A regularizing iterative ensemble Kalman method for PDE-constrained inverse problems'
by Marco A Iglesias, Inverse Problems, Volume 32, Number 2, 2016)
EKS: Algorithm of 'Interacting Langevin Diffusions: Gradient Structure And Ensemble Kalman Sampler'
by Alfredo Garbuno-Inigo, Franca Hoffmann, Wuchen Li, and Andrew M. Stuart, SIAM Journal on Applied Dynamical Systems, Volume 19, Issue 1, February 4, 2020
----------------------------
Created November 26, 2017
----------------------------
Modified April 22, 2020 @ASU
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2017"
__license__ = "GPL"
__version__ = "0.5"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@caltech.edu; lanzithinking@gmail.com; slan@asu.edu"

import numpy as np
import timeit,time

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
        self.J,self.D=self.u.shape # number of ensembles (J) and state dimension (D)
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
        p=self.G(self.u) # (J,m) where m is the data (observation) dimension
        p_m=np.mean(p,axis=0,keepdims=True)
        
        # discrepancy principle
        eta=np.random.multivariate_normal(np.zeros(self.data['size']),self.r*self.data['cov']) if self.alg=='EKI' and self.r>=0 else 0
        y_eta=self.data['obs']+eta; # perturb data by noise eta
        err=np.sqrt((y_eta-p_m).dot(np.linalg.solve(self.data['cov'],(y_eta-p_m).T)))
        
        # analysis step
        p_tld=p-p_m
        C_pp=p_tld.T.dot(p_tld)/(self.J-1) # (m,m)
        C_up=self.u.T.dot(p_tld)/(self.J-1) # (D,m)
        alpha={'EKI':1./self.h,'EKS':self.h}[self.alg]
        while self.reg and self.alg=='EKI':
            alpha*=2
            err_alpha=np.linalg.solve(C_pp+alpha*self.data['cov'],(y_eta-p_m).T)
            if alpha*np.sqrt(err_alpha.T.dot(self.data['cov'].dot(err_alpha)))>=self.rho*err: break
        
#         d=np.linalg.solve((self.alg=='EKI')*C_pp+alpha*self.data['cov'],(y_eta-p).T) # (m,J)
        d=np.linalg.solve(C_pp+alpha*self.data['cov'],(y_eta-p).T) # (m,J)
#         u_=self.u+(C_up.dot(d)).T
        
        if self.alg=='EKI':
#             self.u=u_
            self.u+=(C_up.dot(d)).T
        elif self.alg=='EKS':
            C_uu=np.cov(self.u,rowvar=False)
            if self.adpt: alpha/=np.sqrt(np.sum(d*C_pp.dot(d))*(self.J-1))*alpha+self.eps
            print(alpha)
#             self.u=np.linalg.solve(np.eye(self.D)+alpha*C_uu.dot(np.linalg.inv(self.prior['cov'])),u_.T).T
            self.u=np.linalg.solve(np.eye(self.D)+alpha*C_uu.dot(np.linalg.inv(self.prior['cov'])),self.u.T+C_up.dot(d/alpha*self.h)).T
            self.u+=np.random.multivariate_normal(np.zeros(self.D),2*alpha*C_uu,self.J)
        
        return err,p
    
    # run EnK
    def run(self,max_iter=100,SAVE=False):
        '''
        Run ensemble Kalman methods to collect ensembles estimates/samples
        '''
        print('\nRunning '+self.alg+' now...\n')
        if self.h is None: self.h=1./max_iter
        errs=np.zeros(max_iter)
        fwdouts=np.zeros((max_iter,self.J,self.data['size']))
        ensbls=np.zeros((max_iter+1,self.J,self.D))
        ensbls[0]=self.u # record the initial ensemble
        u_est=np.zeros((max_iter,self.D))
        # start the timer
        tic=timeit.default_timer()
        r=self.r if self.r>0 else 1
        for n in range(max_iter):
            # update the Kalman filter
            errs[n],fwdouts[n]=self.update()
            # record the ensemble
            ensbls[n+1]=self.u
            # estimate unknown parameters
            u_est[n]=np.mean(self.u,axis=0)
#             p_n=self.G(u_est[n]); err_n=np.sqrt((self.data['obs']-p_n).dot(np.linalg.solve(self.data['cov'],(self.data['obs']-p_n).T))) # compute post error
            print('Estimated unknown parameters: '+(min(self.D,10)*"%.4f ") % tuple(u_est[n,:min(self.D,10)]) )
            print(self.alg+' at iteration %d, with error %.8f.\n' % (n+1,errs[n]) )
#             print(self.alg+' at iteration %d, with error %.8f.\n' % (n+1,err_n) )
            # terminate if discrepancy principle satisfied
            if errs[n]<=self.tau*r: break
        # stop timer
        toc=timeit.default_timer()
        t_used=toc-tic
        print('EnK terminates at iteration %d, with error %.4f, using time %.4f.' % (n+1,errs[n],t_used) )
        
        return_list=u_est,errs,fwdouts,ensbls,n,t_used
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
    # generate data
    D=4; m=10
    u_truth=np.arange(D)-1
    u_truth[-1]=1
    G=lambda u: np.sum(u.T[::2],axis=0)+np.sum(u.T[1::2]**2,axis=0) # BBD
#     A=np.random.normal(size=(m,D)) # Linear
#     G=lambda u: A.dot(u.T)
    Gamma=np.diag(np.random.rand(m))
    y=G(u_truth)+np.random.multivariate_normal(np.zeros(m),Gamma)
    data={'obs':y,'size':y.size,'cov':Gamma}
    # define prior
    sigma2_u=4
    pri_m=np.zeros(D); pri_cov=sigma2_u*np.eye(D)
    pri_samp=lambda n=1: np.random.multivariate_normal(pri_m,pri_cov,n)
    prior={'mean':pri_m,'cov':pri_cov,'sample':pri_samp}
    # initial ensemble
    J=100
    u=prior['sample'](J)
    # parameters
    stp_sz=[1,.01]
    nz_lvl=1
    err_thld=1e-1
    
    #### EKI ####
    # define ensemble Kalman method
    fwd_map=lambda u:np.tile(G(u)[:,None],m) # BBD
#     fwd_map=lambda u:G(u).T # Linear
    alg='EKI'
    eki=EnK(u,fwd_map,data,prior,stp_sz=stp_sz[0],nz_lvl=nz_lvl,err_thld=err_thld,alg=alg,reg=True)
    # run ensemble Kalman algorithm
    max_iter=100
    res_eki=eki.run(max_iter=max_iter)
#     eki.save(res_eki)
    
    #### EKS ####
    alg='EKS'
    eks=EnK(u,fwd_map,data,prior,stp_sz=stp_sz[1],nz_lvl=nz_lvl,err_thld=err_thld,alg=alg,adpt=True)
    # run ensemble Kalman algorithm
#     max_iter=100
    res_eks=eks.run(max_iter=max_iter)
#     eks.save(res_eks)
    
    # plot ensembles
    import matplotlib.pyplot as plt
    plt_dim=[0,1]
    niter=min(res_eki[-2],res_eks[-2])+1
    xmin=min(res_eki[3][:niter,:,plt_dim[0]].min(),res_eks[3][:niter,:,plt_dim[0]].min())
    xmax=max(res_eki[3][:niter,:,plt_dim[0]].max(),res_eks[3][:niter,:,plt_dim[0]].max())
    ymin=min(res_eki[3][:niter,:,plt_dim[1]].min(),res_eks[3][:niter,:,plt_dim[1]].min())
    ymax=max(res_eki[3][:niter,:,plt_dim[1]].max(),res_eks[3][:niter,:,plt_dim[1]].max())
    for n in range(niter):
        plt.subplot(121)
        plt.scatter(res_eki[3][n,:,plt_dim[0]],res_eki[3][n,:,plt_dim[1]],c='r',alpha=np.linspace(.2,.8,num=niter)[n])
        plt.subplot(122)
        plt.scatter(res_eks[3][n,:,plt_dim[0]],res_eks[3][n,:,plt_dim[1]],c='b',alpha=np.linspace(.2,.8,num=niter)[n])
    plt.subplot(121)
    plt.xlabel(r'$u_{0}$'.format(plt_dim[0]+1)); plt.ylabel(r'$u_{0}$'.format(plt_dim[1]+1))
    plt.xlim(xmin,xmax); plt.ylim(ymin,ymax)
    plt.title('EKI')
    plt.subplot(122)
    plt.xlabel(r'$u_{0}$'.format(plt_dim[0]+1)); plt.ylabel(r'$u_{0}$'.format(plt_dim[1]+1))
    plt.xlim(xmin,xmax); plt.ylim(ymin,ymax)
    plt.title('EKS')
    plt.subplots_adjust(wspace=.3)
    # save plots
#     plt.savefig('./result/ensembles.png',bbox_inches='tight')
    plt.show()