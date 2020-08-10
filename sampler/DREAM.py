#!/usr/bin/env python
"""
Dimension-Reduced Emulative AutoEncoder Monte-Carlo (DREAM) algorithms
Shiwei Lan @ ASU, 2020
--------------------------------------
Originally created Aug. 6, 2020 @ ASU
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2020, The NN-MCMC project"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
import timeit,time

# functions needed to make even image size
def chop(A):
    A_chopped=A.copy()
    if A.shape[0]%2 !=0:
        A_chopped=A_chopped[:-1,:]
    if A.shape[1]%2 !=0:
        A_chopped=A_chopped[:,:-1]
    return A_chopped

def vec2img(v):
    gdim=2
    imsz = np.floor(v.size**(1./gdim)).astype('int')
    im_shape=(-1,)+(imsz,)*(gdim-1)
    return v.reshape(im_shape)

# prep4dec = lambda u, ae_type: chop(vec2img(u))[None,:,:,None] if 'Conv' in ae_type else u[None,:] # chop image to have even dimensions
def logvol(unknown_lat,autoencoder,coding):
    if 'Conv' in type(autoencoder).__name__:
        u_latin=vec2img(unknown_lat)
        u_latin=chop(u_latin)[None,:,:,None] if autoencoder.activations['latent'] is None else u_latin.flatten()[None,:]
    else:
        u_latin=unknown_lat[None,:]
    return autoencoder.logvol({'encode':autoencoder.decode(u_latin),'decode':u_latin}[coding],coding)

class DREAM:
    """
    Dimension-Reduced Emulative AutoEncoder Monte-Carlo (DREAM) algorithms
    ----------------------------------------------------------------------
    After the class is instantiated with arguments, call sample to collect MCMC samples which will be stored in 'result' folder.
    """
    def __init__(self,parameter_init,model,latent_geom,step_size,step_num,alg_name,adpt_h=False,**kwargs):
        """
        Initialization
        """
        # parameters
        self.q=parameter_init
        self.dim=np.size(self.q)
        self.model=model
        
        target_acpt=kwargs.pop('target_acpt',0.65)
        self.vol_wts=kwargs.pop('vol_wts','adjust')
        if self.vol_wts:
            self.AE=kwargs.pop('AE',None)
            if self.AE is None:
                print('Warning: No proper AutoEncoder found for volume adjustment! No volume weights will be logged.')
                self.vol_wts=False
        self.whitened=kwargs.pop('whitened',False)
        
        # geometry needed
        geom_ord=[0]
        if any(s in alg_name for s in ['MALA','HMC']): geom_ord.append(1)
        if any(s in alg_name for s in ['mMALA','mHMC']): geom_ord.append(2)
        self.geom=lambda parameter: latent_geom(parameter,geom_ord=geom_ord,whitened=self.whitened,**kwargs)
        self.ll,self.g,_,self.eigs=self.geom(self.q)
#         self.ll,self.g,_,self.eigs=self.model.get_geom(self.q,geom_ord,**kwargs)
        
        # sampling setting
        self.h=step_size
        self.L=step_num
        if 'HMC' not in alg_name: self.L=1
        self.alg_name = alg_name
        
        # optional setting for adapting step size
        self.adpt_h=adpt_h
        if self.adpt_h:
            h_adpt={}
#             h_adpt['h']=self._init_h()
            h_adpt['h']=self.h
            h_adpt['mu']=np.log(10*h_adpt['h'])
            h_adpt['loghn']=0.
            h_adpt['An']=0.
            h_adpt['gamma']=0.05
            h_adpt['n0']=10
            h_adpt['kappa']=0.75
            h_adpt['a0']=target_acpt
            self.h_adpt=h_adpt
    
    def randv(self,post_Ga=None):
        """
        sample v ~ N(0,C) or N(0,invK(q))
        """
        if post_Ga is None:
            v = np.random.randn(self.model.input_dim) if self.whitened else self.model.prior['sample']()
        else:
            v = self.model.post_Ga['sample']()
        return v
    
    def DREAMpCN(self):
        """
        preconditioned Crank-Nicolson under DREAM framework
        """
        logwt=0
        # initialization
        q=self.q.copy()
        
        # sample velocity
        v=self.randv()
        
        # correct volume if requested
        if self.vol_wts: logwt+=logvol(q,self.AE,'encode')
        
        # generate proposal according to Crank-Nicolson scheme
        q = ((1-self.h/4)*self.q + np.sqrt(self.h)*v)/(1+self.h/4)
        
        # update geometry
        ll,_,_,_=self.geom(q)
        
        # correct volume if requested
        if self.vol_wts: logwt+=logvol(q,self.AE,'decode') # TODO: need go back to the original scale too?
        
        # Metropolis test
        logr=ll-self.ll+(logwt if self.vol_wts=='adjust' else 0)
        
        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.ll=ll;
            acpt=True
        else:
            if self.vol_wts=='record': logwt=0
            acpt=False
        
        # return accept indicator
        return acpt,logr,logwt
    
    def DREAMinfMALA(self):
        """
        infinite dimensional Metropolis Adjusted Langevin Algorithm under DREAM framework
        """
        logwt=0
        # initialization
        q=self.q.copy()
        rth=np.sqrt(self.h)
        
        # sample velocity
        v=self.randv()
        
        # natural gradient
        ng=self.g if self.whitened else self.model.prior['cov'].dot(self.g)
        
        # update velocity
        v+=rth/2*ng
        
        # current energy
        E_cur = -self.ll - rth/2*self.g.dot(v) + self.h/8*self.g.dot(ng)
        
        # correct volume if requested
        if self.vol_wts: logwt+=logvol(q,self.AE,'encode')
        
        # generate proposal according to Langevin dynamics
        q = ((1-self.h/4)*self.q + rth*v)/(1+self.h/4)
        
        # update velocity
        v = (-(1-self.h/4)*v + rth*self.q)/(1+self.h/4)
        
        # update geometry
        ll,g,_,_=self.geom(q)
        
        # natural gradient
        ng=g if self.whitened else self.model.prior['cov'].dot(g)
        
        # new energy
        E_prp = -ll - rth/2*g.dot(v) + self.h/8*g.dot(ng)
        
        # correct volume if requested
        if self.vol_wts: logwt+=logvol(q,self.AE,'decode')
        
        # Metropolis test
        logr=-E_prp+E_cur+(logwt if self.vol_wts=='adjust' else 0)
        
        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.ll=ll; self.g=g;
            acpt=True
        else:
            if self.vol_wts=='record': logwt=0
            acpt=False
        
        # return accept indicator
        return acpt,logr,logwt
    
    def DREAMinfHMC(self):
        """
        infinite dimensional Hamiltonian Monte Carlo under DREAM framework
        """
        logwt=0
        # initialization
        q=self.q.copy()
        rth=np.sqrt(self.h) # make the scale comparable to MALA
        cos_=np.cos(rth); sin_=np.sin(rth);

        # sample velocity
        v=self.randv()

        # natural gradient
        ng=self.g if self.whitened else self.model.prior['cov'].dot(self.g)

        # accumulate the power of force
        pw = rth/2*self.g.dot(v)

        # current energy
        E_cur = -self.ll - self.h/8*self.g.dot(ng)
        
        # correct volume if requested
        if self.vol_wts: logwt+=logvol(q,self.AE,'encode')

        randL=np.int(np.ceil(np.random.uniform(0,self.L)))

        for l in range(randL):
            # a half step for velocity
            v+=rth/2*ng

            # a full step for position
            q_=q.copy()
            q = cos_*q_ + sin_*v
            v = -sin_*q_ + cos_*v
#             rot=(q+1j*v)*np.exp(-1j*rth)
#             q=rot.real; v=rot.imag

            # update geometry
            ll,g,_,_=self.geom(q)
            ng=g if self.whitened else self.model.prior['cov'].dot(g)

            # another half step for velocity
            v+=rth/2*ng

            # accumulate the power of force
            if l!=randL-1: pw+=rth*g.dot(v)

        # accumulate the power of force
        pw += rth/2*g.dot(v)

        # new energy
        E_prp = -ll - self.h/8*g.dot(ng)
        
        # correct volume if requested
        if self.vol_wts: logwt+=logvol(q,self.AE,'decode')

        # Metropolis test
        logr=-E_prp+E_cur-pw+(logwt if self.vol_wts=='adjust' else 0)

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.ll=ll; self.g=g;
            acpt=True
        else:
            if self.vol_wts=='record': logwt=0
            acpt=False

        # return accept indicator
        return acpt,logr,logwt

    def DREAMinfmMALA(self):
        """
        dimension-reduced infinite dimensional manifold MALA under DREAM framework
        """
        logwt=0
        # initialization
        q=self.q.copy()
        rth=np.sqrt(self.h)
        
        # sample velocity
        v=self.randv(self.model.post_Ga)

        # natural gradient
        ng=self.model.post_Ga['postC'](self.g) # use low-rank posterior Hessian solver

        # update velocity
        v+=rth/2*ng

        # current energy
        E_cur = -self.ll - rth/2*self.g.dot(v) + self.h/8*self.g.dot(ng) +0.5*self.model.post_Ga['Hlr'].norm2(v) -0.5*sum(np.log(1+self.eigs[0])) # use low-rank Hessian inner product

        # correct volume if requested
        if self.vol_wts: logwt+=logvol(q,self.AE,'encode')
        
        # generate proposal according to simplified manifold Langevin dynamics
        q = ((1-self.h/4)*self.q + rth*v)/(1+self.h/4)

        # update velocity
        v = (-(1-self.h/4)*v + rth*self.q)/(1+self.h/4)

        # update geometry
        ll,g,_,eigs=self.geom(q)
        self.model.post_Ga['eigs']=eigs # update the eigen-pairs in low-rank approximation --important!

        # natural gradient
        ng=self.model.post_Ga['postC'](g)

        # new energy
        E_prp = -ll - rth/2*g.dot(v) + self.h/8*g.dot(ng) +0.5*self.model.post_Ga['Hlr'].norm2(v) -0.5*sum(np.log(1+eigs[0]))
        
        # correct volume if requested
        if self.vol_wts: logwt+=logvol(q,self.AE,'decode')

        # Metropolis test
        logr=-E_prp+E_cur+(logwt if self.vol_wts=='adjust' else 0)

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.ll=ll; self.g=g; self.eigs=eigs;
            acpt=True
        else:
            if self.vol_wts=='record': logwt=0
            acpt=False

        # return accept indicator
        return acpt,logr,logwt

    def DREAMinfmHMC(self):
        """
        dimension-reduced infinite dimensional manifold HMC under DREAM framework
        """
        logwt=0
        # initialization
        q=self.q.copy()
        rth=np.sqrt(self.h) # make the scale comparable to MALA
#         cos_=np.cos(rth); sin_=np.sin(rth);
        cos_=(1-self.h/4)/(1+self.h/4); sin_=rth/(1+self.h/4);

        # sample velocity
        v=self.randv(self.model.post_Ga)

        # natural gradient
        ng=self.model.post_Ga['postC'](self.g) # use low-rank posterior Hessian solver

        # accumulate the power of force
        pw = rth/2*np.linalg.solve(self.model.prior['cov'],v).dot(ng)

        # current energy
        E_cur = -self.ll + self.h/4*self.model.prior['logpdf'](ng) +0.5*self.model.post_Ga['Hlr'].norm2(v) -0.5*sum(np.log(1+self.eigs[0])) # use low-rank Hessian inner product

        # correct volume if requested
        if self.vol_wts: logwt+=logvol(q,self.AE,'encode')
        
        randL=np.int(np.ceil(np.random.uniform(0,self.L)))

        for l in range(randL):
            # a half step for velocity
            v+=rth/2*ng

            # a full step rotation
            q_=q.copy()
            q = cos_*q_ + sin_*v
            v = -sin_*q_ + cos_*v
#             rot=(q+1j*v)*np.exp(-1j*rth)
#             q=rot.real; v=rot.imag

            # update geometry
            ll,g,_,eigs=self.geom(q)
            self.model.post_Ga['eigs']=eigs # update the eigen-pairs in low-rank approximation --important!
            ng=self.model.post_Ga['postC'](g)

            # another half step for velocity
            v+=rth/2*ng

            # accumulate the power of force
            if l!=randL-1: pw+=rth*np.linalg.solve(self.model.prior['cov'],v).dot(ng)

        # accumulate the power of force
        pw += rth/2*np.linalg.solve(self.model.prior['cov'],v).dot(ng)

        # new energy
        E_prp = -ll + self.h/4*self.model.prior['logpdf'](ng) +0.5*self.model.post_Ga['Hlr'].norm2(v) -0.5*sum(np.log(1+eigs[0]))
        
        # correct volume if requested
        if self.vol_wts: logwt+=logvol(q,self.AE,'decode')

        # Metropolis test
        logr=-E_prp+E_cur-pw+(logwt if self.vol_wts=='adjust' else 0)

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.ll=ll; self.g=g; self.eigs=eigs;
            acpt=True
        else:
            if self.vol_wts=='record': logwt=0
            acpt=False

        # return accept indicator
        return acpt,logr,logwt
    
    def _init_h(self):
        """
        find a reasonable initial step size
        """
        h=1.
        _self=self
        sampler=getattr(_self,str(_self.alg_name))
        _self.h=h;_self.L=1
        _,logr=sampler()
        a=2.*(np.exp(logr)>0.5)-1.
        while a*logr>-a*np.log(2):
            h*=pow(2.,a)
            _self=self
            _self.h=h;_self.L=1
            _,logr=sampler()
        return h
    
    def _dual_avg(self,iter,an):
        """
        dual-averaging to adapt step size
        """
        hn_adpt=self.h_adpt
        hn_adpt['An']=(1.-1./(iter+hn_adpt['n0']))*hn_adpt['An'] + (hn_adpt['a0']-an)/(iter+hn_adpt['n0'])
        logh=hn_adpt['mu'] - np.sqrt(iter)/hn_adpt['gamma']*hn_adpt['An']
        hn_adpt['loghn']=pow(iter,-hn_adpt['kappa'])*logh + (1.-pow(iter,-hn_adpt['kappa']))*hn_adpt['loghn']
        hn_adpt['h']=np.exp(logh)
        return hn_adpt
    
    # sample with given method
    def sample(self,num_samp,num_burnin,num_retry_bad=0,**kwargs):
        """
        sample with given MCMC method
        """
        name_sampler = str(self.alg_name)
        try:
            sampler = getattr(self, name_sampler)
        except AttributeError:
            print(self.alg_name, 'not found!')
        else:
            print('\nRunning '+self.alg_name+' now...\n')

        # allocate space to store results
        self.samp=np.zeros((num_samp,self.dim))
        self.loglik=np.zeros(num_samp+num_burnin)
        self.logwts=np.zeros(num_samp+num_burnin)
        self.acpt=0.0 # final acceptance rate
        self.times=np.zeros(num_samp+num_burnin) # record the history of time used for each sample
        
        # number of adaptations for step size
        if self.adpt_h:
            self.h_adpt['n_adpt']=kwargs.pop('adpt_steps',num_burnin)
        
        # online parameters
        accp=0.0 # online acceptance
        num_cons_bad=0 # number of consecutive bad proposals

        beginning=timeit.default_timer()
        for s in range(num_samp+num_burnin):

            if s==num_burnin:
                # start the timer
                tic=timeit.default_timer()
                print('\nBurn-in completed; recording samples now...\n')

            # generate MCMC sample with given sampler
            while True:
                try:
                    acpt_idx,logr,logwt=sampler()
                except RuntimeError as e:
                    print(e)
                    if num_retry_bad==0:
                        acpt_idx=False; logr=-np.inf; logwt=0
                        print('Bad proposal encountered! Passing... bias introduced.')
                        break # reject bad proposal: bias introduced
                    else:
                        num_cons_bad+=1
                        if num_cons_bad<num_retry_bad:
                            print('Bad proposal encountered! Retrying...')
                            continue # retry until a valid proposal is made
                        else:
                            acpt_idx=False; logr=-np.inf; logwt=0
                            num_cons_bad=0
                            print(str(num_retry_bad)+' consecutive bad proposals encountered! Passing...')
                            break # reject it and keep going
                else:
                    num_cons_bad=0
                    break

            accp+=acpt_idx

            # display acceptance at intervals
            if (s+1)%100==0:
                print('\nAcceptance at %d iterations: %0.2f' % (s+1,accp/100))
                accp=0.0

            # save results
            self.loglik[s]=self.ll
            self.logwts[s]=logwt
            if s>=num_burnin:
                self.samp[s-num_burnin,]=self.q.T
                self.acpt+=acpt_idx
            
            # record the time
            self.times[s]=timeit.default_timer()-beginning
            
            # adapt step size h if needed
            if self.adpt_h:
                if s<self.h_adpt['n_adpt']:
                    self.h_adpt=self._dual_avg(s+1,np.exp(min(0,logr)))
                    self.h=self.h_adpt['h']
                    print('New step size: %.2f; \t New averaged step size: %.6f\n' %(self.h_adpt['h'],np.exp(self.h_adpt['loghn'])))
                if s==self.h_adpt['n_adpt']:
                    self.h_adpt['h']=np.exp(self.h_adpt['loghn'])
                    self.h=self.h_adpt['h']
                    print('Adaptation completed; step size freezed at:  %.6f\n' % self.h_adpt['h'])

        # stop timer
        toc=timeit.default_timer()
        self.time=toc-tic
        self.acpt/=num_samp
        print("\nAfter %g seconds, %d samples have been collected with the final acceptance rate %0.2f \n"
              % (self.time,num_samp,self.acpt))

        # save to file
        self.save_samp()

    # save samples
    def save_samp(self):
        import os,errno
        import pickle
        # create folder
        cwd=os.getcwd()
        self.savepath=os.path.join(cwd,'result')
        try:
            os.makedirs(self.savepath)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
            else:
                raise
        # name file
        ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.filename=self.alg_name+'_dim'+str(self.dim)+'_'+ctime
        # dump data
        f=open(os.path.join(self.savepath,self.filename+'.pckl'),'wb')
        res2save=[self.h,self.L,self.alg_name,self.samp,self.loglik,self.logwts,self.acpt,self.time,self.times]
        if self.adpt_h:
            res2save.append(self.h_adpt)
        pickle.dump(res2save,f)
        f.close()
#         # load data
#         f=open(os.path.join(self.savepath,self.filename+'.pckl'),'rb')
#         f_read=pickle.load(f)
#         f.close()
