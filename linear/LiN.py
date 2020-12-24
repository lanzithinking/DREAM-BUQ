#!/usr/bin/env python
"""
Linear-Gaussian Inverse Problem
-------------------------------
Shiwei Lan @ ASU, 2020
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2020, The NN-MCMC project"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
import pickle
np.random.seed(2020)

class LiN:
    """
    Linear-Gaussian Inverse Problem
    -----------------------------------------------------
    likelihoood: y = G(u) + eta, eta ~ N(0, sigma^2_eta I)
    forward mapping: G(u) = A u
    prior: u ~ N(0, C)
    posterior: u|y follows Gaussian distribution
    """
    def __init__(self,input_dim=3,output_dim=100,linop=None,nz_var=4.,pr_cov=1.,**kwargs):
        """
        Initialization
        """
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.linop=linop
        if self.linop is None:
            self.A=kwargs.pop('A',np.ones((self.output_dim,self.input_dim))) # (m,d)
            self.linop=lambda x: self.A.dot(x) if np.ndim(x)==1 else x.dot(self.A.T)
        self.nz_var=nz_var
        if np.size(self.nz_var)<self.output_dim: self.nz_var=np.resize(self.nz_var,self.output_dim)
        self.pr_cov=pr_cov
        if np.ndim(self.pr_cov)<2 and np.size(self.pr_cov)<self.input_dim: self.pr_cov=np.resize(self.pr_cov,self.input_dim)
        self.true_input=kwargs.pop('true_input',np.random.rand(self.input_dim))
        self.y=kwargs.pop('y',self._generate_data())
        # analytic solution
        if np.ndim(self.pr_cov)<2:
            ASigma0=self.A*self.pr_cov[None,:]
            self.post_cov=np.diag(self.pr_cov)-ASigma0.T.dot(np.linalg.solve(np.diag(self.nz_var)+ASigma0.dot(self.A.T),ASigma0))
        elif np.ndim(self.pr_cov)==2:
            ASigma0=self.A.dot(self.pr_cov)
            self.post_cov=self.pr_cov-ASigma0.T.dot(np.linalg.solve(np.diag(self.nz_var)+ASigma0.dot(self.A.T),ASigma0))
        self.post_mean=self.post_cov.dot(self.A.T.dot(self.y/self.nz_var))
    
    def forward(self,input,geom_ord=0):
        """
        Forward mapping
        """
        if geom_ord==0:
            output=self.linop(input) # (m,) or (n,m)
        elif geom_ord==1:
            output=self.A if np.ndim(input)==1 else np.tile(self.A,(input.shape[0],)+(1,)*np.ndim(input)) # (m,d) or (n,m,d)
        return output
    
    def _generate_data(self):
        """
        Generate data
        """
        fwdout=self.forward(self.true_input)
        y=fwdout+np.sqrt(self.nz_var)*np.random.randn(self.output_dim)
        return y
    
    def logpdf(self,input,type='likelihood',geom_ord=[0]):
        """
        Log probability density function and its gradient
        """
        fwdout=self.forward(input)
        if 0 in geom_ord: loglik=-0.5*np.sum((self.y-fwdout)**2/self.nz_var) if np.ndim(input)==1 else -0.5*np.sum((self.y[None,:]-fwdout)**2/self.nz_var[None,:],axis=1)
        if 1 in geom_ord:
            dfwdout=self.forward(input,1)
            dloglik=np.sum(((self.y-fwdout)/self.nz_var)[:,None]*dfwdout,axis=0) if np.ndim(input)==1 else np.sum(((self.y[None,:]-fwdout)/self.nz_var[None,:])[:,:,None]*dfwdout,axis=1)
        if type=='posterior':
            if np.ndim(input)==1:
                dlogpri=-input/self.pr_cov if np.ndim(self.pr_cov)==1 else -np.linalg.solve(self.pr_cov,input)
                logpri=0.5*input.dot(dlogpri)
            else:
                dlogpri=-input/self.pr_cov[None,:] if np.ndim(self.pr_cov)==1 else -np.linalg.solve(self.pr_cov,input.T).T
                logpri=0.5*np.sum(input*dlogpri, axis=1)
        elif type=='likelihood':
            logpri=0; dlogpri=0
        out=[]
        if 0 in geom_ord: out.append(loglik+logpri)
        if 1 in geom_ord: out.append(dloglik+dlogpri)
        return out
    
    def get_geom(self,input,geom_ord=[0],**kwargs):
        """
        Get geometric quantities of loglikelihood
        """
        loglik=None; gradlik=None; metact=None; rtmetact=None; eigs=None
        
        out=self.logpdf(input,geom_ord=geom_ord)
        if 0 in geom_ord: loglik=out[0]
        if 1 in geom_ord: gradlik=out[-1]
        
        if any(s>1 for s in geom_ord):
            print('Requested geometric quantity not provided yet!')
        
        if len(kwargs)==0:
            return loglik,gradlik,metact,rtmetact
        else:
            return loglik,gradlik,metact,eigs
    
    def sample(self,prng=np.random.RandomState(2020),num_samp=1,type='prior'):
        """
        Generate sample
        """
        samp=None
        if type=='prior':
            samp=np.sqrt(self.pr_cov)*prng.randn(num_samp,self.input_dim) if np.ndim(self.pr_cov)==1 else prng.multivariate_normal(np.zeros(self.input_dim),self.pr_cov,num_samp)
        elif type=='posterior':
            samp=prng.multivariate_normal(self.post_mean,self.post_cov,num_samp)
        return np.squeeze(samp)
    
    def plot_2dcontour(self,dim=[0,1],type='posterior',**kwargs):
        """
        Plot selected 2d contour of density function
        """
        x=np.linspace(self.true_input[dim[0]]-2.,self.true_input[dim[0]]+2.)
        y=np.linspace(self.true_input[dim[1]]-2.,self.true_input[dim[1]]+2.)
        X,Y=np.meshgrid(x,y)
#         Input=np.zeros((X.size,self.input_dim))
        Input=np.tile(self.true_input,(X.size,1))
        Input[:,dim[0]],Input[:,dim[1]]=X.flatten(),Y.flatten()
        Z=self.logpdf(Input, type)[0].reshape(X.shape)
        levels=kwargs.pop('levels',20)
        grad=kwargs.pop('grad',False)
        if grad:
            x=np.linspace(self.true_input[dim[0]]-2.,self.true_input[dim[0]]+2.,10)
            y=np.linspace(self.true_input[dim[1]]-2.,self.true_input[dim[1]]+2.,10)
            X_,Y_=np.meshgrid(x,y)
#             Input=np.zeros((X_.size,self.input_dim))
            Input=np.tile(self.true_input,(X_.size,1))
            Input[:,dim[0]],Input[:,dim[1]]=X_.flatten(),Y_.flatten()
            G=self.logpdf(Input, type, [1])[0]
            U,V=G[:,dim[0]].reshape(X_.shape),G[:,dim[1]].reshape(X_.shape)
        if 'ax' in kwargs:
            ax=kwargs.pop('ax')
            fig=ax.contourf(X,Y,Z,levels,**kwargs)
            ax.set_xlabel('$u_{}$'.format(dim[0]+1))
            ax.set_ylabel('$u_{}$'.format(dim[1]+1),rotation=0)
            if grad: ax.quiver(X_,Y_,U,V)
            return fig
        else:
            plt.contour(X,Y,Z,levels,**kwargs)
            plt.xlabel('$u_{}$'.format(dim[0]+1))
            plt.ylabel('$u_{}$'.format(dim[1]+1),rotation=0)
            if grad: plt.quiver(X_,Y_,U,V)
            plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os,sys
    sys.path.append( "../" )
    from util.common_colorbar import common_colorbar
    np.random.seed(2020)
    
    # set up
    d=3; m=100
    nz_var=.1; pr_cov=1.
#     true_input=np.random.rand(d) # classic
    true_input=np.arange(-np.floor(d/2),np.ceil(d/2))
#     true_input=np.random.randint(d,size=d)
#     A=np.ones((m,d)) # classic
    A=np.random.rand(m,d)
    lin=LiN(d,m,nz_var=nz_var,pr_cov=pr_cov,true_input=true_input,A=A)
    # save data
    if not os.path.exists('./result'): os.makedirs('./result')
    with open('./result/lin.pickle','wb') as f:
        pickle.dump([lin.nz_var,lin.pr_cov,lin.A,lin.true_input,lin.y],f)
    
    # check gradient
    f,g=lin.logpdf(true_input[None,:],type='posterior',geom_ord=[0,1])
    v=np.random.randn(1,d)
    h=1e-8
    gv_fd=(lin.logpdf(true_input+h*v,type='posterior')-f)/h
    reldif=abs(gv_fd-g.dot(v.T))/np.linalg.norm(v)
    print('Relative difference between finite difference and exacted results: {}'.format(reldif))
    
    # plot
    plt.rcParams['image.cmap'] = 'jet'
    dims=np.vstack(([0,1],[0,2],[1,2]))
    fig,axes = plt.subplots(nrows=1,ncols=dims.shape[0],sharex=False,sharey=False,figsize=(16,5))
    sub_figs = [None]*len(axes.flat)
    for i,ax in enumerate(axes.flat):
#         plt.axes(ax)
        sub_figs[i]=lin.plot_2dcontour(dims[i],ax=ax, grad=True)
        ax.plot(lin.true_input[dims[i,0]],lin.true_input[dims[i,1]],'kx',markersize=10,mew=2)
#         ax.set_xlabel('$u_{}$'.format(dims[i,0]+1))
#         ax.set_ylabel('$u_{}$'.format(dims[i,1]+1),rotation=0)
        ax.set_aspect('auto')
    fig=common_colorbar(fig,axes,sub_figs)
    plt.subplots_adjust(wspace=0.2, hspace=0)
    # save plot
    # fig.tight_layout()
    plt.savefig('./result/lin.png',bbox_inches='tight')
    