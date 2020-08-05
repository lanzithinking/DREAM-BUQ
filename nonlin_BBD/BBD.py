#!/usr/bin/env python
"""
Banana-Biscuit-Donought (BBD) distribution
------------------------------------------
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
import matplotlib.pyplot as plt
np.random.seed(2020)

class BBD:
    """
    Non-linear Banana-Biscuit-Donought (BBD) distribution
    -----------------------------------------------------
    likelihoood: y = G(u) + eta, eta ~ N(0, sigma^2_eta I)
    forward mapping: G(u) = A Su, Su = [u_1, u_2^2, ..., u_k^p(k), ..., u_d^p(d)], p(k) = 2-(k mod 2)
    prior: u ~ N(0, C)
    posterior: u|y follows BBD distribution because it resembles:
               a banana in (i,j) dimension if i,j are one odd and one even;
               a biscuit in (i,j) dimension if both i,j are odd;
               and a doughnut in (i,j) dimension if both i,j are even.
    """
    def __init__(self,input_dim=4,output_dim=100,linop=None,nz_var=4.,pr_cov=1.,**kwargs):
        """
        Initialization
        """
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.linop=linop
        if self.linop is None:
            self.A=kwargs.pop('A',np.ones((self.output_dim,self.input_dim)))
            self.linop=lambda x: self.A.dot(x) if np.ndim(x)==1 else x.dot(self.A.T)
        self.nz_var=nz_var
        if np.size(self.nz_var)<self.output_dim: self.nz_var=np.resize(self.nz_var,self.output_dim)
        self.pr_cov=pr_cov
        if np.ndim(self.pr_cov)<2 and np.size(self.pr_cov)<self.input_dim: self.pr_cov=np.resize(self.pr_cov,self.input_dim)
        self.true_input=kwargs.pop('true_input',np.random.rand(self.input_dim))
        self.y=kwargs.pop('y',self._generate_data())
    
    def forward(self,input):
        """
        Forward mapping
        """
        input_=input.copy()
        if np.ndim(input)==1:
            input_[1::2]=input_[1::2]**2
        elif np.ndim(input)==2:
            input_[:,1::2]=input_[:,1::2]**2
        output=self.linop(input_)
        return output
    
    def _generate_data(self):
        """
        Generate data
        """
        fwdout=self.forward(self.true_input)
        y=fwdout+np.sqrt(self.nz_var)*np.random.randn(self.output_dim)
        return y
    
    def logpdf(self,input,type='likelihood'):
        """
        Log probability density function
        """
        fwdout=self.forward(input)
        loglik=-0.5*np.sum((self.y-fwdout)**2/self.nz_var) if np.ndim(input)==1 else -0.5*np.sum((self.y[None,:]-fwdout)**2/self.nz_var[None,:],axis=1)
        if type=='posterior':
            if np.ndim(input)==1:
                logpri=-0.5*np.sum(input**2/self.pr_cov) if np.ndim(self.pr_cov)==1 else -0.5*input.dot(np.linalg.solve(self.pr_cov,input))
            else:
                logpri=-0.5*np.sum(input**2/self.pr_cov[None,:] if np.ndim(self.pr_cov)==1 else input*np.linalg.solve(self.pr_cov,input.T).T, axis=1)
        elif type=='likelihood':
            logpri=0
        return loglik+logpri
    
    def sample(self,prng=np.random.RandomState(2020),num_samp=1,type='prior'):
        """
        Generate sample
        """
        samp=None
        if type=='prior':
            samp=np.sqrt(self.pr_cov)*prng.randn(num_samp,self.input_dim) if np.ndim(self.pr_cov)==1 else prng.multivariate_normal(np.zeros(self.input_dim),self.pr_cov,num_samp)
        return samp
    
    def plot_2dcontour(self,dim=[1,2],type='posterior',**kwargs):
        """
        Plot selected 2d contour of density function
        """
        x=np.linspace(self.true_input[dim[0]]-2.,self.true_input[dim[0]]+2.)
        y=np.linspace(self.true_input[dim[1]]-2.,self.true_input[dim[1]]+2.)
        X,Y=np.meshgrid(x,y)
        Input=np.zeros((X.size,self.input_dim))
        Input[:,dim[0]],Input[:,dim[1]]=X.flatten(),Y.flatten()
        Z=self.logpdf(Input, type).reshape(X.shape)
        levels=kwargs.pop('levels',20)
        if 'ax' in kwargs:
            ax=kwargs.pop('ax')
            return ax.contourf(X,Y,Z,levels,**kwargs)
        else:
            plt.contour(X,Y,Z,levels,**kwargs)
            plt.show()

if __name__ == '__main__':
    import os,sys
    sys.path.append( "../" )
    from util.common_colorbar import common_colorbar
    np.random.seed(2020)
    
    # set up
    d=4; m=100
#     nz_var=4 # classic
    nz_var=4
#     true_input=np.random.rand(d) # classic
    true_input=np.random.randint(d,size=d)
#     A=np.ones((m,d)) # classic
    A=np.random.rand(m,d)
    bbd=BBD(d,m,nz_var=nz_var,true_input=true_input,A=A)
    # save data
    if not os.path.exists('./result'): os.makedirs('./result')
    with open('./result/BBD.pickle','wb') as f:
        pickle.dump([bbd.true_input,bbd.y],f)
    
    # plot
    plt.rcParams['image.cmap'] = 'jet'
    dims=np.vstack(([0,1],[0,2],[1,3]))
    fig_names=['Banana','Biscuit','Donought']
    fig,axes = plt.subplots(nrows=1,ncols=dims.shape[0],sharex=False,sharey=False,figsize=(16,5))
    sub_figs = [None]*len(axes.flat)
    for i,ax in enumerate(axes.flat):
#         plt.axes(ax)
        sub_figs[i]=bbd.plot_2dcontour(dims[i],ax=ax)
        ax.plot(bbd.true_input[dims[i,0]],bbd.true_input[dims[i,1]],'kx',markersize=10,mew=2)
        ax.set_xlabel('$u_{}$'.format(dims[i,0]+1))
        ax.set_ylabel('$u_{}$'.format(dims[i,1]+1),rotation=0)
        ax.set_title(fig_names[i])
        ax.set_aspect('auto')
    fig=common_colorbar(fig,axes,sub_figs)
    plt.subplots_adjust(wspace=0.2, hspace=0)
    # save plot
    # fig.tight_layout()
    plt.savefig('./result/bbd.png',bbox_inches='tight')
    