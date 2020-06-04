"""
Variational Bayes
-----------------
Shiwei Lan @ ASU, May 2020
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2020"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com"

import numpy as np
import scipy as sp
import scipy.optimize

class VB:
    """
    Vanilla variational Bayes
    -------------------------
    Input: log of probability, usually log-posterior logP(Z|X) in Bayesian statistics
           should be specified as log-likelihood function logP(X|Z) (with gradient) if Gaussian prior P(Z) is used and supplied
    Output: optimal variational parameters (mean and std) of approximating independent Gaussians Q(Z): Z_i ~ N(mean[i],std[i])
    This is achieved by optimizing the following evidence lower bound (ELBO):
    L(X) = -E_Q[logQ(Z)] + E_Q[logP(X|Z)] + E_Q[logP(Z)]
         = H(Q) + E_Q[logP(X|Z)] - H(Q;P(Z))
    where Gaussian entropy H(Q) has analytic form, so does cross Gaussian entropy H(Q;P(Z)) if P(Z) is Gaussian prior;
          E_Q[logP(X|Z)] is approximated by Monte Carlo method.
    """
    def __init__(self,logprob,varpars,num_samples=100,**kwargs):
        '''
        Initialization of a variational Bayes
        logprob: log-posterior (log-likelihood if Gaussian prior is supplied)
        varpars: variational parameters mean and std
        num_samples: number of Monte Carlo samples in approximating the above expectation
        gaussion_prior: Gaussian prior as a dictionary {'mean':mean, 'cov':covariance} or defined externally
        kwargs: optimization parameters
        '''
        self.logprob=logprob
        self.varpars=varpars
        self.D=varpars.shape[1]
        self.num_samples=num_samples
        self.gaussian_prior=kwargs.pop('gaussian_prior',None)
        self.kwargs=kwargs
    
    def gaussian_entropy(self,std=None,grad=False):
        '''
        Gaussian Entropy H(Q)=-E_Q[logQ(Z)]
        '''
        if std is None:
            std=self.varpars[1]
        H=np.sum(np.log(np.abs(std)))
        if not grad:
            return H,
        else:
            dH=np.concatenate((np.zeros(self.D),1./std))
            return H,dH
    
    def _gaussian_cross_entropy(self,mean=None,std=None,grad=False):
        '''
        Gaussian Cross Entropy H(Q;P(Z))=-E_Q[logP(Z)]
        '''
        if mean is None:
            mean=self.varpars[0]
        if std is None:
            std=self.varpars[1]
        if self.gaussian_prior:
            invCm=np.linalg.solve(self.gaussian_prior['cov'],mean-self.gaussian_prior['mean'])
            diaginvC=np.diag(np.inv(self.gaussian_prior['cov']))
            xH=.5*( (mean-self.gaussian_prior['mean']).dot(invCm) + np.sum(diaginvC*std**2) )
        else:
            xH=0
        if not grad:
            return xH,
        else:
            if self.gaussian_prior:
                dxH=np.concatenate( (invCm,diaginvC*std) )
            else:
                dxH=np.zeros(2*self.D)
            return xH,dxH
    
    def expected_logprob(self,mean=None,std=None,t=0,grad=False):
        '''
        Empirical expectation of log-probability logP(X,Z) wrt approximating Gaussian:
        \sum_{i=1}^num_samples logP(X,Z^i) /num_samples, Z^i ~ N(mean, std)
        '''
        if mean is None:
            mean=self.varpars[0]
        if std is None:
            std=self.varpars[1]
        rs=np.random.RandomState(t)
        noise=rs.randn(self.num_samples, self.D)
        samples = noise * std + mean
        logProb = self.logprob(samples,grad=grad)
        exp_logprob = np.mean(logProb[0])
        if not grad:
            return exp_logprob,
        else:
            dlogprob = logProb[1]
            dexp_logprob = np.mean( np.concatenate((dlogprob,dlogprob*noise),axis=1), axis=0 )
#             dexp_logprob = np.mean( logProb[0][:,None]*np.concatenate((noise/std,(-1+noise**2)/np.abs(std)),axis=1), axis=0 ) # theoretic
            return exp_logprob,dexp_logprob
    
    def objective(self,params=None,t=2020,grad=True):
        '''
        Objective function negative ELBO: -L(X) = -( H(Q) + E_Q[logP(X|Z)] - H(Q;P(Z)) )
        '''
        if params is None:
            params=self.varpars
        mean, std = params[:self.D], params[self.D:]
        H=self.gaussian_entropy(std, grad)
        exp_logprob=self.expected_logprob(mean, std, t, grad)
        xH=self._gaussian_cross_entropy(mean, std, grad)
        elbo = H[0] + exp_logprob[0] - xH[0] # negative ELBO
        if not grad:
            return -elbo
        else:
            grad_elbo = H[1] + exp_logprob[1] - xH[1] # negative ELBO gradient
            return -elbo, -grad_elbo
    
    def optimize(self,callback=None,**kwargs):
        '''
        Minimize the objective function wrt variational parameters mean and std
        '''
        self.kwargs.update(kwargs)
        method=self.kwargs.pop('method','Nelder-Mead')
#         res=sp.optimize.minimize(self.objective,self.varpars.flatten(),args=np.random.randint(2020),method='CG',jac=lambda params,t:self.objective(params, t, True)[1],callback=callback,**self.kwargs)
        grad=(method!='Nelder-Mead')
        res=sp.optimize.minimize(self.objective,self.varpars.flatten(),args=(2020,grad),method=method,jac=grad,**self.kwargs)
#         callback(res.x,res.nit)
#         opt=self.kwargs.pop('options',None)
#         max_iter=opt['maxiter'] if opt else 100
#         params=self.varpars.flatten()
#         for t in range(max_iter):
# #             res=sp.optimize.minimize(self.objective,params,args=t,method='L-BFGS-B',jac=lambda params,t:self.objective(params, t, True)[1],options={'maxiter':1})
#             res=sp.optimize.minimize(self.objective,params,args=t,method='Nelder-Mead',jac=True,options={'gtol': 1e-3,'maxiter':1})
#             params=res.x
#             if callback: callback(params,t)
#             if res.success: break
        
        return res

if __name__ == '__main__':
    np.random.seed(2020)
    from scipy.stats import norm

    # Specify an inference problem by its unnormalized log-density.
    D = 2
    def log_density(x,grad=False):
        mu, log_sigma = x[:, 0], x[:, 1]
        sigma_logpdf = norm.logpdf(log_sigma, 0, 1.35)
        mu_logpdf = norm.logpdf(mu, 0, np.exp(log_sigma))
        logpdf=sigma_logpdf + mu_logpdf
        if not grad:
            return logpdf,
        else:
            dsigma_logpdf = np.hstack((np.zeros((x.shape[0],1)),(-log_sigma/1.35**2)[:,None]))
            dmu_logpdf = np.vstack((-mu*np.exp(-2*log_sigma),-1+mu**2*np.exp(-2*log_sigma))).T
            dlogpdf=dsigma_logpdf + dmu_logpdf
        return logpdf,dlogpdf
    
#     a=np.random.randn(4,D)
#     v=np.random.randn(4,D)
#     eps=1e-6
#     l,g=log_density(a,True)
#     kk=(log_density(a+eps*v)-l)/eps-np.sum(g*v,axis=1)
#     print(kk)
    
    # Build variational object.
    init_mean    = -1 * np.ones(D)
    init_std = -1 * np.ones(D)
    init_var_params = np.vstack([init_mean, init_std])
    vb=VB(log_density,init_var_params,num_samples=1000)
    
#     v,w=np.random.randn(D),np.random.randn(D)
#     eps=1e-7
#     l,g=vb.expected_logprob(init_mean,init_std,grad=True)
#     kk=(vb.expected_logprob(init_mean+eps*v, init_std+eps*w)-l)/eps -g.dot(np.concatenate((v,w)))
#     print(kk)
    
    # Set up plotting code
    def plot_isocontours(ax, func, xlimits=[-2, 2], ylimits=[-4, 2], numticks=101):
        x = np.linspace(*xlimits, num=numticks)
        y = np.linspace(*ylimits, num=numticks)
        X, Y = np.meshgrid(x, y)
        zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
        Z = zs.reshape(X.shape)
        plt.contour(X, Y, Z)
        ax.set_yticks([])
        ax.set_xticks([])

    # Set up figure.
    import matplotlib.pylab as plt
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)
    
    from scipy.stats import multivariate_normal as mvn
#     Nfeval = 1
    def callback(params,t):
        print("Iteration {} lower bound {}".format(t, -vb.objective(params, t)[0]))
#         global Nfeval
#         Nfeval+=1
#         print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f} '.format(Nfeval, params[0], params[1], vb.objective(params)) )

        plt.cla()
        target_distribution = lambda x : np.exp(log_density(x))
        plot_isocontours(ax, target_distribution)
 
        mean, std = params[:D], params[D:]
        variational_contour = lambda x: mvn.pdf(x, mean, np.diag(std**2))
        plot_isocontours(ax, variational_contour)
        plt.draw()
        plt.pause(1.0/30.0)

    print("Optimizing variational parameters...")
    options={'maxiter':300,'disp':True}
    variational_params = vb.optimize(method='BFGS',callback=callback,options=options)
    print(variational_params.x)
    print(variational_params.fun)
