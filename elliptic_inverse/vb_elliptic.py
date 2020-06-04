"""
Variational Bayes for UQ of elliptic inverse problem
----------------------------------------------------
Shiwei Lan @ ASU, May 2020
"""

import numpy as np
import sys
sys.path.append( "../" )
from util.linalg import *
from vb.vb import VB 

# the inverse problem
from Elliptic import Elliptic

np.random.seed(2020)

## define the inverse elliptic problem ##
# parameters for PDE model
nx=40;ny=40;
# parameters for prior model
sigma=1.25;s=0.0625
# parameters for misfit model
SNR=50 # 100
# define the inverse problem
elliptic=Elliptic(nx=nx,ny=ny,SNR=SNR,sigma=sigma,s=s)


invC_diag=elliptic.prior.gen_vector()
#     invCop=Solver2Operator(elliptic.prior.Ksolver,init_vector=elliptic.prior.init_vector)
#     get_diagonal(invCop,invC_diag) # it takes too long
estimate_diagonal_inv2(elliptic.prior.Ksolver,500,invC_diag,init_vector=elliptic.prior.init_vector)

# Specify an inference problem by its unnormalized log-density.
D = elliptic.prior.dim
def log_density(x, grad=False):
    assert x.shape[1]==elliptic.prior.dim, 'size of params is incorrect!'
    
    geom=[0] if not grad else [0,1]
    logpdf=np.zeros(x.shape[0])
    dlogpdf=np.zeros(x.shape)
    for i in range(x.shape[0]):
        u=elliptic.prior.gen_vector(x[i])
        res=elliptic.get_geom(u,geom)
        logpdf[i]=res[0]
#         logpdf[i]=res[0]+elliptic.prior.logpdf(u) # Gaussian prior can be calculated as cross entropy
        if grad:
            dlogpdf[i]=res[1].get_local()
#             dlogpdf[i]=(res[1]-elliptic.prior.C_act(u,-1)).get_local()
    if not grad:
        return logpdf,
    else:
        return logpdf,dlogpdf

# a=np.random.randn(2,D)
# v=np.random.randn(2,D)
# eps=1e-4
# l,g=log_density(a,True)
# kk=(log_density(a+eps*v)-l)/eps-np.sum(g*v,axis=1)
# print(kk)

# Build variational object.
init_mean    = np.zeros(D)
init_std = -1 * np.ones(D)
init_var_params = np.vstack([init_mean, init_std])
vb=VB(log_density,init_var_params,num_samples=500)

# v,w=np.random.randn(D),np.random.randn(D)
# eps=1e-3
# l,g=vb.expected_logprob(init_mean,init_std,grad=True)
# kk=(vb.expected_logprob(init_mean+eps*v, init_std+eps*w)-l)/eps -g.dot(np.concatenate((v,w)))
# print(kk)

# redefine gaussian cross entropy for elliptic prior
def _gaussian_cross_entropy(mean,std,grad=False):
    u=elliptic.prior.gen_vector(mean)
    invCm=elliptic.prior.C_act(u,-1)
    diaginvC=invC_diag.get_local()
    xH=.5*( invCm.inner(u) + np.sum(diaginvC*std**2) )
    if not grad:
        return xH,
    else:
        dxH=np.concatenate((invCm.get_local(), diaginvC*std) )
        return xH,dxH
vb._gaussian_cross_entropy=_gaussian_cross_entropy
# vb._grad_gaussian_cross_entropy=_grad_gaussian_cross_entropy

# v,w=np.random.randn(D),np.random.randn(D)
# eps=1e-8
# l,g=vb._gaussian_cross_entropy(init_mean,init_std,True)
# kk=(vb._gaussian_cross_entropy(init_mean+eps*v, init_std+eps*w)-l)/eps -g.dot(np.concatenate((v,w)))
# print(kk)

# Set up plotting code
def plot_isocontours(ax, func, xlimits=[-2, 2], ylimits=[-4, 2], numticks=101):
    x = anp.linspace(*xlimits, num=numticks)
    y = anp.linspace(*ylimits, num=numticks)
    X, Y = anp.meshgrid(x, y)
    zs = func(anp.concatenate([anp.atleast_2d(X.ravel()), anp.atleast_2d(Y.ravel())]).T)
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z)
    ax.set_yticks([])
    ax.set_xticks([])

# Set up figure.
#     fig = plt.figure(figsize=(8,8), facecolor='white')
#     ax = fig.add_subplot(111, frameon=False)
#     plt.ion()
#     plt.show(block=False)

def callback(params, t, g):
    print("\nIteration {} lower bound {}".format(t, -objective(params, t)[0]))
    print('Mean parameters: min {} and max {}'.format(params[:D].min(),params[:D].max()))
    print('Std parameters: min {} and max {}'.format(params[D:].min(),params[D:].max()))
    print('Gradient of mean parameters: min {} and max {}'.format(g[:D].min(),g[:D].max()))
    print('Gradient of std parameters: min {} and max {}'.format(g[D:].min(),g[D:].max()))

#         plt.cla()
#         target_distribution = lambda x : anp.exp(log_density(x, t))
#         plot_isocontours(ax, target_distribution)
# 
#         mean, std = unpack_params(params)
#         variational_contour = lambda x: mvn.pdf(x, mean, np.diag(std**2))
#         plot_isocontours(ax, variational_contour)
#         plt.draw()
#         plt.pause(1.0/30.0)

print("Optimizing variational parameters...")
options={'maxiter':50,'disp':True,'gtol':1e-5}
import timeit
t_start=timeit.default_timer()
res = vb.optimize(method='L-BFGS-B',tol=1e-5,callback=None,options=options)
t_used=timeit.default_timer()-t_start
print('Time used: {}'.format(t_used))

mean, std = res.x[:D], np.abs(res.x[D:])
elbo = -res.fun

# save result
import os,errno
import time,pickle
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
filename='vb'+'_dim'+str(D)+'_'+ctime
# save data
# f=open(os.path.join(savepath,filename+'.pckl'),'wb')
# pickle.dump((elbo,mean,std),f)
# f.close()
np.savez_compressed(file=os.path.join(savepath,filename),elbo=elbo,mean=mean,std=std,t_used=t_used)
