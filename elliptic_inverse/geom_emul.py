"""
Geometric functions by emulator emulation
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import sys,os
sys.path.append( "../" )
from util.dolfin_gadget import vec2fun,fun2img,img2fun
from util.multivector import *
from util.Eigen import *
from posterior import *

def geom(unknown,bip,emulator,geom_ord=[0],whitened=False,**kwargs):
    loglik=None; gradlik=None; metact=None; rtmetact=None; eigs=None
    
    # un-whiten if necessary
    if whitened:
        unknown=self.prior.v2u(unknown)
    
    u_input={'DNN':unknown.get_local()[None,:], 'CNN':fun2img(vec2fun(unknown,bip.pde.V))[None,:,:,None]}[type(emulator).__name__]
    
    ll_f = lambda x: -0.5*bip.misfit.prec*tf.math.reduce_sum((emulator.model(x)-bip.misfit.obs)**2,axis=1)
    
    if any(s>=0 for s in geom_ord):
        loglik = ll_f(u_input).numpy()
    
    if any(s>=1 for s in geom_ord):
        gradlik = img2fun(emulator.gradient(u_input, ll_f), bip.pde.V).vector()
        if whitened:
            gradlik = bip.prior.C_act(gradlik,.5,op='C',transp=True)
    
    if any(s>=1.5 for s in geom_ord):
        jac_ = emulator.jacobian(u_input)
        n_obs = len(bip.misfit.idx)
        jac = MultiVector(unknown,n_obs)
        [jac[i].set_local({'DNN':jac_[i],'CNN':img2fun(jac_[i], bip.pde.V).vector()}[type(emulator).__name__]) for i in range(n_obs)]
        def _get_metact_misfit(u_actedon): # GNH
            if type(u_actedon) is not df.Vector:
                u_actedon = bip.prior.gen_vector(u_actedon)
            v = bip.prior.gen_vector()
            jac.reduce(v,bip.misfit.prec*jac.dot(u_actedon))
            return bip.prior.M*v
        def _get_rtmetact_misfit(u_actedon):
            if type(u_actedon) is df.Vector:
                u_actedon = u_actedon.get_local()
            v = bip.prior.gen_vector()
            jac.reduce(v,np.sqrt(bip.misfit.prec)*u)
            return bip.prior.rtM*v
        metact = _get_metact_misfit
        rtmetact = _get_rtmetact_misfit
        if whitened:
            metact = lambda u: bip.prior.C_act(_get_metact_misfit(bip.prior.C_act(u,.5,op='C')),.5,op='C',transp=True) # ppGNH
            rtmetact = lambda u: bip.prior.C_act(_get_rtmetact_misfit(u),.5,op='C',transp=True)
    
    if any(s>1 for s in geom_ord) and len(kwargs)!=0:
        if whitened:
            # generalized eigen-decomposition (_C^(1/2) F _C^(1/2), M), i.e. _C^(1/2) F _C^(1/2) = M V D V', V' M V = I
            def invM(a):
                a=bip.prior.gen_vector(a)
                invMa=bip.prior.gen_vector()
                bip.prior.Msolver.solve(invMa,a)
                return invMa
            eigs = geigen_RA(metact, lambda u: bip.prior.M*u, invM, dim=bip.pde.V.dim(),**kwargs)
        else:
            # generalized eigen-decomposition (F, _C^(-1)), i.e. F = _C^(-1) U D U^(-1), U' _C^(-1) U = I, V = _C^(-1/2) U
            eigs = geigen_RA(metact,lambda u: bip.prior.C_act(u,-1,op='K'),lambda u: bip.prior.C_act(u,op='K'),dim=bip.pde.V.dim(),**kwargs)
        if any(s>1.5 for s in geom_ord):
            # adjust the gradient
            # update low-rank approximate Gaussian posterior
            bip.post_Ga = Gaussian_apx_posterior(bip.prior,eigs=eigs)
            Hu = bip.prior.gen_vector()
            bip.post_Ga.Hlr.mult(unknown, Hu)
            gradlik.axpy(1.0,Hu)
    
    if len(kwargs)==0:
        return loglik,gradlik,metact,rtmetact
    else:
        return loglik,gradlik,metact,eigs