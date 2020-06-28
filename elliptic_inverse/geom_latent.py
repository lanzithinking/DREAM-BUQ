"""
Geometric functions of latent variables, which are encoder outputs
"""

import numpy as np
import dolfin as df
import sys,os
from logging import raiseExceptions
sys.path.append( "../" )
# from util.dolfin_gadget import vec2fun,fun2img,img2fun
# from util.multivector import *
from util.Eigen import *
from posterior import *

def geom(unknown_lat,bip,ae,geom_ord=[0],whitened=False,**kwargs):
    loglik=None; gradlik=None; metact=None; rtmetact=None; eigs=None
    
#     # un-whiten if necessary
#     if whitened:
#         unknown_lat=bip_lat.prior.v2u(unknown_lat)
    
    unkown=bip.prior.gen_vector(ae.decode(unknown_lat.get_local()[None,:]).flatten())
    bip_lat=kwargs.pop('bip_lat',None)
    
    if len(kwargs)==0:
        loglik,gradlik,metact_,rtmetact_ = bip.get_geom(unkown,geom_ord,whitened)
    else:
        loglik,gradlik,metact_,eigs_ = bip.get_geom(unkown,geom_ord,whitened,**kwargs)
    
    if any(s>=1 for s in geom_ord):
        jac=ae.jacobian(unknown_lat.get_local()[None,:],'decode')
        gradlik_=jac.T.dot(gradlik.get_local())
        gradlik=df.Vector(unknown_lat)
        gradlik.set_local(gradlik_)
    
    if any(s>=1.5 for s in geom_ord):
        def _get_metact_misfit(u_actedon):
            if type(u_actedon) is df.Vector:
                u_actedon=u_actedon.get_local()
            v=df.Vector(unknown_lat)
            v.set_local(jac.T.dot(metact_(jac.dot(u_actedon)).get_local()))
            return v
        def _get_rtmetact_misfit(u_actedon):
            if type(u_actedon) is not df.Vector:
                u_actedon=bip.prior.gen_vector(u_actedon)
            v=df.Vector(unknown_lat)
            v.set_local(jac.T.dot(rtmetact_(u_actedon).get_local()))
            return v
        metact = _get_metact_misfit
        rtmetact = _get_rtmetact_misfit
    
    if any(s>1 for s in geom_ord) and len(kwargs)!=0:
        if bip_lat is None: raise ValueError('No latent inverse problem defined!')
        # compute eigen-decomposition using randomized algorithms
        if whitened:
            # generalized eigen-decomposition (_C^(1/2) F _C^(1/2), M), i.e. _C^(1/2) F _C^(1/2) = M V D V', V' M V = I
            def invM(a):
                a=bip_lat.prior.gen_vector(a)
                invMa=bip_lat.prior.gen_vector()
                bip_lat.prior.Msolver.solve(invMa,a)
                return invMa
            eigs = geigen_RA(metact, lambda u: bip_lat.prior.M*u, invM, dim=bip_lat.pde.V.dim(),**kwargs)
        else:
            # generalized eigen-decomposition (F, _C^(-1)), i.e. F = _C^(-1) U D U^(-1), U' _C^(-1) U = I, V = _C^(-1/2) U
            eigs = geigen_RA(metact,lambda u: bip_lat.prior.C_act(u,-1,op='K'),lambda u: bip_lat.prior.C_act(u,op='K'),dim=bip_lat.pde.V.dim(),**kwargs)
        if any(s>1.5 for s in geom_ord):
            # adjust the gradient
            # update low-rank approximate Gaussian posterior
            bip_lat.post_Ga = Gaussian_apx_posterior(bip_lat.prior,eigs=eigs)
#             Hu = bip_lat.prior.gen_vector()
#             bip_lat.post_Ga.Hlr.mult(unknown, Hu)
#             gradlik.axpy(1.0,Hu)
    
    if len(kwargs)==0:
        return loglik,gradlik,metact,rtmetact
    else:
        return loglik,gradlik,metact,eigs