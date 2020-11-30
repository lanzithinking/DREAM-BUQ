"""
Geometric functions by emulator emulation
"""

import numpy as np
import dolfin as df
import tensorflow as tf
import sys,os
sys.path.append( "../" )
from util.dolfin_gadget import vec2fun
from util.multivector import *
from util.Eigen import *
from posterior import *

# functions to convert vectors between P1 and Pn
def vinP1(v, V):
    """project v to P1 space"""
    mesh = V.mesh()
    V_P1 = df.FunctionSpace(mesh, V.ufl_element().family(), 1)
    d2v = df.dof_to_vertex_map(V_P1)
    f = df.Function(V)
    f.vector().set_local(v)
#     vec = df.Function(V_P1).vector()
    vec = df.Vector(mesh.mpi_comm(),mesh.num_vertices())
    vec.set_local(f.compute_vertex_values(mesh)[d2v])
    return vec

def vinP(v, V):
    """project v to Pn space"""
    V_P1 = df.FunctionSpace(V.mesh(), V.ufl_element().family(), 1)
    f_P1 = df.Function(V_P1)
    f_P1.vector().set_local(v)
    f = df.Function(V)
    f.interpolate(f_P1)
    return f.vector()

def geom(unknown,bip,emulator,geom_ord=[0],whitened=False,**kwargs):
    loglik=None; gradlik=None; metact=None; rtmetact=None; eigs=None
    
    # un-whiten if necessary
    if whitened:
        unknown=bip.prior.v2u(unknown)
    
    eldeg = bip.prior.V.ufl_element().degree()
    u_input = {'DNN':unknown.get_local()[None,:] if eldeg==1 else vinP1(unknown,bip.prior.V).get_local()[None,:], 'CNN':bip.vec2img(unknown)[None,:,:,None]}[type(emulator).__name__]
    
    ll_f = lambda x: -0.5*tf.math.reduce_sum((emulator.model(x)-bip.misfit.obs)**2/bip.misfit.noise_variance,axis=1)
    
    if any(s>=0 for s in geom_ord):
        loglik = ll_f(u_input).numpy()
    
    if any(s>=1 for s in geom_ord):
        inP1 = kwargs.pop('inP1',False)
        gradlik_ = emulator.gradient(u_input, ll_f)
        if type(emulator).__name__=='DNN':
            gradlik = vec2fun(gradlik_,df.FunctionSpace(bip.prior.V.mesh(), 'Lagrange', 1)).vector() if eldeg==1 or inP1 else vinP(gradlik_, bip.prior.V)
        elif type(emulator).__name__=='CNN':
            gradlik = bip.img2vec(gradlik_, bip.prior.V if eldeg>1 and not inP1 else None)
        if whitened:
            gradlik = bip.prior.C_act(gradlik,.5)
    
    if any(s>=1.5 for s in geom_ord):
        jac_ = emulator.jacobian(u_input)
        n_obs = len(bip.misfit.obs)
        jac = MultiVector(unknown,n_obs)
        [jac[i].set_local({'DNN':jac_[i] if eldeg==1 else vinP(jac_[i],bip.prior.V),'CNN':bip.img2vec(jac_[i], bip.prior.V if eldeg>1 else None)}[type(emulator).__name__]) for i in range(n_obs)]
        def _get_metact_misfit(u_actedon): # GNH
            if type(u_actedon) is not df.Vector:
                u_actedon = bip.prior.gen_vector(u_actedon)
            v = bip.prior.gen_vector()
            jac.reduce(v,jac.dot(u_actedon)/bip.misfit.noise_variance)
            return bip.prior.M*v
        def _get_rtmetact_misfit(u_actedon):
            if type(u_actedon) is df.Vector:
                u_actedon = u_actedon.get_local()
            v = bip.prior.gen_vector()
            jac.reduce(v,u/np.sqrt(bip.misfit.noise_variance))
            return bip.prior.rtM*v
        metact = _get_metact_misfit
        rtmetact = _get_rtmetact_misfit
        if whitened:
            metact = lambda u: bip.prior.C_act(_get_metact_misfit(bip.prior.C_act(u,.5)),.5) # ppGNH
            rtmetact = lambda u: bip.prior.C_act(_get_rtmetact_misfit(u),.5)
    
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
            eigs = geigen_RA(metact,lambda u: bip.prior.C_act(u,-1),lambda u: bip.prior.C_act(u),dim=bip.pde.V.dim(),**kwargs)
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