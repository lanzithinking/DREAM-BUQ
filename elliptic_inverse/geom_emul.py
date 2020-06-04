"""
Geometric functions by CNN emulation
"""

import numpy as np
import tensorflow as tf
import sys,os
sys.path.append( "../" )
from util.dolfin_gadget import vec2fun,fun2img,img2fun

def geom(unknown,model,cnn,geom_ord=[0],whitened=False):
    loglik=None; gradlik=None;
    
    # un-whiten if necessary
    if whitened:
        unknown=self.prior.v2u(unknown)
    
    u_img=fun2img(vec2fun(unknown,model.pde.V))
    
    ll_f = lambda x: 0.5*model.misfit.prec*tf.math.reduce_sum((cnn.model(x)-model.misfit.obs)**2,axis=1)
    
    loglik = ll_f(u_img[None,:,:,None]).numpy()
    
    if any(s>=1 for s in geom_ord):
        gradlik= img2fun(cnn.gradient(u_img[None,:,:,None], ll_f), model.pde.V)
    
    return loglik,gradlik,None,None