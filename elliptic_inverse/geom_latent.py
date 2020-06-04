"""
Geometric functions of latent variables, which are encoder outputs
"""

import numpy as np

def geom(latent_u,model,ae,geom_ord=[0],whitened=False):
    
    unkown=model.prior.gen_vector(ae.decode(latent_u.reshape(1,latent_u.size)).flatten())
    
    loglik,gradlik,_,_ = model.get_geom(unkown,geom_ord,whitened)
    
    if any(s>=1 for s in geom_ord):
        gradlik=ae.jacobian(latent_u,'decode').T.dot(gradlik.get_local())
    
    return loglik,gradlik,None,None