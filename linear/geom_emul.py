"""
Geometric functions by emulator emulation
"""

import numpy as np
import tensorflow as tf

def vec2img(v):
    gdim=2
    imsz = np.floor(v.size**(1./gdim)).astype('int')
    im_shape=(-1,)+(imsz,)*(gdim-1)
    return v.reshape(im_shape)

def geom(unknown,bip,emulator,geom_ord=[0],whitened=False,**kwargs):
    loglik=None; gradlik=None; metact=None; rtmetact=None; eigs=None
    
    # un-whiten if necessary
    if whitened:
        unknown=bip.prior['cov'].dot(unknown)
    
    u_input = {'DNN':unknown[None,:], 'CNN':vec2img(unknown)[None,:,:,None]}[type(emulator).__name__]
    
    ll_f = lambda x: -0.5*tf.math.reduce_sum((emulator.model(x)-bip.y[None,:])**2/bip.nz_var[None,:],axis=1)
    
    if any(s>=0 for s in geom_ord):
        loglik = ll_f(u_input).numpy()
    
    if any(s>=1 for s in geom_ord):
        gradlik = emulator.gradient(u_input, ll_f)
        if type(emulator).__name__=='CNN':
            gradlik = img2vec(gradlik)
        if whitened:
            cholC = np.linalg.cholesky(bip.prior['cov'])
            gradlik = cholC.T.dot(gradlik)
    
    if any(s>=1.5 for s in geom_ord):
        jac = emulator.jacobian(u_input)
        if type(emulator).__name__=='CNN':
            jac = jac.reshape((jac.shape[0],-1))
        _get_metact_misfit=lambda u_actedon: jac.T.dot(jac.dot(u_actedon)/bip.nz_var) # GNH
        _get_rtmetact_misfit=lambda u_actedon: jac.T.dot(u_actedon/np.sqrt(bip.nz_var))
        metact = _get_metact_misfit
        rtmetact = _get_rtmetact_misfit
        if whitened:
            metact = lambda u: cholC.T.dot(_get_metact_misfit(cholC.dot(u))) # ppGNH
            rtmetact = lambda u: cholC.T.dot(_get_rtmetact_misfit(u))
    
    if any(s>1 for s in geom_ord) and len(kwargs)!=0:
        if whitened:
            # generalized eigen-decomposition (_C^(1/2) F _C^(1/2), M), i.e. _C^(1/2) F _C^(1/2) = M V D V', V' M V = I
            eigs = geigen_RA(metact, lambda u: u, lambda u: u, dim=bip.input_dim,**kwargs)
        else:
            # generalized eigen-decomposition (F, _C^(-1)), i.e. F = _C^(-1) U D U^(-1), U' _C^(-1) U = I, V = _C^(-1/2) U
            eigs = geigen_RA(metact,lambda u: np.linalg.solve(bip.prior['cov'],u),lambda u: bip.prior['cov'].dot(u),dim=bip.input_dim,**kwargs)
        if any(s>1.5 for s in geom_ord):
            # adjust the gradient
            # update low-rank approximate Gaussian posterior
            bip.post_Ga = Gaussian_apx_posterior(bip.prior,eigs=eigs)
            Hu= bip.post_Ga['Hlr'].dot(unknown)
            gradlik+=Hu
    
    if len(kwargs)==0:
        return loglik,gradlik,metact,rtmetact
    else:
        return loglik,gradlik,metact,eigs