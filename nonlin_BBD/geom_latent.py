"""
Geometric functions of latent variables, which are encoder outputs,
with geometric quantities emulated or extracted from emulator
------------------------------------------------------------------
Shiwei Lan @ ASU, August 2020
"""

import numpy as np

# functions needed to make even image size
def pad(A,width=[1]):
    shape=A.shape
    if len(width)==1: width=np.tile(width,len(shape))
    if not any(width): return A
    assert len(width)==len(shape), 'non-matching padding width!'
    pad_width=tuple((0,i) for i in width)
    return np.pad(A, pad_width)
def chop(A,width=[1]):
    shape=A.shape
    if len(width)==1: width=np.tile(width,len(shape))
    if not any(width): return A
    assert len(width)==len(shape), 'non-matching chopping width!'
    chop_slice=tuple(slice(0,-i) for i in width)
    return A[chop_slice]

def vec2img(v):
    gdim=2
    imsz = np.floor(v.size**(1./gdim)).astype('int')
    im_shape=(-1,)+(imsz,)*(gdim-1)
    return v.reshape(im_shape)

def geom(unknown_lat,autoencoder,geom_ord=[0],whitened=False,**kwargs):
    loglik=None; gradlik=None; metact=None; rtmetact=None; eigs=None
    
    # un-whiten if necessary
    if whitened=='latent':
        bip_lat=kwargs.get('bip_lat')
        unknown_lat=bip_lat.prior['cov'].dot(unknown_lat)
    
#     u_latin={'AutoEncoder':unknown_lat[None,:],'ConvAutoEncoder':chop(vec2img(unknown_lat))[None,:,:,None]}[type(autoencoder).__name__]
    if 'Conv' in type(autoencoder).__name__:
        u_latin=vec2img(unknown_lat)
        width=tuple(np.mod(i,2) for i in u_latin.shape)
        u_latin=chop(u_latin,width)[None,:,:,None] if autoencoder.activations['latent'] is None else u_latin.flatten()[None,:]
        unknown=img2vec(pad(np.squeeze(autoencoder.decode(u_latin)),width))
    else:
        u_latin=unknown_lat[None,:]
        unknown=autoencoder.decode(u_latin).flatten()
    
    emul_geom=kwargs.pop('emul_geom',None)
    full_geom=kwargs.pop('full_geom',None)
    bip_lat=kwargs.pop('bip_lat',None)
    bip=kwargs.pop('bip',None)
    try:
        if len(kwargs)==0:
            loglik,gradlik,metact_,rtmetact_ = emul_geom(unknown,geom_ord,whitened=='emulated')
        else:
            loglik,gradlik,metact_,eigs_ = emul_geom(unknown,geom_ord,whitened=='emulated',**kwargs)
    except:
        try:
            if len(kwargs)==0:
                loglik,gradlik,metact_,rtmetact_ = full_geom(unknown,geom_ord,whitened=='original')
            else:
                loglik,gradlik,metact_,eigs_ = full_geom(unknown,geom_ord,whitened=='original',**kwargs)
        except:
            raise RuntimeError('No geometry in the original space available!')
    
    if any(s>=1 for s in geom_ord):
        if whitened=='latent':
            cholC = cholC = np.linalg.cholesky(bip.prior['cov'])
            gradlik = cholC.T.dot(gradlik)
#         jac=autoencoder.jacobian(u_latin,'decode')
        if 'Conv' in type(autoencoder).__name__:
            jac=autoencoder.jacobian(u_latin,'decode')
            jac=pad(jac,width*2 if autoencoder.activations['latent'] is None else width+(0,))
            jac=jac.reshape((np.prod(jac.shape[:2]),np.prod(jac.shape[2:])))
#         gradlik=jac.T.dot(gradlik)
        gradlik=autoencoder.jacvec(u_latin,gradlik[None,:])
#         print('time consumed:{}'.format(timeit.default_timer()-t_start))
    
    if any(s>=1.5 for s in geom_ord):
        _get_metact_misfit=lambda u_actedon: jac.dot(metact_(jac.T.dot(u_actedon)))
        _get_rtmetact_misfit=lambda u_actedon: jac.dot(rtmetact_(u_actedon))
        metact = _get_metact_misfit
        rtmetact = _get_rtmetact_misfit
    
    if any(s>1 for s in geom_ord) and len(kwargs)!=0:
        if bip_lat is None: raise ValueError('No latent inverse problem defined!')
        # compute eigen-decomposition using randomized algorithms
        if whitened=='latent':
            # generalized eigen-decomposition (_C^(1/2) F _C^(1/2), M), i.e. _C^(1/2) F _C^(1/2) = M V D V', V' M V = I
            def invM(a):
                a=bip_lat.prior.gen_vector(a)
                invMa=bip_lat.prior.gen_vector()
                bip_lat.prior.Msolver.solve(invMa,a)
                return invMa
            eigs = geigen_RA(metact, lambda u: u, lambda u: u, dim=bip_lat.input_dim,**kwargs)
        else:
            # generalized eigen-decomposition (F, _C^(-1)), i.e. F = _C^(-1) U D U^(-1), U' _C^(-1) U = I, V = _C^(-1/2) U
            eigs = geigen_RA(metact,lambda u: np.linalg.solve(bip_lat.prior['cov'],u),lambda u: bip_lat.prior['cov'].dot(u),dim=bip_lat.input_dim,**kwargs)
        if any(s>1.5 for s in geom_ord):
            # adjust the gradient
            # update low-rank approximate Gaussian posterior
            bip_lat.post_Ga = Gaussian_apx_posterior(bip_lat.prior,eigs=eigs)
#             Hu= bip.post_Ga['Hlr'].dot(unknown)
#             gradlik+=Hu
    
    if len(kwargs)==0:
        return loglik,gradlik,metact,rtmetact
    else:
        return loglik,gradlik,metact,eigs