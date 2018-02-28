#!/usr/bin/env python

import numpy as np
import os
import ctypes
from utils import hyperparameters

def czcompgrad(params, x, y, features, zstates, get_grad=True, get_exp=False):
    ''' Interface with margloglik.so,
        an efficient C code for calculating log marginal likelihood and gradient.
    '''
    _path = os.path.dirname(__file__)
    # clib = np.ctypeslib.load_library('../lib/logmarglik.so', _path)
    clib = np.ctypeslib.load_library('../lib/logmarglik_bslmm.so', _path)
    double_pointer = ctypes.POINTER(ctypes.c_double)

    czcomps = clib.logmarglik
    czcomps.restype = ctypes.c_bool
    czcomps.argtypes = [ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_bool,
                        ctypes.c_bool,
                        np.ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS, ALIGNED'),
                        np.ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS, ALIGNED'),
                        np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                        np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                        np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                        np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                        np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                        np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                        np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED')
                       ]

    nsnps = x.shape[0]
    nsample = x.shape[1]
    nfeat = features.shape[1]
    nhyp = nfeat + 4
    pi_arr = 1 / (1 + np.exp(-np.einsum('k, ik -> i', params[:nfeat], features)))
    mu =      params[ nfeat + 0 ]
    sigma =   params[ nfeat + 1 ]
    sigmabg = params[ nfeat + 2 ]
    tau =     params[ nfeat + 3 ]
    print(params[0], mu, sigma, sigmabg, tau)
    
    zlen = len(zstates)
    zarr = np.array([item for sublist in zstates for item in sublist], dtype=np.int32)
    znorm = np.array([len(sublist) for sublist in zstates], dtype=np.int32)
    
    zcomps = np.zeros(zlen)
    grad = np.zeros( nhyp )
    zexp = np.zeros(zlen * nsnps)
    logmarglik = np.zeros(1)
    
    success = czcomps(nsnps, nsample, zlen, nfeat, pi_arr, mu, sigma, sigmabg, tau,
                         get_grad, get_exp, zarr, znorm, x.reshape(-1,), y.reshape(-1,), features.reshape(-1,), zcomps, grad, zexp, logmarglik)
    if not (success):
        print ("Python Error: C library could not compute z-components. Check C errors above.")
    
    return success, logmarglik[0], zcomps, grad, zexp

def func_grad(scaledparams, x, y, features, zstates):
    params = hyperparameters.descale(scaledparams)
    success, logmarglik, zprob, grad, zexp = czcompgrad(params, x, y, features, zstates, get_grad=True)
    grad = hyperparameters.gradscale(params, grad)
    return success, -logmarglik, -grad

def func(scaledparams, x, y, features, zstates):
    params = hyperparameters.descale(scaledparams)
    success, logmarglik, zprob, grad, zexp = czcompgrad(params, x, y, features, zstates, get_grad=False)
    return -logmarglik

def prob_comps(scaledparams, x, y, features, zstates):
    params = hyperparameters.descale(scaledparams)
    success, logmarglik, zprob, grad, zexp = czcompgrad(params, x, y, features, zstates, get_grad=False)
    return zprob

def model_exp(scaledparams, x, y, features, zstates):
    params = hyperparameters.descale(scaledparams)
    success, logmarglik, zprob, grad, zexp = czcompgrad(params, x, y, features, zstates, get_grad=False, get_exp=True)
    zexp = zexp.reshape(len(zstates), x.shape[0])
    return zprob, zexp
