#!/usr/bin/env python

import numpy as np
import os
import ctypes
from utils import hyperparameters

def czcompgrad(params, x, y, zstates, get_grad=True, get_exp=False):
    ''' Interface with margloglik.so,
        an efficient C code for calculating log marginal likelihood and gradient.
    '''
    _path = os.path.dirname(__file__)
    clib = np.ctypeslib.load_library('../lib/logmarglik.so', _path)

    czcomps = clib.logmarglik
    czcomps.restype = ctypes.c_double
    czcomps.argtypes = [ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_double,
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
                        np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED')
                       ]
    
    nvar = x.shape[0]
    nsample = x.shape[1]
    
    zlen = len(zstates)
    zarr = np.array([item for sublist in zstates for item in sublist], dtype=np.int32)
    znorm = np.array([len(sublist) for sublist in zstates], dtype=np.int32)
    
    zcomps = np.zeros(zlen)
    grad = np.zeros(5)
    zexp = np.zeros(zlen * nvar)
    
    logmarglik = czcomps(nvar, nsample, zlen, params[0], params[1], params[2], params[3], params[4],
                         get_grad, get_exp, zarr, znorm, x.reshape(-1,), y.reshape(-1,), zcomps, grad, zexp)
    
    return logmarglik, zcomps, grad, zexp

def func_grad(scaledparams, x, y, zstates):
    params = hyperparameters.descale(scaledparams)
    logmarglik, zprob, grad, zexp = czcompgrad(params, x, y, zstates, get_grad=True)
    grad = hyperparameters.gradscale(params, grad)
    return -logmarglik, -grad

def func(scaledparams, x, y, zstates):
    params = hyperparameters.descale(scaledparams)
    logmarglik, zprob, grad, zexp = czcompgrad(params, x, y, zstates, get_grad=False)
    return -logmarglik

def prob_comps(scaledparams, x, y, zstates):
    params = hyperparameters.descale(scaledparams)
    logmarglik, zprob, grad, zexp = czcompgrad(params, x, y, zstates, get_grad=False)
    return zprob

def model_exp(scaledparams, x, y, zstates):
    params = hyperparameters.descale(scaledparams)
    logmarglik, zprob, grad, zexp = czcompgrad(params, x, y, zstates, get_grad=False, get_exp=True)
    zexp = zexp.reshape(len(zstates), x.shape[0])
    return zprob, zexp
