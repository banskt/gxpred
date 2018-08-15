#!/usr/bin/env python

import numpy as np
import os
import ctypes
from utils import hyperparameters

def czcompgrad(params, x, y, features, dist_feature, zstates, get_grad=True, get_exp=False):
    ''' Interface with margloglik.so,
        an efficient C code for calculating log marginal likelihood and gradient.
    '''
    _path = os.path.dirname(__file__)
    # clib = np.ctypeslib.load_library('../lib/logmarglik.so', _path)
    # clib = np.ctypeslib.load_library('../lib/logmarglik_bslmm.so', _path)
    # clib = np.ctypeslib.load_library('../lib/logmarglik_bslmm_fixedPI.so', _path)
    # clib = np.ctypeslib.load_library('../lib/logmarglik_bslmm_fixedPI.bak.so', _path)
    # clib = np.ctypeslib.load_library('../lib/logmarglik_bslmm_fixing_sigmas.so', _path)
    clib = np.ctypeslib.load_library('../lib/logmarglik_point_normal.so', _path)
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

    # this can overflow
    # pi_arr2 = 1 / (1 + dist_feature * np.exp(-np.einsum('k, ik -> i', params[:nfeat], features)))

    gT_A = np.einsum('k, ik -> i', params[:nfeat], features)
    ix_pos = gT_A > 0 
    ix_neg = gT_A <= 0 
    pi_arr = np.zeros(gT_A.shape[0])
    # pi_arr = 1 / (1 + dist_feature*np.exp(-gT_A))
    # pi_arr[ix_pos] = 1 / (1 + np.exp(-gT_A[ix_pos]))
    pi_arr[ix_pos] = 1 / (1 + dist_feature[ix_pos]*np.exp(-gT_A[ix_pos]))
    # pi_arr[ix_neg] = np.exp(gT_A[ix_neg]) / (1 + np.exp(gT_A[ix_neg]))
    pi_arr[ix_neg] = np.exp(gT_A[ix_neg]) / (dist_feature[ix_neg] + np.exp(gT_A[ix_neg]))

    mu =      params[ nfeat + 0 ]
    sigma =   params[ nfeat + 1 ]
    sigmabg = params[ nfeat + 2 ]
    tau =     params[ nfeat + 3 ]
    # print("Before czcomps:", params[:nfeat], mu, sigma, sigmabg, tau)
    
    zlen = len(zstates)
    zarr = np.array([item for sublist in zstates for item in sublist], dtype=np.int32)
    znorm = np.array([len(sublist) for sublist in zstates], dtype=np.int32)
    
    zcomps = np.zeros(zlen)         # P(z|x,y,theta,tau) Eq. 3.25
    grad = np.zeros( nhyp )
    zexp = np.zeros(zlen * nsnps)   # m_vz used in Eq. 4.3, defined in Eq. 3.22
    logmarglik = np.zeros(1)
    
    success = czcomps(nsnps, nsample, zlen, nfeat, pi_arr, mu, sigma, sigmabg, tau,
                         get_grad, get_exp, zarr, znorm, x.reshape(-1,), y.reshape(-1,), features.reshape(-1,), zcomps, grad, zexp, logmarglik)
    if not (success):
        print ("Python Error: C library could not compute z-components. Check C errors above.")
    
    return success, logmarglik[0], zcomps, grad, zexp

def add_prior(params):
    lambd = 0.1
    lml_term = -np.log(lambd) - np.abs(params[1])/lambd
    grad_term = -np.sign(params[1])/lambd
    # logmarglik += lml_term
    # grad[1] += grad_term
    return lml_term, grad_term
    

def func_grad(scaledparams, x, y, features, dist_feature, zstates):
    params = hyperparameters.descale(scaledparams)
    success, logmarglik, zprob, grad, zexp = czcompgrad(params, x, y, features, dist_feature, zstates)
    grad = hyperparameters.gradscale(params, grad)
    return success, -logmarglik, -grad

def func(scaledparams, x, y, features, dist_feature, zstates):
    params = hyperparameters.descale(scaledparams)
    success, logmarglik, zprob, grad, zexp = czcompgrad(params, x, y, features, dist_feature, zstates, get_grad=False)
    return -logmarglik

def prob_comps(scaledparams, x, y, features, dist_feature, zstates):
    params = hyperparameters.descale(scaledparams)
    success, logmarglik, zprob, grad, zexp = czcompgrad(params, x, y, features, dist_feature, zstates, get_grad=False)
    return zprob

def model_exp(scaledparams, x, y, features, dist_feature, zstates):
    params = hyperparameters.descale(scaledparams)
    success, logmarglik, zprob, grad, zexp = czcompgrad(params, x, y, features, dist_feature, zstates, get_grad=False, get_exp=True)
    zexp = zexp.reshape(len(zstates), x.shape[0])
    return zprob, zexp
