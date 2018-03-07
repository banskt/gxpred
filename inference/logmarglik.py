#!/usr/bin/env python

import numpy as np
import os
import ctypes
from utils import hyperparameters

def czcompgrad(params, x, y, zstates, get_grad=True, get_exp=False, prior="gxpred-mg"):
    ''' Interface with margloglik.so,
        an efficient C code for calculating log marginal likelihood and gradient.
    '''
    _path = os.path.dirname(__file__)
    # print("czcompgrad using prior: {:s}".format(prior))
    if prior == "gxpred-mg":
        clib = np.ctypeslib.load_library('../lib/logmarglik.so', _path)
    if prior == "gxpred-mg-old":
        clib = np.ctypeslib.load_library('../lib/logmarglik_original.so', _path)
    if prior == "gxpred-bslmm":
        clib = np.ctypeslib.load_library('../lib/logmarglik_bslmm.so', _path)
    double_pointer = ctypes.POINTER(ctypes.c_double)

    czcomps = clib.logmarglik
    czcomps.restype = ctypes.c_bool

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
    logmarglik = np.zeros(1)

    
    success = czcomps(nvar, nsample, zlen, params[0], params[1], params[2], params[3], params[4],
                         get_grad, get_exp, zarr, znorm, x.reshape(-1,), y.reshape(-1,), zcomps, grad, zexp, logmarglik)
    if not (success):
        print ("Python Error: C library could not compute z-components. Check C errors above.")
    
    return success, logmarglik[0], zcomps, grad, zexp


def add_prior(param, prior, priorparams):
    if not priorparams:
        raise Exception("Prior parameters are empty")
    if prior == "L1":
        lambd = priorparams["lambda"]
        lml_term = -np.log(lambd) - np.abs(param)/lambd
        grad_term = np.sign(param)/lambd
    if prior == "S2":
        a = priorparams["alpha"]
        lml_term = -( params[3]**2 / (2 * a**2) ) - np.log(a) - 0.5 * np.log(2*np.pi)
        grad_term = params[3] / (a**2)
    return lml_term, grad_term

def func_grad(scaledparams, x, y, zstates, prior, hyperpriors, hyperparams):
    params = hyperparameters.descale(scaledparams)
    success, logmarglik, zprob, grad, zexp = czcompgrad(params, x, y, zstates, get_grad=True, prior=prior)
    
    for i, hp in enumerate(hyperpriors):
        if hp:
            lml_term, grad_term = add_prior(params[i], hp, hyperparams)
            logmarglik += lml_term
            grad[i] -= grad_term

    ### S2 prior
    # a = 0.01
    # logmarglik += -( params[3]**2 / (2 * a**2) ) - np.log(a) - 0.5 * np.log(2*np.pi)
    # grad[3] -= params[3] / (a**2)

    ### L1 prior (1/lambda)*exp(-abs(sigmabg)/lambda)
    # lambd = 0.05
    # logmarglik += -np.log(lambd) - np.abs(params[3])/lambd
    # grad[3] -= np.sign(params[3])/lambd

    grad = hyperparameters.gradscale(params, grad)
    return success, -logmarglik, -grad

def func(scaledparams, x, y, zstates, prior, hyperpriors, hyperparams):
    params = hyperparameters.descale(scaledparams)
    success, logmarglik, zprob, grad, zexp = czcompgrad(params, x, y, zstates, get_grad=False, prior=prior)
    
    for i, hp in enumerate(hyperpriors):
        if hp:
            lml_term, grad_term = add_prior(params[i], hp, hyperparams)
            logmarglik += lml_term

    ### S2 prior
    # a = 0.01
    # logmarglik += -( params[3]**2 / (2 * a**2) ) - np.log(a) - 0.5 * np.log(2*np.pi)

    ### L1 prior (1/lambda)*exp(-abs(sigmabg)/lambda)
    # lambd = 0.05
    # logmarglik += -np.log(lambd) - np.abs(params[3])/lambd
    return -logmarglik

def prob_comps(scaledparams, x, y, zstates, prior):
    params = hyperparameters.descale(scaledparams)
    success, logmarglik, zprob, grad, zexp = czcompgrad(params, x, y, zstates, get_grad=False, prior=prior)
    return zprob

def model_exp(scaledparams, x, y, zstates, prior):
    params = hyperparameters.descale(scaledparams)
    success, logmarglik, zprob, grad, zexp = czcompgrad(params, x, y, zstates, get_grad=False, get_exp=True, prior=prior)
    zexp = zexp.reshape(len(zstates), x.shape[0])
    return zprob, zexp
