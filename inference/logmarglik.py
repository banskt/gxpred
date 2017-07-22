#!/usr/bin/env python

import numpy as np
import os
import ctypes

def czcompgrad(pi, mu, sig2, z, v, nhyp, feat, mureg, reg2, prec, get_gradient=True, is_covariate=False):
    ''' An efficient C code for calculating marginal log likelihood and its gradient if requested.
        This is the only interface with margloglik.so
        Inputs:
              pi, mu, sigma, sigmabg, tau: hyperparameters
              z: all the zstates // could be empty for the 0 z-state
              nhyp: number of hyperparameters which are being optimized
              nsample: number of samples
              x: vector of explanatory variables, flattened I x N matrix
              y: vector of target variables, size N
              get_gradient: boolean, if false gradient calculation is skipped. Default: True
         Returns:
              zcomps: P(z)*F(z) for each zstate
              grad: gradient of each hyperparameter. Returns a zeros array if get_gradient is false
    '''

    _path = os.path.dirname(__file__)
    lib = np.ctypeslib.load_library('../lib/margloglik.so', _path)
    ccomps = lib.margloglik_zcomps
    ccomps.restype = ctypes.c_int
    ccomps.argtypes = [ctypes.c_int,                                                                   # nvar, number of SNPs
                       ctypes.c_int,                                                                   # nhyp, number of hyperparameters
                       ctypes.c_int,                                                                   # nfeat, number of features
                       ctypes.c_int,                                                                   # zlen
                       ctypes.c_int,                                                                   # max(znorm)
                       np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'), # pi
                       np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'), # mu
                       np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'), # sig2
                       np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'), # v^tilde
                       np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'), # Lambda^tilde
                       np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'), # Feature matrix
                       np.ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS, ALIGNED'),            # zarr
                       np.ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS, ALIGNED'),            # znorm
                       ctypes.c_double,                                                                # sigreg2
                       ctypes.c_double,                                                                # mureg
                       np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'), # zcomps
                       np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'), # grad
                       ctypes.c_bool,                                                                  # boolean to get the gradient
                       ctypes.c_bool                                                                   # boolean for the covariate
                      ]

