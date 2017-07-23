#!/usr/bin/env python

import numpy as np
import time
import os
import ctypes

import time
from utils import model
from inference import log_marginal_likelihood
from inference import zstates as zs
from inference import zstates_old_method as zs_old
from utils import hyperparameters

_path = os.path.dirname(__file__)
lib = np.ctypeslib.load_library('lib/margloglik.so', _path)
czcomps = lib.logmarglik
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
                    np.ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS, ALIGNED'),
                    np.ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS, ALIGNED'),
                    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED')
                   ]


pi = 0.1
mu = 0.001
sigmabg = 0.001
sigma = 0.1
tau = 1 / (0.005 * 0.005)

x, y, csnps, v = model.simulate(pi = pi,
                             mu = mu,
                             sigma = sigma,
                             sigmabg = sigmabg,
                             tau = tau)

nvar = x.shape[0]
nsample = x.shape[1]
params = np.array([pi, mu, sigma, sigmabg, tau])
scaledparams = hyperparameters.scale(params)
cmax = 2
zstates_new = zs.create(scaledparams, x, y, cmax, nvar, 0.98)
zstates_old = zs_old.create(scaledparams, x, y, cmax, nvar, 0.98)
zstates = zstates_old

params = np.array([0.00495222, 0, 0.48975, 0.0680367, 1 / 0.0001 / 0.0001])
scaledparams = hyperparameters.scale(params)

start_time = time.time()
m, der = log_marginal_likelihood.func_grad(scaledparams, x, y, zstates)
print ("Log marginal likelihood from python calculation: {:f}".format(m))
print ("Derivatives: {:f} {:f} {:f} {:f} {:f}".format(der[0], der[1], der[2], der[3], der[4]))
print("Calculated in {:f} seconds ---\n".format(time.time() - start_time))

start_time = time.time()
zlen = len(zstates)
zarr = np.array([item for sublist in zstates for item in sublist], dtype=np.int32)
znorm = np.array([len(sublist) for sublist in zstates], dtype=np.int32)

zcomps = np.zeros(zlen)
grad = np.zeros(5)

logmarglik = czcomps(nvar, nsample, zlen, params[0], params[1], params[2], params[3], params[4],
                     True, zarr, znorm, x.reshape(-1,), y.reshape(-1,), zcomps, grad)
grad = hyperparameters.gradscale(params, grad)
print ("Log marginal likelihood from C calculation: {:f}".format(logmarglik))
print ("Derivatives: {:f} {:f} {:f} {:f} {:f}".format(grad[0], grad[1], grad[2], grad[3], grad[4]))
print("Calculated in {:f} seconds ---\n".format(time.time() - start_time))
