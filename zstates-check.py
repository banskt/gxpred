import numpy as np
import time
import os
import sys
from utils import model
from utils import hyperparameters
from inference import logmarglik
from inference import zstates_old_method as zs_old
from inference import zstates as zs

pi = 0.01
mu = 0.0
sigmabg = 0.001
sigma = 0.1
tau = 1 / (0.0005 * 0.0005)

nsample = 60
nsnps = 250
x, y, csnps, v = model.simulate(nsample, nsnps, 
                                pi = pi,
                                mu = mu,
                                sigma = sigma,
                                sigmabg = sigmabg,
                                tau = tau)
#params = np.array([pi, mu, sigma, sigmabg, tau])
params = np.array([0.01, 0.0, 0.1, 0.1, 1000])
scaledparams = hyperparameters.scale(params)
znew = zs.create(scaledparams, x, y, 2, x.shape[0], 0.98)
print (len(znew))
