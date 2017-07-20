#!/usr/bin/env python

import numpy as np
import time
import argparse
from utils import model
from utils import hyperparameters
from inference import log_marginal_likelihood
from inference import zstates_old_method as zs

def parse_args():

    parser = argparse.ArgumentParser(description='Check optimization')

    parser.add_argument('--params',
                        nargs='*',
                        type=float,
                        dest='params',
                        metavar='FLOAT',
                        help='initialization parameters')
    opts = parser.parse_args()
    return opts

opts = parse_args()

pi = 0.1
mu = 0.0
sigmabg = 0.001
sigma = 0.03
sigerr = 0.05
tau = 1 / sigerr / sigerr

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
target = 0.98
zstates = zs.create(scaledparams, x, y, cmax, nvar, target)

#zstates = [[]]
#zstates += [[i] for i in range(nvar)]

#start_time = time.time()
#l, m = log_marginal_likelihood.full(params, x, y, zstates)
#print ("Log marginal likelihood from full calculation: {:f}".format(m))
#print("Calculated in {:f} seconds ---\n".format(time.time() - start_time))
#
#start_time = time.time()
#m, der = log_marginal_likelihood.iterative_inverse(params, x, y, zstates)
#print ("Log marginal likelihood from fast calculation: {:f}".format(-m))
#print("Calculated in {:f} seconds ---\n".format(time.time() - start_time))
#
from scipy import optimize
#init_params = np.array([0.005, 0.0, 0.1, 0.1, 1 / 0.5 / 0.5])
init_params = np.array(opts.params)
scaledparams = hyperparameters.scale(init_params)

bounds = [[scaledparams[i], scaledparams[i]] for i in range(5)]
bounds[0] = [None, None]
#bounds[1] = [None, None]
bounds[2] = [None, None]
bounds[3] = [None, None]
bounds[4] = [None, None]

args = x, y, zstates
lml_min = optimize.minimize(log_marginal_likelihood.func_grad,
                            scaledparams,
                            args = args,
                            method='L-BFGS-B',
                            jac=True,
                            bounds=bounds,
                            options={'maxiter': 200000,
                                     'maxfun': 2000000,
                                     'ftol': 1e-9,
                                     'gtol': 1e-9,
                                     'disp': True})

print(lml_min)

res = hyperparameters.descale(lml_min.x)
res[4] = np.sqrt(1 / res[4])
print('\n'.join(['{:g}'.format(x) for x in list(res)]))
