#!/usr/bin/env python

import numpy as np
import time
import argparse
from scipy import optimize

from utils import model
from utils import hyperparameters
from inference import logmarglik
from inference import log_marginal_likelihood
from inference import zstates_old_method as zs
from inference.empirical_bayes import EmpiricalBayes

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

pi = 0.01
mu = 0.0
sigmabg = 0.001
sigma = 0.1
sigerr = 0.005
tau = 1 / sigerr / sigerr

x, y, csnps, v = model.simulate(pi = pi,
                                mu = mu,
                                sigma = sigma,
                                sigmabg = sigmabg,
                                tau = tau)

init_params = np.array(opts.params)

#emp_bayes = EmpiricalBayes(x, y, 2, init_params, method="old")
#emp_bayes.fit()
#res = emp_bayes.params
#res[4] = np.sqrt(1 / res[4])
#print('\n'.join(['{:g}'.format(x) for x in list(res)]))

emp_bayes = EmpiricalBayes(x, y, 5, init_params, method="new")
emp_bayes.fit()
res = emp_bayes.params
res[4] = np.sqrt(1 / res[4])
print('\n'.join(['{:g}'.format(x) for x in list(res)]))


# ======================== OLD ===========================
#zstates = [[]]
#zstates += [[i] for i in range(nvar)]
#nvar = x.shape[0]
#nsample = x.shape[1]
#params = np.array([pi, mu, sigma, sigmabg, tau])
#scaledparams = hyperparameters.scale(init_params)
#
#cmax = 1
#target = 0.98
#zstates = zs.create(scaledparams, x, y, cmax, nvar, target)
#print ("{:d} zstates created".format(len(zstates)))
#
#start_time = time.time()
#l, m = log_marginal_likelihood.full(params, x, y, zstates)
#print ("Log marginal likelihood from full calculation: {:f}".format(m))
#print("Calculated in {:f} seconds ---\n".format(time.time() - start_time))
#
#start_time = time.time()
#m, der = log_marginal_likelihood.func_grad(scaledparams, x, y, zstates)
#print ("Log marginal likelihood from fast calculation: {:f}".format(m))
#print ("Derivatives: {:f} {:f} {:f} {:f} {:f}".format(der[0], der[1], der[2], der[3], der[4]))
#print("Calculated in {:f} seconds ---\n".format(time.time() - start_time))
#
#init_params = np.array([0.005, 0.0, 0.1, 0.1, 1 / 0.5 / 0.5])

#start_time = time.time()
#m, der = logmarglik.func_grad(scaledparams, x, y, zstates)
#print ("Log marginal likelihood: {:f}".format(m))
#print ("Derivatives: {:f} {:f} {:f} {:f} {:f}".format(der[0], der[1], der[2], der[3], der[4]))
#print("Calculated in {:f} seconds ---\n".format(time.time() - start_time))


##m, der = emp_bayes._log_marginal_likelihood()
##print ("Log marginal likelihood: {:f}".format(m))
##print ("Derivatives: {:f} {:f} {:f} {:f} {:f}".format(der[0], der[1], der[2], der[3], der[4]))
##print("Calculated in {:f} seconds ---\n".format(time.time() - start_time))
#
#
#bounds = [[scaledparams[i], scaledparams[i]] for i in range(5)]
#bounds[0] = [None, None]
##bounds[1] = [None, None]
#bounds[2] = [None, None]
#bounds[3] = [None, None]
#bounds[4] = [None, None]
#
#args = x, y, zstates
#lml_min = optimize.minimize(logmarglik.func_grad,
#                            scaledparams,
#                            args = args,
#                            method='L-BFGS-B',
#                            jac=True,
#                            bounds=bounds,
#                            options={'maxiter': 200000,
#                                     'maxfun': 2000000,
#                                     'ftol': 1e-9,
#                                     'gtol': 1e-9,
#                                     'disp': True})
#
#print(lml_min)
#
#res = hyperparameters.descale(lml_min.x)
#res[4] = np.sqrt(1 / res[4])
#print('\n'.join(['{:g}'.format(x) for x in list(res)]))
#
