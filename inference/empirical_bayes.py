#!/usr/bin/env python

import numpy as np
from scipy import optimize

from utils import hyperparameters
from inference import logmarglik
from inference import zstates_old_method as zs_old
from inference import zstates as zs

class EmpiricalBayes:

    _global_zstates = list()
    _update_zstates = True

    def __init__(self, genotype, phenotype, cmax, params, ztarget = 0.98, method = "old"):

        self._genotype = genotype
        self._phenotype = phenotype
        self._cmax = cmax
        self._params = hyperparameters.scale(params)
        self._ztarget = ztarget
        self._method = method

        self._nsnps = genotype.shape[0]
        self._nsample = genotype.shape[1]


    @property
    def params(self):
        return hyperparameters.descale(self._params)


    @property
    def zstates(self):
        return self._global_zstates


    @property
    def success(self):
        return self._success


    def fit(self):
        scaledparams = self._params

        bounds = [[None, None] for i in range(5)]
        bounds[1] = [scaledparams[1], scaledparams[1]]
        bounds[2] = [None, 2]
        bounds[3] = [None, 2]
        bounds[4] = [None, 20]

        self._callback_zstates(scaledparams)
        #self._update_zstates = False

        lml_min = optimize.minimize(self._log_marginal_likelihood,
                                    scaledparams,
                                    method='L-BFGS-B',
                                    jac=True,
                                    bounds=bounds,
                                    callback=self._callback_zstates,
                                    options={'maxiter': 200000,
                                             'maxfun': 2000000,
                                             'ftol': 1e-9,
                                             'gtol': 1e-9,
                                             'disp': True})
        self._params = lml_min.x
        self._success = lml_min.success
        print(lml_min)


    def _log_marginal_likelihood(self, scaledparams):
        #if (scaledparams[0] < -500 or scaledparams[2] > 1 or scaledparams[3] > 1 or scaledparams[4] > 10):
        #    print ("Correcting scaledparams and derivatives")
        #    print (scaledparams)
        #    lml = self._lml
        #    der = self._der
        #else:
        lml, der = logmarglik.func_grad(scaledparams, self._genotype, self._phenotype, self._global_zstates)
        self._lml = lml
        self._der = der
        return lml, der


    def _callback_zstates(self, scaledparams):
        if self._update_zstates:
            if   self._method == "old":
                self._global_zstates = zs_old.create(scaledparams, self._genotype, self._phenotype, self._cmax, self._nsnps, self._ztarget)
                #print ("|OLD| {:d} zstates".format(len(self._global_zstates)))
            elif self._method == "new":
                #print ("|NEW| Creating zstates.")
                self._global_zstates =     zs.create(scaledparams, self._genotype, self._phenotype, self._cmax, self._nsnps, self._ztarget)
                #print ("|NEW| {:d} zstates".format(len(self._global_zstates)))
            elif self._method == "basic":
                self._global_zstates = [[]] + [[i] for i in range(self._nsnps)]
