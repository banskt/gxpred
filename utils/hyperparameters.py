#!/usr/bin/env python

import numpy as np

def scale(params, pointnormal = False):
    nhyp = params.shape[0]
    gammas = params [ :(nhyp - 4) ]
    #beta_pi = - np.log((1 / params[0]) - 1)
    beta_mu = 100 * params[ nhyp - 4 ]
    beta_sigma = np.log(params[ nhyp - 3 ])

    # current hack, propagate it to source calls later
    if params[ nhyp - 2 ] == 0:
        pointnormal = True
    if pointnormal:
        beta_sigmabg = 0
    else:
        beta_sigmabg = np.log(params[ nhyp - 2 ])
    beta_tau = - 0.5 * np.log(params[ nhyp - 1])
    scaledparams = np.array( list(gammas) + [beta_mu, beta_sigma, beta_sigmabg, beta_tau])
    return scaledparams
    
def descale(scaledparams, pointnormal = False):
    nhyp = scaledparams.shape[0]
    gammas = scaledparams [:(nhyp - 4)]
    #pi = np.exp(scaledparams[0]) / (1 + np.exp(scaledparams[0]))
    mu = scaledparams[nhyp - 4] / 100
    sigma = np.exp(scaledparams[nhyp - 3])

    # current hack, propagate it to source calls later
    if scaledparams[ nhyp - 2 ] == 0:
        pointnormal = True

    if pointnormal:
        sigmabg = 0
    else:
        sigmabg = np.exp(scaledparams[nhyp - 2])

    tau = np.exp(- 2.0 * scaledparams[nhyp - 1])
    params = np.array(list(gammas) + [mu, sigma, sigmabg, tau])
    return params

def gradscale(params, der):
    #beta_pi_grad = params[0] * (1 - params[0]) * der[0]
    nhyp = params.shape[0]
    gammas = params [ :(nhyp - 4) ]
    gamma_grad = der[ :(nhyp - 4) ]
    beta_mu_grad = der[ nhyp - 4 ] / 100
    beta_sigma_grad = params[ nhyp - 3 ] * der[ nhyp - 3 ]
    beta_sigmabg_grad = params[ nhyp - 2 ] * der[ nhyp - 2 ]
    beta_tau_grad = - 2.0 * params[ nhyp - 1 ] * der[ nhyp - 1 ]
    scaled_der = np.array(list(gamma_grad) + [beta_mu_grad, beta_sigma_grad, beta_sigmabg_grad, beta_tau_grad])
    return scaled_der
