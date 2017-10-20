#!/usr/bin/env python

import numpy as np

def scale(params):
    beta_pi = - np.log((1 / params[0]) - 1)
    beta_mu = 100 * params[1]
    beta_sigma = np.log(params[2])
    beta_sigmabg = np.log(params[3])
    beta_tau = - 0.5 * np.log(params[4])
    scaledparams = np.array([beta_pi, beta_mu, beta_sigma, beta_sigmabg, beta_tau])
    return scaledparams
    
def descale(scaledparams):
    pi = np.exp(scaledparams[0]) / (1 + np.exp(scaledparams[0]))
    mu = scaledparams[1] / 100
    sigma = np.exp(scaledparams[2])
    sigmabg = np.exp(scaledparams[3])
    tau = np.exp(- 2.0 * scaledparams[4])
    params = np.array([pi, mu, sigma, sigmabg, tau])
    return params

def gradscale(params, der):
    beta_pi_grad = params[0] * (1 - params[0]) * der[0]
    beta_mu_grad = der[1] / 100
    beta_sigma_grad = params[2] * der[2]
    beta_sigmabg_grad = params[3] * der[3]
    beta_tau_grad = - 2.0 * params[4] * der[4]
    scaled_der = np.array([beta_pi_grad, beta_mu_grad, beta_sigma_grad, beta_sigmabg_grad, beta_tau_grad])
    return scaled_der
