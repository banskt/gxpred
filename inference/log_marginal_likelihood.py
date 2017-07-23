#!/usr/bin/env python

import numpy as np
import random
import math
from utils import hyperparameters

def func_grad(scaledparams, x, y, zstates):
    params = hyperparameters.descale(scaledparams)
    pi = params[0]
    mu = params[1]
    sigma = params[2]
    sigmabg = params[3]
    tau = params[4]

    logmarglik, der, pz, l = iterative_inverse(pi, mu, sigma, sigmabg, tau, x, y, zstates)
    der = hyperparameters.gradscale(params, der)

    #delta = 0.0001
    #newparams = scaledparams.copy()
    #newparams[4] += delta
    #newtau = hyperparameters.descale(newparams)[4]
    #newm, dummy1, dummy2 = iterative_inverse(pi, mu, sigma, sigmabg, newtau, x, y, zstates)
    #tau_grad = (newm - logmarglik) / delta
    #sigmatau = np.sqrt(1 / tau)
    #print ("Params here are: {:g} {:g} {:g} {:g} {:g}".format(pi, mu, sigma, sigmabg, sigmatau))
    #print ("Derivative of tau by brute force is {:f}".format(tau_grad))
    #print ("Derivative of tau from equation is {:f}".format(der[4]))

    return -logmarglik, -der

def func(scaledparams, x, y, zstates):
    params = hyperparameters.descale(scaledparams)
    pi = params[0]
    mu = params[1]
    sigma = params[2]
    sigmabg = params[3]
    tau = params[4]

    logmarglik, der, pz, l = iterative_inverse(pi, mu, sigma, sigmabg, tau, x, y, zstates, grad = False)

    return -logmarglik

def prob_comps(scaledparams, x, y, zstates):
    params = hyperparameters.descale(scaledparams)
    pi = params[0]
    mu = params[1]
    sigma = params[2]
    sigmabg = params[3]
    tau = params[4]

    logmarglik, der, pz, l = iterative_inverse(pi, mu, sigma, sigmabg, tau, x, y, zstates, grad = False)

    return pz

def raw_comps(scaledparams, x, y, zstates):
    params = hyperparameters.descale(scaledparams)
    pi = params[0]
    mu = params[1]
    sigma = params[2]
    sigmabg = params[3]
    tau = params[4]

    logmarglik, der, pz, l = iterative_inverse(pi, mu, sigma, sigmabg, tau, x, y, zstates, grad = False)

    return l


def mat3mul(A, B, C):
    return np.dot(A, np.dot(B, C))

def full(scaledparams, x, y, zstates):
    params = hyperparameters.descale(scaledparams)
    pi = params[0]
    mu = params[1]
    sigma = params[2]
    sigmabg = params[3]
    tau = params[4]
    nvar = x.shape[0]
    nsample = x.shape[1]
    constpi = math.pi
    marglik_k = 0
    lmlzlist = list()
    logk = 0

    sigma2 = sigma * sigma
    sigmabg2 = sigmabg * sigmabg
    muz0 = np.zeros(nvar)
    sigz0 = np.repeat(sigmabg2, nvar)

    for z in zstates:
        nz = len(z)
        muz = muz0.copy()
        muz[z] = mu
        sigz = sigz0.copy()
        sigz[z] = sigma2

        # P(z | theta)
        log_probz = nz * np.log(pi) + (nvar - nz) * np.log(1 - pi)

        # N(y | Mz, Sz)
        #Sz = np.einsum('ki, k, kj -> ij', x, sigz, x)
        Sz = mat3mul(x.T, np.diag(sigz), x)
        Sz[np.diag_indices_from(Sz)] += 1/tau
        Mz = np.einsum('ij, i -> j', x, muz)
        logdetS = np.linalg.slogdet(Sz)[1]
        invS = np.linalg.inv(Sz)
        y_minus_M = y - Mz
        nterm = np.einsum('i, ij, j', y_minus_M, invS, y_minus_M)
        log_normz = - 0.5 * (logdetS + (nsample * np.log(2 * constpi)) + nterm)

        # Combine the log values
        lmlzlist.append(log_probz + log_normz)

    lmlzarr = np.array(lmlzlist)
    logk = np.max(lmlzarr)
    marglik_k = np.sum(np.exp(lmlzarr - logk))
    logmarglik = logk + np.log(marglik_k)
    return lmlzlist, -logmarglik

def iterative_inverse(pi, mu, sigma, sigmabg, tau, x, y, zstates, grad = True):
    nvar = x.shape[0]
    nsample = x.shape[1]
    constpi = math.pi
    marglik_k = 0
    lmlzlist = list()
    BZinvlist = list()
    Sinvlist = list()
    _tmplist = list()
    logk = 0

    sigma2 = sigma * sigma
    sigmabg2 = sigmabg * sigmabg
    h = 1/sigma2 - 1/sigmabg2

    # calculate for zstate = [[]]
    nz = 0
    sigz0 = np.repeat(sigmabg2, nvar)

    log_probz = nz * np.log(pi) + (nvar - nz) * np.log(1 - pi)

    B0 = np.linalg.inv(np.diag(sigz0)) + tau * np.dot(x, x.T)
    B0inv = np.linalg.inv(B0)
    #Sinv = - tau * tau * np.einsum('li, lk, kj -> ij', x, B0inv, x)
    Sinv = - tau * tau * np.dot(np.dot(x.T, B0inv), x)
    Sinv[np.diag_indices_from(Sinv)] += tau

    logB0det = np.linalg.slogdet(B0)[1]
    logdetS = - nsample * np.log(tau) + (nvar - nz) * np.log(sigmabg2) + nz * np.log(sigma2) + logB0det

    y_minus_M = y
    nterm = np.einsum('i, ij, j', y_minus_M, Sinv, y_minus_M)
    log_normz = - 0.5 * (logdetS + (nsample * np.log(2 * constpi)) + nterm)

    #logk = - log_probz - log_normz
    #lmlz = log_probz + log_normz + logk
    #kmarglik += np.exp(lmlz)
    #lmlzlist.append(lmlz - logk)
    lmlzlist.append(log_probz + log_normz)
    _tmplist.append(y[0])
    BZinvlist.append(B0inv)
    Sinvlist.append(Sinv)

    for z in zstates[1:]:
        nz = len(z)


        # Start from B0, and update the BZinv and logBZdet sequentially
        base_BZinv = B0inv
        base_logBZdet = logB0det
        Mz = np.zeros(nsample)
        for zpos in z:
           mod = h / (1 + h * base_BZinv[zpos, zpos])
           BZinv = base_BZinv - mod * np.einsum('i, j -> ij', base_BZinv[:,zpos], base_BZinv[zpos,:])
           logBZdet = base_logBZdet + np.log(1 + h * base_BZinv[zpos, zpos])
           base_BZinv = BZinv
           base_logBZdet = logBZdet
           #y_minus_M -= mu * x[zpos, :]
           Mz += mu * x[zpos, :]

        #muz = np.zeros(nvar)
        #muz[z] = mu
        #Mz = np.einsum('ij, i -> j', x, muz)
        y_minus_M = y - Mz

        # P(z | theta)
        log_probz = nz * np.log(pi) + (nvar - nz) * np.log(1 - pi)

        # N(y | Mz, Sz)
        Sinv = - tau * tau * mat3mul(x.T, BZinv, x)
        Sinv[np.diag_indices_from(Sinv)] += tau

        logdetS = - nsample * np.log(tau) + (nvar - nz) * np.log(sigmabg2) + nz * np.log(sigma2) + logBZdet

        nterm = np.einsum('i, ij, j', y_minus_M, Sinv, y_minus_M)
        log_normz = - 0.5 * (logdetS + (nsample * np.log(2 * constpi)) + nterm)

        # Combine the log values
        lmlzlist.append(log_probz + log_normz)
        _tmplist.append(y[0])
        BZinvlist.append(BZinv)
        Sinvlist.append(Sinv)


    lmlzarr = np.array(lmlzlist)
    logk = np.max(lmlzarr)
    marglik_k = np.sum(np.exp(lmlzarr - logk))
    logmarglik = logk + np.log(marglik_k)

    pz = np.exp(lmlzlist - logmarglik)
    if (grad):
        der = gradients(pi, mu, sigma, sigmabg, tau, x, y, zstates, pz, BZinvlist, Sinvlist)
        #der = hyperparameters.gradscale(params, der)
    else:
        der = np.zeros(5)

    return logmarglik, der, pz, _tmplist
    #return -logmarglik, -der, pz

def gradients(pi, mu, sigma, sigmabg, tau, x, y, zstates, pz, BZinvlist, Sinvlist):

    pi_grad = 0
    mu_grad = 0
    sigma_grad = 0
    sigbg_grad = 0
    tau_grad = 0

    nvar = x.shape[0]
    nsample = x.shape[1]
    sigma2 = sigma * sigma
    sigmabg2 = sigmabg * sigmabg
    h = 1/sigma2 - 1/sigmabg2

    for i, z in enumerate(zstates):
        nz = len(z)
        picomp = nz / pi - (nvar - nz) / (1 - pi)
        pi_grad += pz[i] * picomp
        #print (pi_grad)


        if nz == 0:
            B0inv = BZinvlist[i]
            Sinv  = Sinvlist[i]
    
            mu_grad += 0
            sigma_grad += 0

            y_minus_M = y

            dlogdetS_dsigbg = 2 * np.sum(sigmabg2 - np.diag(B0inv)) / sigmabg2 / sigmabg
            dB0_dsigbg = np.diag(np.repeat( - 2.0 / sigmabg2 / sigmabg, nvar))
            dB0inv_dsigbg = - np.dot(B0inv, np.dot(dB0_dsigbg, B0inv))
            dSinv_dsigbg = - tau * tau * np.dot(x.T, np.dot(dB0inv_dsigbg, x))
            term1 = - 0.5 * dlogdetS_dsigbg
            term2 = - 0.5 * np.dot(y_minus_M.T, np.dot(dSinv_dsigbg, y_minus_M))
            sigbg_grad += pz[i] * (term1 + term2)

            dlogdetS_dtau = np.trace(- Sinv) / tau / tau
            dSinv_dtau = np.dot(Sinv, Sinv) / tau / tau
            term1 = - 0.5 * dlogdetS_dtau
            term2 = - 0.5 * np.dot(y_minus_M.T, np.dot(dSinv_dtau, y_minus_M))
            tau_grad += pz[i] * (term1 + term2)

        else:
            BZinv = BZinvlist[i]
            Sinv  = Sinvlist[i]
            
            muzTx = np.zeros(nsample)
            zTx   = np.zeros(nsample)
            for zpos in z:
                muzTx += mu * x[zpos, :]
                zTx   += x[zpos, :]
            y_minus_M = y - muzTx
            mucomp = np.dot(np.dot(zTx, Sinv), y_minus_M)
            mu_grad += pz[i] * mucomp

            #zpos = z[0]
            dlogdetS_dsigma = 0
            dBZinv_dsigma = np.zeros((nvar, nvar))
            for zpos in z:
                dlogdetS_dsigma += 2 * (1 - BZinv[zpos, zpos] / sigma2) / sigma
                dBZinv_dsigma += 2 * np.outer(BZinv[:, zpos], BZinv[zpos, :]) / sigma2 / sigma
                
            #dlogdetS_dsigma = 2 * (1 - BZinv[zpos, zpos] / sigma2) / sigma
            #dBZinv_dsigma = 2 * np.outer(BZinv[:, zpos], BZinv[zpos, :]) / sigma2 / sigma
            dSinv_dsigma = - tau * tau * np.dot(x.T, np.dot(dBZinv_dsigma, x))
            term1 = - 0.5 * dlogdetS_dsigma
            term2 = - 0.5 * np.dot(y_minus_M.T, np.dot(dSinv_dsigma, y_minus_M))

            sigma_grad += pz[i] * (term1 + term2)

            #Afull = np.sum(sigmabg2 - np.diag(BZinv))
            Afull = nvar * sigmabg2 - np.trace(BZinv)
            Asub = 0
            dBZ_dsigbg = np.diag(np.repeat( - 2.0 / sigmabg2 / sigmabg, nvar))
            for zpos in z:
                Asub += sigmabg2 - BZinv[zpos, zpos]
                dBZ_dsigbg[zpos, zpos] = 0.0
            #Asub = sigmabg2 - BZinv[zpos, zpos]
            dlogdetS_dsigbg = 2 * (Afull - Asub) / sigmabg2 / sigmabg
            dBZinv_dsigbg = - np.dot(BZinv, np.dot(dBZ_dsigbg, BZinv))
            dSinv_dsigbg = - tau * tau * np.dot(x.T, np.dot(dBZinv_dsigbg, x))
            term1 = - 0.5 * dlogdetS_dsigbg
            term2 = - 0.5 * np.dot(y_minus_M.T, np.dot(dSinv_dsigbg, y_minus_M))

            sigbg_grad += pz[i] * (term1 + term2)

            dlogdetS_dtau = np.trace(- Sinv) / tau / tau
            dSinv_dtau = np.dot(Sinv, Sinv) / tau / tau
            term1 = - 0.5 * dlogdetS_dtau
            term2 = - 0.5 * np.dot(y_minus_M.T, np.dot(dSinv_dtau, y_minus_M))
            tau_grad += pz[i] * (term1 + term2)

    return np.array([pi_grad, mu_grad, sigma_grad, sigbg_grad, tau_grad])
