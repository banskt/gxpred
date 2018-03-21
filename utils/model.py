#!/usr/bin/env python

import numpy as np
import random

def simulate(nsample, nsnps, pi = 0.1, mu = 0.0, sigma = 0.05, sigmabg = 0.001, tau = 200, features=None):
    # Simulate genotype =================
    dosage = list()
    freq = list()
    for i in range(nsnps):
        choose = True
        while choose:
            snpfreq = np.random.rand(1)
            if snpfreq > 0.5:
                snpfreq = 1 - snpfreq
            if snpfreq < 0.05:
                choose = True
            else:
                choose = False
        n1 = int(snpfreq * nsample)
        n2 = int(snpfreq * nsample / 2)
        n0 = nsample - n1 - n2
        dosagelist =([0] * n0) + ([1] * n1) + ([2] * n2)
        random.shuffle(dosagelist)
        dosage.append(np.array(dosagelist))
        freq.append(sum(dosagelist) / 2 / nsample)
    genotype = np.array(dosage)
    freq = np.array(freq).reshape(-1, 1)
    gt = (genotype - (2 * freq)) / np.sqrt(2 * freq * (1 - freq))

    # Simulate gene expression ============
    # Select the causal SNPs
    cnum = np.random.binomial(nsnps, pi)

    if features == None:
        causal_snps = np.random.choice(nsnps, cnum)
    else:
        feature1 = np.ones((nsnps, 1))



    # initialize the v's to all non-causal
    v = np.random.normal(0, sigmabg, nsnps)
    v[causal_snps] = np.random.normal(mu, sigma, cnum)

    # Gene expression, Y = X'v + e
    error = np.random.normal(0, 1/tau, nsample)
    expression = np.einsum('ij, i -> j', gt, v) + error

    return gt, expression, causal_snps, v
