#!/usr/bin/env python

import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import utils.mpl_stylesheet as mplstyle

mplstyle.banskt_presentation()

def simulate_allsnps(nsample, nsnps, hg2):
    dosage = np.zeros((nsnps, nsample))
    freq = np.zeros(nsnps)
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
        dosage[i,:] = np.random.binomial(2, snpfreq, size = nsample)
        freq[i] = snpfreq
    freq = np.array(freq).reshape(-1, 1)
    gt = (dosage - (2 * freq)) / np.sqrt(2 * freq * (1 - freq))

    betas = np.random.normal(0, 1, nsnps)
    betas *= np.sqrt( hg2 / np.sum(np.square(betas)) )
    sigma = np.std(betas)

    # Simulate gene expression
    expression = np.zeros(nsample)
    if hg2 > 1:
        print("Error: Heritability {:g} is greater than 1".format(hg2))
        sys.exit(0)
    else:
        randstd = np.sqrt(1 - hg2)
        print("Creating phenotype with sigma {:g}, heritability {:g}, sigma_tau {:g}".format(sigma, hg2, randstd))
        print("Using {:d} causal SNPs".format(nsnps))
        # Gene expression, Y = X'v + e
        error = np.random.normal(0, randstd, nsample)
        expression = np.einsum('ij, i -> j', gt, betas) + error

    return gt, expression, betas, sigma


def simulate_base(nsample, nsnps, pi, hg2):
    dosage = np.zeros((nsnps, nsample))
    freq = np.zeros(nsnps)
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
        dosage[i,:] = np.random.binomial(2, snpfreq, size = nsample)
        freq[i] = snpfreq
    freq = np.array(freq).reshape(-1, 1)
    gt = (dosage - (2 * freq)) / np.sqrt(2 * freq * (1 - freq))

    causal_snps = list()
    for i in range(nsnps):
        mrand = random.uniform(0, 1)
        if pi > mrand:
            causal_snps.append(i)
    cnum = len(causal_snps)
    causal_snps = np.array(causal_snps)

    # initialize the v's to all non-causal
    v = np.zeros(nsnps)
    betas = np.random.normal(0, 1, cnum)
    betas *= np.sqrt( hg2 / np.sum(np.square(betas)) )
    sigma = np.std(betas)
    if cnum > 0:
        v[causal_snps] = betas #np.random.normal(0, sigma, cnum)
    else:
        print("Warning: Could not select any causal SNP")
        print("Maximum causal probability was {:g}".format(np.max(pi)))

    # Simulate gene expression
    expression = np.zeros(nsample)
    if hg2 > 1:
        print("Error: Heritability {:g} is greater than 1".format(hg2))
        sys.exit(0)
    else:
        randstd = np.sqrt(1 - hg2)
        print("Creating phenotype with sigma {:g}, heritability {:g}, sigma_tau {:g}".format(sigma, hg2, randstd))
        print("Using {:d} causal SNPs".format(cnum))
        # Gene expression, Y = X'v + e
        error = np.random.normal(0, randstd, nsample)
        expression = np.einsum('ij, i -> j', gt, v) + error

    return gt, expression, causal_snps, v, sigma
        

def simulate(nsample, nsnps, features, distfeat, gamma, mu = 0.0, sigma = 0.05, sigmabg = 0.001):
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
    #genotype = np.random.normal(0, 1, nsnps * nsample).reshape(nsnps, nsample)
    #gt = (genotype - np.mean(genotype, axis = 1).reshape(-1, 1)) / (np.std(genotype, axis =1).reshape(-1, 1))

    # Select the causal SNPs
    A = np.einsum('k, ik -> i', gamma, features)
    pi = 1 / (1 + distfeat * np.exp(-A))

    plt.figure()
    plt.hist(pi)
    plt.ylabel("Density")
    plt.xlabel(r'$\pi$')
    plt.show()
    
    causal_snps = list()
    for i in range(nsnps):
        mrand = random.uniform(0, 1)
        if pi[i] > mrand:
            causal_snps.append(i)
    cnum = len(causal_snps)
    causal_snps = np.array(causal_snps)

    # initialize the v's to all non-causal
    if sigmabg == 0:
        v = np.zeros(nsnps)
    else:
        v = np.random.normal(0, sigmabg, nsnps)

    if cnum > 0:
        v[causal_snps] = np.random.normal(mu, sigma, cnum)
        hg2 = np.sum(np.square(v[causal_snps]))
    else:
        print("Warning: Could not select any causal SNP")
        print("Maximum causal probability was {:g}".format(np.max(pi)))
        hg2 = 0

    # Simulate gene expression
    expression = np.zeros(nsample)
    if hg2 > 1:
        print("Error: Heritability {:g} is greater than 1".format(hg2))
        sys.exit(0)
    else:
        randstd = np.sqrt(1 - hg2)
        print("Creating phenotype with sigma {:g}, heritability {:g}, sigma_tau {:g}".format(sigma, hg2, randstd))
        print("Using {:d} causal SNPs".format(cnum))
        # Gene expression, Y = X'v + e
        error = np.random.normal(0, randstd, nsample)
        expression = np.einsum('ij, i -> j', gt, v) + error

    return gt, expression, causal_snps, v
