#!/usr/bin/env python

import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import utils.mpl_stylesheet as mplstyle
mplstyle.banskt_presentation()

from utils.containers import SnpInfo
from sklearn.preprocessing import scale


def simulate_dosage(nsample = 450, nsnp = 200, fmin = 0.1, fmax = 0.9, maketest = False):

    f = np.random.uniform(fmin, fmax, nsnp)
    if maketest:
        f = np.repeat(0.1, nsnp)

    dosage = np.zeros((nsnp, nsample))
    snpinfo = list()
    for i in range(nsnp):
        if maketest:
            nfreq = np.array([[279,  54,   5]])[0]
        else:
            mafratios = np.array([(1 - f[i])**2, 2 * f[i] * (1 - f[i]), f[i]**2])
            nfreq  = np.random.multinomial(nsample, mafratios, size=1)[0]
        f1 = np.repeat(0, nfreq[0])
        f2 = np.repeat(1, nfreq[1])
        f3 = np.repeat(2, nfreq[2])
        x  = np.concatenate((f1,f2,f3))
        dosage[i, :] = np.random.permutation(x)
        this_snp = SnpInfo(chrom      = 1,
                           bp_pos     = i,
                           varid      = 'rs{:d}'.format(i),
                           ref_allele = 'A',
                           alt_allele = 'T',
                           maf        = f[i])
        snpinfo.append(this_snp)

    maf2d = f.reshape(-1, 1)
    gtnorm = (dosage - (2 * maf2d)) / np.sqrt(2 * maf2d * (1 - maf2d))
    # gtcent = dosage - np.mean(dosage, axis = 1).reshape(-1, 1)

    return snpinfo, gtnorm


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
#    cnum = np.random.binomial(nsnps, pi)
#
#    if features == None:
#        causal_snps = np.random.choice(nsnps, cnum)
#    else:
#        feature1 = np.ones((nsnps, 1))

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

def simulate_gx_fixed_h2(gt, nsnps, base_pi = 0.01, nfeat=0, nfeat_causal=0, sigma_herited_sq = 0.3, debug = False):
    feature0 = np.ones((nsnps, 1))
    features = feature0
    for i in range(0,nfeat):
        new_feature = np.random.randint(2, size=nsnps)[np.newaxis].T
        features = np.concatenate((features,new_feature), axis=1)

    g0 = -np.log((1-base_pi)/base_pi)
    gammas = np.zeros(nfeat+1) + g0

    if nfeat_causal > 0:
        feat_causal_ix = np.random.choice(np.arange(1, nfeat+1), nfeat_causal, replace=False)
        causal_feature = features[:,feat_causal_ix]
        
        max_enrichment = 1.0 + np.exp(-g0) # substract a tiny amount maybe?
        min_enrichment = 0.0
        
        enrichment = np.random.uniform(low=min_enrichment, high=max_enrichment, size=nfeat_causal)
        enrichbeta = - g0 - np.log(((1 + np.exp(-g0)) / enrichment) - 1)
        gammas[feat_causal_ix] = enrichbeta
    else:
        causal_feature = np.zeros(1)
        enrichment = np.zeros(1)
        feat_causal_ix = np.zeros(1)
    
    # Causal probabilities
    gT_A   = np.einsum('k, ik -> i', gammas, features)
    ix_pos = gT_A > 0 
    ix_neg = gT_A <= 0 
    pi_arr = np.zeros(gT_A.shape[0])
    pi_arr = 1 / (1 + np.exp(-gT_A))
    
    # Select the causal SNPs
    rand_v = np.random.rand(nsnps)
    causal_ix = rand_v < pi_arr
    ncausal = np.sum(causal_ix)

    #####
    # obtain effect size
    
    beta = np.zeros(ncausal)
    beta = np.random.normal(0, 1, size = ncausal)

    if np.sum(np.square(beta)) > 0:
        beta *= np.sqrt( sigma_herited_sq / np.sum(np.square(beta)) )

    if ncausal > 0:
        gt_causal    = gt[causal_ix, :]
        _sigma       = np.std(beta)
    else:
        gt_causal    = np.zeros((1, gt.shape[1]))
        _sigma       = 0

    gt_causal    = gt[causal_ix, :]
    gt_noncausal = gt[~causal_ix, :]

    # Get gene expression and add background noise
    geno_eff  = np.sum((beta[np.newaxis].T * gt_causal), axis=0)
    rand_var  = np.sqrt(1 - sigma_herited_sq)
    rand_eff  = np.random.normal(0, rand_var, gt_causal.shape[1])
    liability = geno_eff + rand_eff
    liability = scale(liability, with_mean=True, with_std=True)
    
    y_var = np.std(liability)**2

    h2 = np.var(geno_eff) / np.var(liability)

    _tau = 1/rand_var

    if debug:
        print("Sigma: {:f}".format(_sigma))
        print("sigma check:", np.sqrt(sigma_herited_sq/ncausal))
        print("Gamma: ", gammas)
        print("Causal SNPs: {:d}/{:d}".format(ncausal, nsnps))
        print("Enrichment:", enrichment)
        print("Tau:", _tau)
        # print("Y var:", y_var)
        print("h2:", h2)
        print("geno_eff var:", np.var(geno_eff))
        print("gx var:", np.var(liability))
    
    rand_eff2  = np.random.normal(0, rand_var, gt_causal.shape[1])
    liability2 = geno_eff + rand_eff2
    return liability, liability2, gammas, ncausal, enrichment, feat_causal_ix, causal_feature, np.where(causal_ix)[0], _sigma, _tau

def simulate_gx_with_params(gt, nsnps, base_pi = 0.01, nfeat=0, nfeat_causal=0, sigma = 0.1, sigma_bg = 0.01, debug=False):
    feature0 = np.ones((nsnps, 1))
    features = feature0
    for i in range(0,nfeat):
        new_feature = np.random.randint(2, size=nsnps)[np.newaxis].T
        features = np.concatenate((features,new_feature), axis=1)

    # feat_base_pi = 10e-2
    g0 = -np.log((1-base_pi)/base_pi)
    # featgammas0 = -np.log((1-feat_base_pi)/feat_base_pi)
    gammas = np.zeros(nfeat+1) + g0
    # gammas[0] = g0

    if nfeat_causal > 0:
        feat_causal_ix = np.random.choice(np.arange(1, nfeat+1), nfeat_causal, replace=False)
        causal_feature = features[:,feat_causal_ix]

        max_enrichment = 1.0 + np.exp(-g0) # substract a tiny amount maybe?
        min_enrichment = 0.0

        enrichment = np.random.uniform(low=min_enrichment, high=max_enrichment, size=nfeat_causal)
        enriched_g = - g0 - np.log(((1 + np.exp(-g0)) / enrichment) - 1)
        gammas[feat_causal_ix] = enriched_g
    else:
        causal_feature = np.zeros(1)
        enrichment = np.zeros(1)
        feat_causal_ix = np.zeros(1)
    
    # Causal probabilities
    gT_A   = np.einsum('k, ik -> i', gammas, features)

    ix_pos = gT_A > 0 
    ix_neg = gT_A <= 0 
    pi_arr = np.zeros(gT_A.shape[0])
    pi_arr = 1 / (1 + np.exp(-gT_A))

    # Select the causal SNPs
    rand_v    = np.random.rand(nsnps)
    causal_ix = rand_v < pi_arr
    ncausal   = np.sum(causal_ix)

    # _tmp = np.concatenate((pi_arr[np.newaxis].T, features, causal_ix[np.newaxis].T), axis=1)
    # print(_tmp)
    
    #####
    # obtain effect size
    
    beta = np.zeros(ncausal)
    beta = np.random.normal(0, sigma+sigma_bg, size = ncausal)

    if ncausal > 0:
        gt_causal    = gt[causal_ix, :]
        gt_noncausal = gt[~causal_ix, :]
        _sigma        = np.std(beta)
    else:
        gt_noncausal = gt 
        gt_causal    = np.zeros((1, gt.shape[1]))
        _sigma        = 0

    # Get gene expression and add background noise
    geno_eff  = np.sum((beta[np.newaxis].T * gt_causal), axis=0)
    if ncausal == nsnps:
        rand_eff  = 0
        rand_eff2 = 0
        _sigma_bg = sigma_bg
    else:
        beta_noncausal = np.random.normal(0, sigma_bg, size = nsnps-ncausal)
        rand_eff       = np.sum((beta_noncausal[np.newaxis].T * gt_noncausal), axis=0)
        _sigma_bg      = np.std(beta_noncausal)

        beta_noncausal = np.random.normal(0, sigma_bg, size = nsnps-ncausal)
        rand_eff2      = np.sum((beta_noncausal[np.newaxis].T * gt_noncausal), axis=0)
    liability  = geno_eff + rand_eff
    liability2 = geno_eff + rand_eff2

    if debug:
        print("Sigma: {:f}".format(_sigma))
        print("Sigma_bg: {:f}".format(_sigma_bg))
        print("Causal SNPs: {:d}/{:d}".format(ncausal, nsnps))
        print("Gammas:", gammas)
        print("Enrichment:", enrichment)
    
    return liability, liability2, gammas, ncausal, enrichment, feat_causal_ix, causal_feature, np.where(causal_ix)[0]
