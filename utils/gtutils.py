#!/usr/bin/env python

import numpy as np


def normalize(snps, gt):
    f = [x.maf for x in snps]
    f = np.array(f).reshape(-1, 1)
    newgt = (gt - (2 * f)) / np.sqrt(2 * f * (1 - f))
    return newgt


def remove_low_maf(snps, gt, maf):
    qc_indices = [i for i, snp in enumerate(snps) if snp.maf >= maf and snp.maf <= (1.0 - maf)]
    newsnps = [snps[i] for i in qc_indices]
    newgt = gt[np.array(qc_indices), :]
    return newsnps, newgt


def prediction_variables(snps, subsnps, gt):
    ''' check if SNP is present in the genotype
        check if the minor and major alleles of the SNP are in proper order as the prediction SNP.
    '''
    nsnps = len(subsnps)
    nsample = gt.shape[1]
    newgt = np.zeros((nsnps, nsample))
    snp_bplist = [x.bp_pos for x in snps]
    found = len(subsnps)
    for i, snp in enumerate(subsnps):
        f = snp.maf
        if snp.bp_pos in snp_bplist:
            j = snp_bplist.index(snp.bp_pos)
            if snps[j].ref_allele == snp.ref_allele and snps[j].alt_allele == snp.alt_allele:
                x = gt[j, :]
            elif snps[j].ref_allele == snp.alt_allele and snps[j].alt_allele == snp.ref_allele:
                x = 2.0 - gt[j, :]
            else:
                x = np.array([f] * nsample)
                found -= 1
                #x = np.zeros(nsample)
        else:
            x = np.array([f] * nsample)
            #x = np.zeros(nsample)
        newgt[i, :] = x
    print("Found {:d}/{:d} SNPs for prediction".format(found, len(subsnps)))
    return newgt
