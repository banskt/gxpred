#!/usr/bin/env python

import numpy as np

def normalize(snps, gt):
    f = [x.maf for x in snps]
    f = np.array(f).reshape(-1, 1)
    newgt = (gt - (2 * f)) / np.sqrt(2 * f * (1 - f))
    return newgt

def remove_low_maf(snps, gt):
    qc_indices = [i for i, snp in enumerate(snps) if snp.maf > 0.01]
    newsnps = [snps[i] for i in qc_indices]
    newgt = gt[np.array(qc_indices), :]
    return newsnps, newgt
