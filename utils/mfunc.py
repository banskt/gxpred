#!/usr/bin/env python

import numpy as np

def select_genes(info, names):
    ''' Select genes which would be analyzed. 
        Make sure the indices are not mixed up
    '''
    allowed = [x.ensembl_id for x in info]
    common  = [x for x in names if x in allowed]
    genes = [x for x in info if x.ensembl_id in common]
    indices = [names.index(x.ensembl_id) for x in genes]
    return genes, indices


def select_donors(vcf_donors, expr_donors):
    ''' Make sure that donors are in the same order for both expression and genotype
    '''
    common_donors = [x for x in vcf_donors if x in expr_donors]
    vcfmask = np.array([vcf_donors.index(x) for x in common_donors])
    exprmask = np.array([expr_donors.index(x) for x in expr_donors])
    return vcfmask, exprmask


def select_snps(gene, snpinfo, window):
    ''' Find indices of the genotype matrix
        corresponding to the cis SNPs of gene
    '''
    chrom = gene.chrom
    start = gene.start - window
    end = gene.end + window
    indices = [i for i, snp in enumerate(snpinfo) if snp.chrom == chrom and
                                                     np.int64(snp.bp_pos) < end and
                                                     np.int64(snp.bp_pos) > start]
    return np.array(indices)
