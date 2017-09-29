#!/usr/bin/env python

import numpy as np

SNP_COMPLEMENT = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}

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
    exprmask = np.array([expr_donors.index(x) for x in common_donors])
    return vcfmask, exprmask


def select_snps(gene, snpinfo, window):
    ''' Find indices of the genotype matrix
        corresponding to the cis SNPs of gene
    '''
    chrom = gene.chrom
    start = gene.start - window
    end = gene.end + window
    indices = [i for i, snp in enumerate(snpinfo) if snp.chrom == chrom and
                                                     snp.bp_pos < end and
                                                     snp.bp_pos > start and
                                                     snp.alt_allele != SNP_COMPLEMENT[snp.ref_allele] and
                                                     max(len(snp.ref_allele), len(snp.alt_allele)) == 1 and
                                                     snp.maf > 0.05 and snp.maf < 1.95]
    return np.array(indices)


def match_snps(snps, subset):
    mask = list()
    snp_bplist = [x.bp_pos for x in snps]
    for snp in subset:
        if snp.bp_pos in snp_bplist:
            i = snp_bplist.index(snp.bp_pos)
            mask.append(i)
        else:
            mask.append(-1)
    return mask


def write_gcta_phenotype(fileprefix, samples, gx):
    filepath = '{0}.txt'.format(fileprefix)
    with open(filepath, 'w') as pfile:
        for i, sample in enumerate(samples):
            pheno = ['{0}'.format(x.expr_arr[i]) for x in gx]
            pstr  = '\t'.join(pheno)
            pfile.write('{0} \t {0} \t {1}\n'.format(sample, pstr))

    filepath = '{0}.geneids'.format(fileprefix)
    with open(filepath, 'w') as mfile:
        for x in gx:
            mfile.write('{0}\n'.format(x.geneid))
