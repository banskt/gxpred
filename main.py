#!/usr/bin/env python

import argparse
import numpy as np

from iotools.readvcf import ReadVCF
from iotools.readrpkm import ReadRPKM
from iotools import readgtf
from utils import gtutils
from utils import mfunc

def parse_args():

    parser = argparse.ArgumentParser(description='Bayesian model for learning genetic contribution in gene expression')

    parser.add_argument('--vcf',
                        type=str,
                        dest='vcfpath',
                        metavar='FILE',
                        help='input genotype file in vcf.gz format')


    parser.add_argument('--rpkm',
                        type=str,
                        dest='rpkmpath',
                        metavar='FILE',
                        help='RNA-Seq RPKM counts for genes')


    parser.add_argument('--gtf',
                        type=str,
                        dest='gtfpath',
                        metavar='FILE',
                        help='Gene Annotation file')

    parser.add_argument('--chrom',
                        nargs='*',
                        type=str,
                        dest='chrom',
                        metavar='CHR',
                        help='choose genes from this chromosome only')


    opts = parser.parse_args()
    return opts


# Checks
# 1. check if geneids file exist
# 2. check if vcf file exist
# 3. check if expression file exist

opts = parse_args()

# Genotype
vcf = ReadVCF(opts.vcfpath)
genotype = vcf.dosage
vcf_donors = vcf.donor_ids
snps = vcf.snpinfo

# Quality control
snps, genotype = gtutils.remove_low_maf(snps, genotype)
gt = gtutils.normalize(snps, genotype)

# Annotation
gene_info = readgtf.gencode_v12(opts.gtfpath, include_chroms = opts.chrom)

# Gene Expression
rpkm = ReadRPKM(opts.rpkmpath)
expression = rpkm.expression
expr_donors = rpkm.donor_ids
gene_names = rpkm.gene_names

# Selection
vcfmask, exprmask = mfunc.select_donors(vcf_donors, expr_donors)
genes, indices = mfunc.select_genes(gene_info, gene_names)

for i, gene in enumerate(genes[:3]):
    k = indices[i]
    snpmask = mfunc.select_snps(gene, snps, 1e4)
    if len(snpmask) > 0:
        print ("Found genotype for {:s}\n".format(gene.name))
        target = expression[k, exprmask]
        predictor = gt[snpmask][:, vcfmask]
    #else:
    #    print("No genotype for gene {:s}".format(gene.ensembl_id))
    #    model = SparseMultipleRegression(target, predictor)


print ("All done")
