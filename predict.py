#!/usr/bin/env python

import argparse
import numpy as np

from iotools.readvcf import ReadVCF
from iotools.readrpkm import ReadRPKM
from iotools.io_model import ReadModel
from utils.containers import GeneExpressionArray
from utils import mfunc
from utils import gtutils


from iotools import readgtf
from inference.linreg_association import LinRegAssociation
from inference.empirical_bayes import EmpiricalBayes

def parse_args():

    parser = argparse.ArgumentParser(description='Prediction of gene expression')

    parser.add_argument('--vcf',
                        type=str,
                        dest='vcfpath',
                        metavar='FILE',
                        help='input genotype file in vcf.gz format')

    parser.add_argument('--model',
                        type=str,
                        dest='modelpath',
                        metavar='DIR',
                        help='input directory of the trained model')

    parser.add_argument('--chrom',
                        type=int,
                        dest='chrom',
                        metavar='CHR',
                        help='predict egenes from this chromosome only')

    parser.add_argument('--outprefix',
                        type=str,
                        dest='outfileprefix',
                        metavar='FILE',
                        help='name of the output file of the predicted gene expression')


    opts = parser.parse_args()
    return opts


opts = parse_args()

# Genotype
vcf = ReadVCF(opts.vcfpath, mode="DS")
genotype = vcf.dosage
vcf_donors = vcf.donor_ids
snps = vcf.snpinfo
nsample = len(vcf_donors)

# Model
model = ReadModel(opts.modelpath, opts.chrom)
genes = model.genes

# Prediction
gx = list()
for gene in genes:
    model.read_gene(gene)
    model_snps = model.snps
    model_zstates = model.zstates

    x = gtutils.prediction_variables(snps, model_snps, genotype)
    x = gtutils.normalize(model_snps, x)

    ypred = np.zeros(nsample)
    for z in model_zstates:
        ypred += z.prob * np.dot(x.T, z.exp)

    gx.append(GeneExpressionArray(geneid = gene.ensembl_id, expr_arr = ypred))


# Write output
mfunc.write_gcta_phenotype(opts.outfileprefix, vcf_donors, gx)
