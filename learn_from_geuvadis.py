#!/usr/bin/env python

import argparse
import numpy as np

from iotools.readvcf import ReadVCF
from iotools.readrpkm import ReadRPKM
from iotools.io_model import IOModel
from inference.linreg_association import LinRegAssociation
from inference.empirical_bayes import EmpiricalBayes
from utils import hyperparameters
from inference import logmarglik
from iotools import readgtf
from utils import gtutils
from utils import mfunc
from utils.containers import ZstateInfo

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
                        #nargs='*',
                        type=int,
                        dest='chrom',
                        metavar='CHR',
                        help='choose genes from this chromosome only')

    parser.add_argument('--params',
                        nargs='*',
                        default=[0.1, 0.0, 0.001, 0.01, 0.005],
                        type=float,
                        dest='params',
                        metavar='FLOAT',
                        help='initialization parameters [pi, mu, sigma, sigma_bg, sigma_tau]')

    parser.add_argument('--outdir',
                        type=str,
                        dest='outdir',
                        metavar='DIR',
                        help='name of the output directory for storing the model')


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
snps, genotype = gtutils.remove_low_maf(snps, genotype, 0.1)
gt = gtutils.normalize(snps, genotype)

# Annotation
gene_info = readgtf.gencode_v12(opts.gtfpath, include_chrom = opts.chrom)

# Gene Expression
rpkm = ReadRPKM(opts.rpkmpath)
expression = rpkm.expression
expr_donors = rpkm.donor_ids
gene_names = rpkm.gene_names

# Selection
vcfmask, exprmask = mfunc.select_donors(vcf_donors, expr_donors)
genes, indices = mfunc.select_genes(gene_info, gene_names)


min_snps = 100
pval_cutoff = 0.005
window = 1000000
cmax = 2
init_params = np.array(opts.params)

model = IOModel(opts.outdir, opts.chrom)

for i, gene in enumerate(genes[:3]):
    k = indices[i]

    # select only the cis-SNPs
    cismask = mfunc.select_snps(gene, snps, window)
    if len(cismask) > 0:
        target = expression[k, exprmask]
        predictor = gt[cismask][:, vcfmask]
        snpmask = cismask

        # if number of cis SNPs > threshold, use p-value cut-off
        if len(cismask) > min_snps:
            assoc_model = LinRegAssociation(predictor, target, min_snps, pval_cutoff)
            pvalmask = cismask[assoc_model.selected_variables]
            print ("Found {:d} SNPs, reduced to {:d} SNPs (min p-value {:g}) for {:s}".format(len(cismask), len(pvalmask), assoc_model.ordered_pvals[len(pvalmask) - 1], gene.name))
            predictor = gt[pvalmask][:, vcfmask]
            snpmask = pvalmask
        else:
            print ("Found {:d} SNPs for {:s}".format(len(cismask), gene.name))

        # perform the analysis
        emp_bayes = EmpiricalBayes(predictor, target, 1, init_params, method="new")
        emp_bayes.fit()
        res = emp_bayes.params
        if cmax > 1:
            emp_bayes = EmpiricalBayes(predictor, target, cmax, res, method="new")
            emp_bayes.fit()
            res = emp_bayes.params
            print('\n'.join(['{:g}'.format(x) for x in list(res)]))

        model_snps = [snps[x] for x in snpmask]
        model_zstates = list()
        scaledparams = hyperparameters.scale(emp_bayes.params)
        zprob, zexp = logmarglik.model_exp(scaledparams, predictor, target, emp_bayes.zstates)
        for j, z in enumerate(emp_bayes.zstates):
            this_zstate = ZstateInfo(state = z,
                                     prob  = zprob[j],
                                     exp   = list(zexp[j, :]) )
            model_zstates.append(this_zstate)
        model.write_gene(gene, model_snps, model_zstates)

    else:
        print("No genotype for gene {:s}".format(gene.name))

print ("All done")
