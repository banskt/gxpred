#!/usr/bin/env python

import argparse
import numpy as np
import math

from iotools.readvcf import ReadVCF
from iotools.readOxford import ReadOxford
from iotools.readrpkm import ReadRPKM
from iotools.io_model import WriteModel
from inference.linreg_association import LinRegAssociation
from inference.empirical_bayes import EmpiricalBayes
from utils import hyperparameters
from inference import logmarglik
from iotools import readgtf
from utils import gtutils
from utils import mfunc
from utils.containers import ZstateInfo
from utils.printstamp import printStamp

import pdb

from sklearn.preprocessing import scale

def parse_args():

    parser = argparse.ArgumentParser(description='Bayesian model for learning genetic contribution in gene expression')

    parser.add_argument('--gen',
                        type=str,
                        dest='gtpath',
                        metavar='FILE',
                        help='input genotype file in Oxford format')

    parser.add_argument('--sample',
                        type=str,
                        dest='samplepath',
                        metavar='FILE',
                        help='input file with list of samples')


    parser.add_argument('--expr',
                        type=str,
                        dest='rpkmpath',
                        metavar='FILE',
                        help='RNA-Seq RPKM counts for genes')


    parser.add_argument('--gtf',
                        type=str,
                        dest='gtfpath',
                        metavar='FILE',
                        help='Gene Annotation file')

    parser.add_argument('--chr',
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

    parser.add_argument('--out',
                        type=str,
                        dest='outdir',
                        metavar='DIR',
                        help='name of the output directory for storing the model')

    parser.add_argument('--split',
                        type=int,
                        dest='split',
                        metavar='SPLIT',
                        help='split the genes in SPLIT batches.')

    parser.add_argument('--from',
                        type=int,
                        dest='fromg',
                        metavar='FROM',
                        help='Start from gene number.')

    parser.add_argument('--to',
                        type=int,
                        dest='tog',
                        metavar='TO',
                        help='Calculate to gene number.')

    opts = parser.parse_args()
    return opts


# Checks
# 1. check if geneids file exist
# 2. check if vcf file exist
# 3. check if expression file exist

opts = parse_args()

# read Genotype
oxf = ReadOxford(opts.gtpath, opts.samplepath, opts.chrom, "gtex")
genotype = np.array(oxf.dosage)
samplenames = oxf.samplenames
snps = oxf.snps_info

# Quality control
snps, genotype = gtutils.remove_low_maf(snps, genotype, 0.1)
gt = gtutils.normalize(snps, genotype)

# Annotation (use complete gene name in gtf without trimming the version)
gene_info = readgtf.gencode_v12(opts.gtfpath, include_chrom = opts.chrom, trim=False)

# Gene Expression
rpkm = ReadRPKM(opts.rpkmpath, "gtex")
expression = rpkm.expression
expr_donors = rpkm.donor_ids
gene_names = rpkm.gene_names

# Selection
printStamp("Selection of samples")
vcfmask, exprmask = mfunc.select_donors(samplenames, expr_donors)
genes, indices = mfunc.select_genes(gene_info, gene_names)


min_snps = 200
pval_cutoff = 0.001
window = 1000000
zmax = 2    # z parameter
init_params = np.array(opts.params)
init_params[4] = 1 / init_params[4] / init_params[4]

model = WriteModel(opts.outdir, opts.chrom)


if opts.fromg < 0 or opts.tog <= opts.fromg:
    raise Exception("Must specify FROM and TO gene number")

if opts.fromg >= len(genes):
    raise Exception("FROM is larger than number of genes")

if opts.tog > len(genes):
    opts.tog = len(genes)
    printStamp("Changing TO_gene number to max genes "+str(len(genes)))

printStamp("Actual number of calculated genes is "+str(opts.tog-opts.fromg))

for i, gene in enumerate(genes[opts.fromg:opts.tog]):

    printStamp("Learning for gene "+str(i))

    k = indices[i]

    # select only the cis-SNPs
    cismask = mfunc.select_snps(gene, snps, window)
    if len(cismask) > 0:
        target = expression[k, exprmask]
        target = scale(target, with_mean=True, with_std=True)
        predictor = gt[cismask][:, vcfmask]
        snpmask = cismask

        # if number of cis SNPs > threshold, use p-value cut-off
        if len(cismask) > min_snps:
            assoc_model = LinRegAssociation(predictor, target, min_snps, pval_cutoff)
            pvalmask = cismask[assoc_model.selected_variables]
            print ("Found {:d} SNPs, reduced to {:d} SNPs (max p-value {:g}) for {:s}".format(len(cismask), len(pvalmask), assoc_model.ordered_pvals[len(pvalmask) - 1], gene.name))
            predictor = gt[pvalmask][:, vcfmask]
            snpmask = pvalmask
        else:
            print ("Found {:d} SNPs for {:s}".format(len(cismask), gene.name))

        # perform the analysis
        
        print ("Starting first optimization ==============")
        emp_bayes = EmpiricalBayes(predictor, target, 1, init_params, method="new")
        emp_bayes.fit()
        if zmax > 1:
            if emp_bayes.success:
                res = emp_bayes.params
                print ("Starting second optimization from previous results ================")
                # Python Error: C library could not compute z-components. Check C errors above.
            else:
                res = init_params
                print ("Starting second optimization from initial parameters ================")
            emp_bayes = EmpiricalBayes(predictor, target, zmax, res, method="new")
            emp_bayes.fit()
            

        if emp_bayes.success:
            res = emp_bayes.params
            res[4] = 1 / np.sqrt(res[4])
            print('\n'.join(['{:g}'.format(x) for x in list(res)]))

            model_snps = [snps[x] for x in snpmask]
            model_zstates = list()
            scaledparams = hyperparameters.scale(emp_bayes.params)  # [pi, mu, sigma, sigmabg, tau]
            zprob, zexp = logmarglik.model_exp(scaledparams, predictor, target, emp_bayes.zstates)
            for j, z in enumerate(emp_bayes.zstates):
                this_zstate = ZstateInfo(state = z,
                                         prob  = zprob[j],
                                         exp   = list(zexp[j, :]) )
                model_zstates.append(this_zstate)
            model.write_success_gene(gene, model_snps, model_zstates, res)
            # model.write_params(gene, scaledparams)
        else:
            model.write_failed_gene(gene, np.zeros_like(init_params))
            print ("Failed optimization")
    else:
        print("No genotype for gene {:s}".format(gene.name))
