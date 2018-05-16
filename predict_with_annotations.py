import sys
import os
import pickle
from utils.printstamp import printStamp
from iotools.io_model import ReadModel
from iotools.readOxford import ReadOxford
from utils.containers import GeneExpressionArray
from utils import gtutils
from utils import mfunc
import numpy as np
import argparse
# import config_annots as config


def parse_args():

    parser = argparse.ArgumentParser(description='Bayesian model for learning genetic contribution in gene expression')

    parser.add_argument('--dir',
                        type=str,
                        dest='dir',
                        metavar='DIR',
                        help='Name of the base model directory')

    parser.add_argument('--chr',
                        type=int,
                        dest='chrom',
                        default=-1,
                        metavar='CHROM',
                        help='Chromosome number to use')

    parser.add_argument('--config',
                        type=str,
                        dest='config_file',
                        default="none",
                        metavar='CONFIG',
                        help='Config file to load')

    opts = parser.parse_args()
    return opts

opts = parse_args()


if opts.config_file != "none":
    import importlib
    config = importlib.import_module(opts.config_file)
else:
    import config_annots as config

if opts.chrom < 0:
    opts.chrom = config.chrom



if opts.config_file != "none":
    p_gtpath=os.path.join(config.reference_home, "genotypes/CG_"+str(opts.chrom)+".imputed.gz")
    p_oxf = ReadOxford(p_gtpath, config.p_samplepath, opts.chrom, config.predicting_dataset)
    p_genotype = np.array(p_oxf.dosage)
    p_samplenames = p_oxf.samplenames
    p_snps = p_oxf.snps_info
    p_nsample = len(p_oxf.samplenames)
else:
    if not os.path.exists(config.p_pickfile):
        # Read genotype (quite slow for testing) use pickle below
        p_oxf = ReadOxford(config.p_gtpath, config.p_samplepath, opts.chrom, config.predicting_dataset)
        p_genotype = np.array(p_oxf.dosage)
        p_samplenames = p_oxf.samplenames
        p_snps = p_oxf.snps_info
        p_nsample = len(p_oxf.samplenames)

        printStamp("Dumping CHR {:d} genotype".format(chrom))
        with open(config.p_pickfile, 'wb') as output:
            pickle.dump(p_oxf, output, pickle.HIGHEST_PROTOCOL)
    else:
        printStamp("Reading pickled genotype")
        with open(config.p_pickfile, 'rb') as input:
            pickled_oxf = pickle.load(input)

        printStamp("Done reading")

        p_genotype = np.array(pickled_oxf.dosage)
        p_samplenames = pickled_oxf.samplenames
        p_snps = pickled_oxf.snps_info
        p_nsample = len(pickled_oxf.samplenames)


modelpath = opts.dir

outfileprefix = os.path.join(modelpath,"pred_chr"+str(opts.chrom))

printStamp("Predicting for "+modelpath)
# Write predictions for each model
p_model = ReadModel(modelpath, opts.chrom)
p_genes = p_model.genes
gx = list()
for gene in p_genes:

    p_model.read_gene(gene)
    p_model_snps = p_model.snps
    p_model_zstates = p_model.zstates

    x = gtutils.prediction_variables(p_snps, p_model_snps, p_genotype)
    x = gtutils.normalize(p_model_snps, x)

    ypred = np.zeros(p_nsample)
    for z in p_model_zstates:
        ypred += z.prob * np.dot(x.T, z.exp)

    gx.append(GeneExpressionArray(geneid = gene.ensembl_id, expr_arr = ypred))


# Write output
printStamp("Done predicting for "+modelpath)
mfunc.write_gcta_phenotype(outfileprefix, p_samplenames, gx)