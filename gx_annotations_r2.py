
import sys
import os
from iotools import readgtf
from iotools.readrpkm import ReadRPKM
from iotools.readPrediction import ReadPrediction
from scipy.stats import pearsonr
from utils.helper_functions import write_r2_dataframe, get_common_elements, pearson_corr_rowwise
import math
import pickle
from utils.printstamp import printStamp

import config_annots as config
import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='Bayesian model for learning genetic contribution in gene expression')

    parser.add_argument('--dir',
                        type=str,
                        dest='dir',
                        metavar='DIR',
                        help='Name of the base model directory')

    opts = parser.parse_args()
    return opts

opts = parse_args()


# Load reference dataset Gene Expression
reference_rpkm = ReadRPKM(config.reference_expdatapath, config.predicting_dataset)
reference_expression = reference_rpkm.expression
reference_expr_donors = reference_rpkm.donor_ids
reference_gene_names = reference_rpkm.gene_names

# use the selected_gene_ids with high R² values as targets, only those in the selected chrom will appear
gene_info = readgtf.gencode_v12(config.gtfpath, trim=True, include_chrom=config.chrom)

target_genelist = [g.ensembl_id for g in gene_info]
target_donors = reference_expr_donors
print("There are {:d} genes in gencode".format(len(target_genelist)))


### Predixcan assessment ###

if not os.path.exists(config.predixcan_pickfile):
    predixcanpred = ReadPrediction(config.pxpred_predpath, config.reference_samplepath, "predixcan", trim=True)

    if len(predixcanpred.gene_names) > 0:
        printStamp("Dumping Predixcan prediction")
        with open(config.predixcan_pickfile, 'wb') as output:
            pickle.dump(predixcanpred, output, pickle.HIGHEST_PROTOCOL)
    else:
        raise("No prediction data found")
else:
    printStamp("Reading pickled Predixcan prediction")
    with open(config.predixcan_pickfile, 'rb') as input:
        predixcanpred = pickle.load(input)

# get genes for this chr available in the reference dataset
cardiobase_gene_names, ix_genes = get_common_elements(reference_gene_names, target_genelist)

# use previousle found genes to filter predixcan and gxpred
predixcanpred.sort_by_gene(cardiobase_gene_names)
predixcanpred.sort_by_samples(target_donors, use_prev=True)

# now sort samples according to predixcan
sorted_gene_names, ix_genes = get_common_elements(reference_gene_names, predixcanpred.sorted_gene_names)
sorted_expr_donors, ix_samples = get_common_elements(reference_expr_donors, predixcanpred.sorted_samples)
sorted_expression = reference_expression[ix_genes,:][:, ix_samples].T


predixcan_r = pearson_corr_rowwise(predixcanpred.sorted_expr_mat.T, sorted_expression.T)

### GXpred assessment ###

modelpath = opts.dir
print(modelpath)

gxpred_predpath = os.path.join(modelpath)
gxpred = ReadPrediction(gxpred_predpath, config.reference_samplepath, "gxpred", trim=True)

# filter gxpred predicted values
gxpred.sort_by_gene(cardiobase_gene_names)
gxpred.sort_by_samples(target_donors, use_prev=True)

# Filter and sort the reference expression values
# Cardiogenics variables
# expression
# expr_donors
# gene_names

sorted_expr_donors, ix_samples = get_common_elements(reference_expr_donors, gxpred.sorted_samples)
sorted_gene_names, ix_genes = get_common_elements(reference_gene_names, gxpred.sorted_gene_names)
sorted_expression = reference_expression[ix_genes,:][:, ix_samples].T

# Calculate Pearson correlation
gxpred_r = pearson_corr_rowwise(gxpred.sorted_expr_mat.T, sorted_expression.T)

write_r2_dataframe(modelpath, config.chrom, "predixcan", predixcan_r, predixcanpred, overwrite=True)
write_r2_dataframe(modelpath, config.chrom, "gxpred-bslmm", gxpred_r, gxpred)