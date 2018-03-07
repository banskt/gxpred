import sys
sys.path.append("../")
import os
from iotools import readgtf
from iotools.readrpkm import ReadRPKM
from iotools.readPrediction import ReadPrediction
from scipy.stats import pearsonr
from helper_functions import load_target_genes, write_r2_dataframe, get_common_elements, new_write_predicted_r2, pearson_corr_rowwise
import math
import pickle
from utils.printstamp import printStamp

from config import *


# load annotation for whole genome
gene_info = readgtf.gencode_v12(gtfpath, trim=False)

# Load reference dataset Gene Expression
reference_rpkm = ReadRPKM(reference_expdatapath, "cardiogenics")
reference_expression = reference_rpkm.expression
reference_expr_donors = reference_rpkm.donor_ids
reference_gene_names = reference_rpkm.gene_names

# use the selected_gene_ids with high R² values as targets, only those in the selected chrom will appear
selected_gene_ids = load_target_genes(genelistfile, gene_info, chrom)
target_genelist = [g.split(".")[0] for g in selected_gene_ids]
target_donors = reference_expr_donors


### Predixcan assessment ###

if not os.path.exists(predixcan_pickfile):
	# pxpred_predpath = os.path.join(home, "predictions/cardiogenics/predixcan_predictions_klinikum")
	predixcanpred = ReadPrediction(pxpred_predpath, reference_samplepath, "predixcan", trim=True)

	printStamp("Dumping Predixcan prediction")
	with open(predixcan_pickfile, 'wb') as output:
	    pickle.dump(predixcanpred, output, pickle.HIGHEST_PROTOCOL)
else:
	printStamp("Reading pickled Predixcan prediction")
	with open(predixcan_pickfile, 'rb') as input:
	    predixcanpred = pickle.load(input)

# filter predixcan predictions with only those in gxpred
predixcanpred.sort_by_gene(target_genelist)
predixcanpred.sort_by_samples(target_donors, use_prev=True)

sorted_expr_donors, ix_samples = get_common_elements(reference_expr_donors, predixcanpred.sorted_samples)
sorted_gene_names, ix_genes = get_common_elements(reference_gene_names, predixcanpred.sorted_gene_names)
sorted_expression = reference_expression[ix_genes,:][:, ix_samples].T

predixcan_r = pearson_corr_rowwise(predixcanpred.sorted_expr_mat.T, sorted_expression.T)


### GXpred assessment ###

for p in parameters:

	prior = p[0]
	params = p[1]
	hyperpriors = p[2]
	hyperparams = p[3]
	run_description = p[4]
	model_dir = prior+"_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(params[0], params[1], params[2], params[3], params[4])
	print(prior, params)

	modelpath = os.path.join("./z"+str(zmax), run_description, model_dir)

	gxpred_predpath = os.path.join(modelpath)
	gxpred = ReadPrediction(gxpred_predpath, reference_samplepath, "gxpred", trim=True)

	# filter gxpred predicted values
	gxpred.sort_by_gene(target_genelist)
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

	write_r2_dataframe(modelpath, chrom, "predixcan", predixcan_r, predixcanpred)
	write_r2_dataframe(modelpath, chrom, prior, gxpred_r, gxpred)