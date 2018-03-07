
import sys
sys.path.append("../")
import os
import pickle
from utils.printstamp import printStamp
from iotools.io_model import ReadModel
from utils.containers import GeneExpressionArray
from utils import gtutils
from utils import mfunc
import numpy as np
from config import *


if not os.path.exists(p_pickfile):
# Read genotype (quite slow for testing) use pickle below
	p_oxf = ReadOxford(p_gtpath, p_samplepath, chrom, dataset)
	p_genotype = np.array(p_oxf.dosage)
	p_samplenames = p_oxf.samplenames
	p_snps = p_oxf.snps_info
	p_nsample = len(p_oxf.samplenames)

	printStamp("Dumping CHR {:d} genotype".format(chrom))
	with open(p_pickfile, 'wb') as output:
	    pickle.dump(p_oxf, output, pickle.HIGHEST_PROTOCOL)
else:
	printStamp("Reading pickled genotype")
	with open(p_pickfile, 'rb') as input:
	    pickled_oxf = pickle.load(input)
    
	printStamp("Done reading")

	p_genotype = np.array(pickled_oxf.dosage)
	p_samplenames = pickled_oxf.samplenames
	p_snps = pickled_oxf.snps_info
	p_nsample = len(pickled_oxf.samplenames)


# Use parameters from config.py
for p in parameters:

	prior = p[0]
	params = p[1]
	hyperpriors = p[2]
	hyperparams = p[3]
	run_description = p[4]
	model_dir = prior+"_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(params[0], params[1], params[2], params[3], params[4])
	modelpath = os.path.join("./z"+str(zmax), run_description, model_dir)

	outfileprefix = os.path.join(modelpath,"pred_chr"+str(chrom))


	printStamp("Predicting for "+model_dir)
	# Write predictions for each model
	p_model = ReadModel(modelpath, chrom)
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
	mfunc.write_gcta_phenotype(outfileprefix, p_samplenames, gx)
