
import sys, os
sys.path.append("../")

import numpy as np
import math
import pickle
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
from helper_functions import load_target_genes, write_params
from sklearn.preprocessing import scale
from iotools import snp_annotator


from config_annots import *

# Annotation (use complete gene name in gtf without trimming the version)
# load annotation for whole genome
gene_info = readgtf.gencode_v12(gtfpath, trim=False)

# Load gene list
selected_gene_ids = load_target_genes(genelistfile, gene_info, chrom)


if not os.path.exists(learn_pickfile):
    # read Genotype
    oxf = ReadOxford(gtex_gtpath, gtex_samplepath, chrom, learning_dataset)
    genotype = np.array(oxf.dosage)
    samplenames = oxf.samplenames
    snps = oxf.snps_info

    printStamp("Dumping CHR {:d} genotype".format(chrom))
    with open(learn_pickfile, 'wb') as output:
        pickle.dump(oxf, output, pickle.HIGHEST_PROTOCOL)
else:
    printStamp("Reading pickled genotype")
    with open(learn_pickfile, 'rb') as input:
        pickled_oxf = pickle.load(input)

    printStamp("Done reading")
    genotype = np.array(pickled_oxf.dosage)
    samplenames = pickled_oxf.samplenames
    snps = pickled_oxf.snps_info
    nsample = len(pickled_oxf.samplenames)

# Quality control
f_snps, f_genotype = gtutils.remove_low_maf(snps, genotype, 0.1)
gt = gtutils.normalize(f_snps, f_genotype)

# Gene Expression
rpkm = ReadRPKM(gtex_rpkmpath, "gtex")
expression = rpkm.expression
expr_donors = rpkm.donor_ids
gene_names = rpkm.gene_names

# Selection
printStamp("Selection of samples")
vcfmask, exprmask = mfunc.select_donors(samplenames, expr_donors)
genes, indices = mfunc.select_genes(gene_info, gene_names)


# obtain from the list of genes, the indices in the gene-expression
gene_training_list = []
for i, gene in enumerate(genes):
	k = indices[i]
	if gene.ensembl_id in selected_gene_ids and gene.chrom == chrom:
		gene_training_list.append((k,gene))


for p in parameters:
	prior = p[0]
	params = p[1]

	print(params)

	hyperpriors = []
	hyperparams = p[3]
	run_description = p[4]


	model_dir = prior+"_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(params[0], params[1], params[2], params[3], params[4])
	modelpath = os.path.join("./z"+str(zmax), run_description, model_dir)

	write_params(modelpath, p)

	model = WriteModel(modelpath, chrom)

	for i in range(0,len(gene_training_list)):

		k, gene = gene_training_list[i]

		print(k, gene)

		# select only the cis-SNPs
		cismask = mfunc.select_snps(gene, f_snps, window)
		if len(cismask) > 0:
			target = expression[k, exprmask]
			target = scale(target, with_mean=True, with_std=True)
			predictor = gt[cismask][:, vcfmask]
			snpmask = cismask

			# if number of cis SNPs > threshold, use p-value cut-off

			if len(cismask) > min_snps:
				assoc_model = LinRegAssociation(predictor, target, min_snps, pval_cutoff)
				pvalmask = cismask[assoc_model.selected_variables]
				if pvalmask.shape[0] == 0:
					print("No significant SNPs found for gene {:s}".format(gene.ensembl_id))
					continue
				print ("Found {:d} SNPs, reduced to {:d} SNPs (max p-value {:g}) for {:s}".format(len(cismask), len(pvalmask), assoc_model.ordered_pvals[len(pvalmask) - 1], gene.name))
				predictor = gt[pvalmask][:, vcfmask]
				snpmask = pvalmask
			else:
				print ("Found {:d} SNPs for {:s}".format(len(cismask), gene.name))

			nsnps_used = len(snpmask)
			selected_snps = [f_snps[x] for x in snpmask]

			# read the features
			# TO-DO: call another module for getting the features
			# for now, it contains only a list of 1's
			feature1 = np.ones((predictor.shape[0], 1))
			# feature2 = snp_annotator.get_dummy_dist_feature(selected_snps, gene, window)

			# Get DHS distance feature
			# dist_feature = np.ones(len(selected_snps))
			dist_feature = snp_annotator.get_dist_feature(selected_snps, gene, window)

			# features = np.concatenate((feature1,feature2), axis=1)
			features = feature1

			nfeat = features.shape[1]
			print("Loaded {:d} features".format(nfeat))


			init_params = np.zeros(nfeat + 4)
			init_params[0] = - np.log((1 / params[0]) - 1)
			for i in range(1, nfeat):
				init_params[i] = 0
			init_params[nfeat + 0] = params[1] # mu
			init_params[nfeat + 1] = params[2] # sigma
			init_params[nfeat + 2] = params[3] # sigmabg
			init_params[nfeat + 3] = 1 / params[4] / params[4] # tau

			# perform the analysis

			print ("Starting first optimization ==============")
			emp_bayes = EmpiricalBayes(predictor, target, features, dist_feature, 1, init_params, method="new")
			emp_bayes.fit()
			if zmax > 1:
				if emp_bayes.success:
					res = emp_bayes.params
					print ("Starting second optimization from previous results ================")
					# Python Error: C library could not compute z-components. Check C errors above.
				else:
					res = init_params
					print ("Starting second optimization from initial parameters ================")
				emp_bayes = EmpiricalBayes(predictor, target, features, dist_feature, zmax, res, method="new")
				emp_bayes.fit()

			if emp_bayes.success:
				res = emp_bayes.params
	            res[4] = 1 / np.sqrt(res[4])


	            print("PI: \t",res[0])
	            print("mu: \t",res[1])
	            print("sigma: \t",res[2])
	            print("sigmabg: \t",res[3])
	            print("tau: \t",res[4])

				model_snps = [f_snps[x] for x in snpmask]
				model_zstates = list()
				scaledparams = hyperparameters.scale(emp_bayes.params)
				zprob, zexp = logmarglik.model_exp(scaledparams, predictor, target, features, dist_feature, emp_bayes.zstates)
				for j, z in enumerate(emp_bayes.zstates):
					this_zstate = ZstateInfo(state = z,
											 prob  = zprob[j],
											 exp   = list(zexp[j, :]) )
					model_zstates.append(this_zstate)
				# print(model_snps)
				# for i,m in enumerate(model_zstates):
				#     print("z-state: ",i," Prob:", m.prob)
				model.write_success_gene(gene, model_snps, model_zstates, res)
			else:
				model.write_failed_gene(gene, res) # np.zeros_like(init_params)
				print ("Failed optimization")