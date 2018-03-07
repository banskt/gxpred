
import os

home = '/home/fsimone'
pred_path=os.path.join(home,"gxpred/devtools")
gtfpath = os.path.join(home,"datasets/gtex/gencode.v19.annotation.gtf.gz")
# gtfpath = os.path.join(home,"gxpred/devtools/gencode_crop.gtf.gz")
chrom = 12

# genelistfile = "target_gene_list" # r2 > 0.1 and px > 2*gx
# genelistfile = "genes4testing_highr2" # r2 > 0.1 and r2 < predixcan_r2
genelistfile = "genes4testing_high_and_low_r2" # r2 > 0.05


############################
#
# General Settings
#
############################
# run_description: name of the folder to store results for this combination of parameters
# Init_params: initial parameters for [pi, mu, sigma, sigmabg, pi]
# Prior: prior used for causal and non-causal SNPs (sigma and sigmabg) [gxpred-mg, gxpred-mg-old, gxpred-bslmm]
# Hyperpriors: one for each parameter [pi, mu, sigma, sigmabg, pi]
#				None: no prior
#				L1: L1 reg prior
# 				S2:
# Hyperparams: dictionary with parameters needed by the used hyperpriors (lambda, alpha, etc)


run_description = "unbound_L1_0.1"
prior = "gxpred-bslmm"
# bounds = [[None, None] for i in range(5)]  ## not implemented
hyperpriors = [None, None, None, "L1", None]
hyperparams = {"lambda":0.1}
init_params = [0.005, 0.0, 0.1, 0.001, 0.005]
parameters = []
parameters.append([prior, init_params, hyperpriors, hyperparams, run_description])

############################
#
# Learning Settings
#
############################
learning_dataset = "gtex"
gtex_samplepath = os.path.join(home, "datasets/gtex/donor_ids.fam")
gtex_gtpath = os.path.join(home, "datasets/gtex/GTEx_450Indiv_genot_imput_info04_maf01_HWEp1E6_dbSNP135IDs_donorIDs_dosage_chr"+str(chrom)+".gz")
# gtex_rpkmpath = os.path.join(home, "datasets/gtex/gtex_wholeblood_normalized.expression.txt")
gtex_rpkmpath = os.path.join(home, "datasets/gtex/gtex_wholeblood_normalized.lm_corr_final2.exp.txt")

learn_pickfile = "GTEx_v6p_chr"+str(chrom)+".pkl"

min_snps = 200
pval_cutoff = 0.001
window = 1000000
zmax = 1    # z parameter


############################
#
# Prediction Settings
#
############################

predicting_dataset="cardiogenics"

reference_home = os.path.join(home,"datasets/cardiogenics/")
p_gtpath=os.path.join(reference_home, "genotypes/CG_"+str(chrom)+".imputed.gz")
p_samplepath=os.path.join(reference_home, "genotypes/CG.sample")
p_pickfile = "CG_"+str(chrom)+".pkl"


predixcan_pickfile = "Predixcan.pkl"


# Load reference expression data for cardiogenics
reference_expdatapath = os.path.join(reference_home, "expression/lumi_mono_all_vsn_batches_adjusted.RData.exp.txt")
reference_samplepath=os.path.join(reference_home, "genotypes/CG.sample")


# Predixcan prediction path
pxpred_predpath = os.path.join(home, "predictions/cardiogenics/predixcan_predictions_klinikum")
