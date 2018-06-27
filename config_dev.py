
import os

# home = '/usr/users/fsimone'
home = '/home/franco/cluster2'
gtfpath = os.path.join(home,"datasets/gtex/gencode.v19.annotation.gtf.gz")
chrom = 12

############################
#
# General Settings
#
############################
# run_description: name of the folder to store results for this combination of parameters
# Init_params: initial parameters for [pi, mu, sigma, sigmabg, pi]
# Prior: prior used for causal and non-causal SNPs (sigma and sigmabg) [gxpred-mg, gxpred-mg-old, gxpred-bslmm]
# Hyperpriors: one for each parameter [pi, mu, sigma, sigmabg, pi]
#               None: no prior
#               L1: L1 reg prior
#               S2:
# Hyperparams: dictionary with parameters needed by the used hyperpriors (lambda, alpha, etc)


parameters = []
prior = "gxpred-bslmm"

# set_init_params = [[0.1, 0.0, 0.1, 0.1, 0.005]]
# set_init_params = [[0.1, 0.0, 0.1, 0.001, 0.005]]
set_init_params = [[0.9, 0.0, 0.1, 0.1, 0.005]]

#### Set 9
run_description = "test_gotohell"
shuffle_geno = False

hyperpriors = [None, None, None, None, None]
hyperparams = None #{"lambda":0.05, "Galpha":2, "Gbeta":0.5}
cutoffs = ["soft"] #,"soft", "hard", "pval", "min"]
usedists = ["nodist"] #["dhs", "nodist", "random"]
usefeats = ["nofeat"] #["nofeat", "randomint", "1kg"]
for init_params in set_init_params:
    for cutoff in cutoffs:
        for usedist in usedists:
            for usefeat in usefeats:
                parameters.append([prior, init_params, hyperpriors, hyperparams, run_description, cutoff, usedist, usefeat])


############################
#
# Learning Settings
#
############################
learning_dataset = "gtex"
gtex_samplepath = os.path.join(home, "datasets/gtex/donor_ids.fam")
gtex_gtpath = os.path.join(home, "datasets/gtex/GTEx_450Indiv_genot_imput_info04_maf01_HWEp1E6_dbSNP135IDs_donorIDs_dosage_chr"+str(chrom)+".gz")
# gtex_rpkmpath = os.path.join(home, "datasets/gtex/gtex_wholeblood_normalized.expression.txt")
gtex_rpkmpath = os.path.join(home, "datasets/gtex/gtex_wholeblood_normalized.lm_corr.exp.klinikum.txt")
# gtex_rpkmpath = os.path.join(home, "datasets/gtex/gtex_wholeblood_normalized.lm_corr_final2.exp.txt")
learn_pickfile_dev = os.path.join("/home/franco", "cbscratch/datasets","GTEx_v6p_chr"+str(chrom)+".pkl")
# learn_pickfile_dev = os.path.join("/cbscratch/franco/datasets","GTEx_v6p_chr"+str(chrom)+".pkl")

min_snps = 200
pval_cutoff = 0.001
window = 1000000
zmax = 1    # z parameter

annot1kg_dir = os.path.join(home, "datasets/1KG_annots")

############################
#
# LD Settings
#
############################

prune_LD = False
ld_path = "/home/franco/cbscratch/datasets/ldscores"
genofile_plink = "/home/franco/cbscratch/gtex_genotype_pipeline/genotype_split_by_chr/GTEx_450Indiv_chr"+str(chrom)+"_genot_imput_info04_maf01_HWEp1E6_ConstrVarIDs_donorIDs"
ldstorepath = "/home/franco/bin/ldstore"


############################
#
# Prediction Settings
#
############################

predicting_dataset="cardiogenics"
reference_home = os.path.join(home,"datasets/cardiogenics/")
p_gtpath=os.path.join(reference_home, "genotypes/CG_"+str(chrom)+".imputed.gz")
p_samplepath=os.path.join(reference_home, "genotypes/CG.sample")
p_pickfile = os.path.join("/cbscratch/franco/datasets","CG_"+str(chrom)+".pkl")
p_pickfile_dev = os.path.join("/home/franco", "cbscratch/datasets","CG_"+str(chrom)+".pkl")


# Load reference expression data for cardiogenics
reference_expdatapath = os.path.join(reference_home, "expression/lumi_mono_all_vsn_batches_adjusted.RData.exp.txt")
reference_samplepath=os.path.join(reference_home, "genotypes/CG.sample")


# Predixcan prediction path
pred_basedir = os.path.join("/home/franco/cbscratch/datasets")
pxpred_predpath = os.path.join(pred_basedir, "cardiogenics/predixcan_predictions_klinikum")
# pxpred_predpath = os.path.join(pred_basedir, "cardiogenics/predixcan_predictions_klinikum_lmcorr_expr_random")
predixcan_pickfile = os.path.join(pred_basedir,"Predixcan.pkl")
predixcan_pickfile_dev = os.path.join("/home/franco/cbscratch/datasets/","Predixcan.pkl")
# predixcan_pickfile = os.path.join(pred_basedir,"Predixcan_random.pkl")

# pxpred_predpath = os.path.join(pred_basedir, "cardiogenics/predixcan_predictions_klinikum_lmcorr_expr")
# predixcan_pickfile = os.path.join(pred_basedir,"Predixcan_lmcorr.pkl")
