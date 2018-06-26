
import sys, os
import numpy as np
import math
import pickle
from iotools import snp_annotator
from iotools.io_model import WriteModel
from iotools.data import Data
from inference.empirical_bayes import EmpiricalBayes
from inference import logmarglik
from utils import hyperparameters
from utils.containers import ZstateInfo
from utils.printstamp import printStamp
from utils.helper_functions import write_params
from sklearn.preprocessing import scale
from collections import defaultdict
import gzip

from utils.args import Args
from utils.logs import MyLogger

args = Args()
logger = MyLogger(__name__)

data = Data(args)
data.load()

gene_info = data.geneinfo
genotype = data.dosage
samplenames = data.samplenames
snps = data.snpinfo
expression = data.expr_batch
genes = data.gene_batch

# args.write_args()



prior = "gxpred-bslmm"
init_params = [0.9, 0.0, 0.1, 0.1, 0.005]
run_description = "test_1KGannots_fixedPI"
hyperpriors = [None, None, None, None, None]
hyperparams = None #{"lambda":0.05, "Galpha":2, "Gbeta":0.5}
cutoff = "newsoft" #,"soft", "hard", "pval", "min"]
usedist = "nodist" #["dhs", "nodist", "random"]
usefeat = "nofeat" #["nofeat", "randomint"]

p = [prior, init_params, hyperpriors, hyperparams, run_description, cutoff, usedist, usefeat]
prior = p[0]
params = p[1]
hyperpriors = []
hyperparams = p[3]
run_description = p[4]
cutoff = p[5]
usedist = p[6]
usefeat = p[7]

model_dir = "{:s}_{:s}_{:s}_{:s}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(prior, cutoff, usedist, usefeat, params[0], params[1], params[2], params[3], params[4])
modelpath = os.path.join(args.outdir, "z"+str(args.cmax), run_description, model_dir)

logger.debug( "Writing to {:s}".format(modelpath))  

model = WriteModel(modelpath, args.chrom)

for i, gene in enumerate(data.gene_batch):

    logger.debug("Learning for gene "+gene.ensembl_id)

    target = data.expr_batch[i,:]
    target = scale(target, with_mean=True, with_std=True)

    # select only the cis-SNPs
    predictor = data.select_cis_snps(gene, target)
    if predictor.shape[0] > 0:        

        # if args.shuffle_geno:
        #     np.random.shuffle(predictor.T)

        selected_snps = data.cis_snps

        # if args.prune_LD:
        #     # later used for LD prunning, but needed before removing some cis-snps
        #     min_pos = f_snps[cismask[0]].bp_pos - 1000
        #     max_pos = f_snps[cismask[-1]].bp_pos + 1000

        #     ld_indices = snp_annotator.get_snps_LD(gene, selected_snps, min_pos, max_pos, args.genofile_plink, args.ldstorepath, args.ld_path)
        #     snpmask = np.delete(snpmask, np.reshape(ld_indices, -1))
        #     predictor = gt[snpmask][:, vcfmask]

        #     # replace with the pruned snsp in LD
        #     selected_snps = [f_snps[x] for x in snpmask]
            
        #     print ("Reduced to {:d} SNPs".format(len(snpmask)))

        # read the features
        features = data.load_annotations(usefeat)

        # Get DHS distance feature (this is different than the one above)
        dist_feature = snp_annotator.get_distance_feature(selected_snps, gene, usedist)

        nfeat = features.shape[1]
        print("Loaded {:d} features".format(nfeat))

        init_params = np.zeros(nfeat + 4)
        init_params[0] = - np.log((1 / params[0]) - 1)
        if nfeat > 1:
            for i in range(1, nfeat):
                init_params[i] = 0
        init_params[nfeat + 0] = params[1] # mu
        init_params[nfeat + 1] = params[2] # sigma
        init_params[nfeat + 2] = params[3] # sigmabg
        init_params[nfeat + 3] = 1 / params[4] / params[4] # tau

        print(init_params)

        # perform the analysis

        print ("Starting first optimization ==============")
        emp_bayes = EmpiricalBayes(predictor, target, features, dist_feature, args.cmax, init_params, method="new")
        emp_bayes.fit()
        if args.cmax > 1:
            if emp_bayes.success:
                res = emp_bayes.params
                print ("Starting second optimization from previous results ================")
                # Python Error: C library could not compute z-components. Check C errors above.
            else:
                res = init_params
                print ("Starting second optimization from initial parameters ================")
            emp_bayes = EmpiricalBayes(predictor, target, features, dist_feature, args.cmax, res, method="new")
            emp_bayes.fit()

        if emp_bayes.success:
            res = emp_bayes.params
            res[nfeat + 3] = 1 / np.sqrt(res[nfeat + 3])

            print(res)
            # print("PI: \t",res[0])
            # print("mu: \t",res[1])
            # print("sigma: \t",res[2])
            # print("sigmabg: \t",res[3])
            # print("tau: \t",res[4])

            model_snps = selected_snps
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
            model.write_failed_gene(gene, np.zeros_like(init_params))
            print ("Failed optimization")
    else:
        print("No cis SNPs found for {:s}".format(gene.ensembl_id))
