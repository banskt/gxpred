

import sys, os
sys.path.append("../")

import numpy as np
import math
import pickle
from config import *
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

gene_training_list = []
for i, gene in enumerate(genes):
    k = indices[i]
    if gene.ensembl_id in selected_gene_ids and gene.chrom == chrom:
        gene_training_list.append((k,gene))
        print(k,gene)



def select_trans_snps(snpinfo):
    SNP_COMPLEMENT = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}

    ''' Find indices of the genotype matrix
        corresponding to the cis SNPs of gene
    '''
    selected_snp = [(i,snp) for i, snp in enumerate(snpinfo) if max(len(snp.ref_allele), len(snp.alt_allele)) == 1]
    indices = [snp[0] for snp in selected_snp if snp[1].alt_allele != SNP_COMPLEMENT[snp[1].ref_allele] and
                                                 max(len(snp[1].ref_allele), len(snp[1].alt_allele)) == 1 and
                                                 snp[1].maf >= 0.1 and snp[1].maf <= 0.9]
    return np.array(indices)


# Set parameters and priors
# parameters = []
# parameters.append(["gxpred-bslmm", 0.1, 0.0, 0.1, 0.01, 0.005])
# # parameters.append(["gxpred-mg", 0.1, 0.0, 0.001, 0.01, 0.005])
# # parameters.append(["gxpred-mg-old", 0.1, 0.0, 0.001, 0.01, 0.005])

for p in parameters:

    print("\n\n###############")
    print("# Learning for new set of parameters")
    print(p)
    print("###############\n\n")
    prior = p[0]
    params = p[1]
    hyperpriors = p[2]
    hyperparams = p[3]
    run_description = p[4]
    init_params = np.array(params)
    # account for Tau inverse value
    init_params[4] = 1 / init_params[4] / init_params[4]

    model_dir = prior+"_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(params[0], params[1], params[2], params[3], params[4])
    modelpath = os.path.join("./z"+str(zmax), run_description, model_dir)
    if os.path.exists(modelpath):
        print("Results for current parameters exists! Continue..")
        continue
    write_params(modelpath, p)

    model = WriteModel(modelpath, chrom)

    for i in range(0,len(gene_training_list)):
        
        k, gene = gene_training_list[i]


        trans_snps_file = os.path.join("transeqtls", "geno_genes", "geno."+gene.ensembl_id.split(".")[0]+".snp.gz")
        if os.path.exists(trans_snps_file):
            print("Gene {:s} has transqtls!".format(gene.ensembl_id))
            # Load trans-eQTLs for gene, if exists
            trans_oxf = ReadOxford(trans_snps_file, gtex_samplepath, chrom, learning_dataset)
            trans_genotype = np.array(trans_oxf.dosage)
            trans_snps = trans_oxf.snps_info
            print ("Found {:d} trans SNPs".format(len(trans_snps)))
        else:
            print("No trans!")

       
        print("\n\n####### Learning New Gene ########")
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
                print ("Found {:d} SNPs, reduced to {:d} SNPs (max p-value {:g}) for {:s}".format(len(cismask), len(pvalmask), assoc_model.ordered_pvals[len(pvalmask) - 1], gene.name))
                predictor = gt[pvalmask][:, vcfmask]
                snpmask = pvalmask
            else:
                print ("Found {:d} SNPs for {:s}".format(len(cismask), gene.name))

            
            # add trans-eQTLs
            ix = select_trans_snps(trans_snps)
            transSNPs = trans_genotype[ix][:,vcfmask]

            ###### cis + trans SNPs #######
            predictor = np.concatenate((predictor, transSNPs), axis=0)


            ###### ONLY trans SNPs #######
            # predictor = transSNPs
            
            # perform the analysis

            print ("Starting first optimization ==============")
            emp_bayes = EmpiricalBayes(predictor, target, 1, init_params, method="new", prior=prior, hyperpriors= hyperpriors, hyperparams= hyperparams)
            emp_bayes.fit()
            if zmax > 1:
                if emp_bayes.success:
                    res = emp_bayes.params
                    print ("Starting second optimization from previous results ================")
                    # Python Error: C library could not compute z-components. Check C errors above.
                else:
                    res = init_params
                    print ("Starting second optimization from initial parameters ================")
                emp_bayes = EmpiricalBayes(predictor, target, zmax, res, method="new", prior=prior, hyperpriors= hyperpriors, hyperparams= hyperparams)
                emp_bayes.fit()

            if emp_bayes.success:
                res = emp_bayes.params
                res[4] = 1 / np.sqrt(res[4])
                print("PI: \t",res[0])
                print("mu: \t",res[1])
                print("sigma: \t",res[2])
                print("sigmabg: \t",res[3])
                print("tau: \t",res[4])

                ###### cis + trans SNPs #######
                model_snps = [f_snps[x] for x in snpmask] + list(trans_snps)

                ###### ONLY trans SNPs #######
                # trans_snps = list(trans_snps)
                # model_snps = [trans_snps[i] for i in ix]

                model_zstates = list()
                scaledparams = hyperparameters.scale(emp_bayes.params)
                zprob, zexp = logmarglik.model_exp(scaledparams, predictor, target, emp_bayes.zstates, prior)
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
