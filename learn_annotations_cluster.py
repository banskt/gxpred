
import sys, os
import argparse
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
from utils.helper_functions import write_params
from sklearn.preprocessing import scale
from iotools import snp_annotator
from collections import defaultdict
import gzip

# import config_annots as config


def parse_args():

    parser = argparse.ArgumentParser(description='Bayesian model for learning genetic contribution in gene expression')

    parser.add_argument('--out',
                        type=str,
                        dest='outdir',
                        metavar='DIR',
                        help='Name of the output directory for storing the model')

    parser.add_argument('--split',
                        type=int,
                        dest='split',
                        metavar='SPLIT',
                        help='Split the genes in SPLIT batches.')

    parser.add_argument('--section',
                        type=int,
                        dest='section',
                        metavar='SECTION',
                        help='Selects which batch to run, must be between 0 and SPLIT-1')

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

    # parser.add_argument('--dist',
    #                     type=str,
    #                     dest='dist',
    #                     metavar='DIST',
    #                     help='Include distance feature or not')

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

if opts.section != None and opts.split != None and opts.section > opts.split:
    raise Exception("SECTION number cannot be greater than number of available batches to run (SPLIT)")


# Annotation (use complete gene name in gtf without trimming the version)
# load annotation for whole genome
gene_info = readgtf.gencode_v12(config.gtfpath, include_chrom = opts.chrom, trim=False)

# read Genotype

if opts.config_file != "none":
    gtex_gtpath = os.path.join(config.home, "datasets/gtex/GTEx_450Indiv_genot_imput_info04_maf01_HWEp1E6_dbSNP135IDs_donorIDs_dosage_chr"+str(opts.chrom)+".gz")
    oxf = ReadOxford(gtex_gtpath, config.gtex_samplepath, opts.chrom, config.learning_dataset)
else:
    oxf = ReadOxford(config.gtex_gtpath, config.gtex_samplepath, opts.chrom, config.learning_dataset)
genotype = np.array(oxf.dosage)
samplenames = oxf.samplenames
snps = oxf.snps_info


# Quality control
f_snps, f_genotype = gtutils.remove_low_maf(snps, genotype, 0.1)
gt = gtutils.normalize(f_snps, f_genotype)

# Gene Expression
rpkm = ReadRPKM(config.gtex_rpkmpath, config.learning_dataset)
expression = rpkm.expression
expr_donors = rpkm.donor_ids
gene_names = rpkm.gene_names

# Selection
printStamp("Selection of samples")
vcfmask, exprmask = mfunc.select_donors(samplenames, expr_donors)
genes, indices = mfunc.select_genes(gene_info, gene_names)


batch_size = None
gene_number = len(genes)

if opts.split and opts.split > 1:
    if gene_number > opts.split:
        print("Gene number: ", gene_number)
        batch_size = math.ceil(gene_number / opts.split)
        print("Splitting in batches of ", batch_size)
    else:
        raise Exception("Split number is greater than number of genes. Cannot split the job")




for p in config.parameters:
    prior = p[0]
    params = p[1]
    hyperpriors = []
    hyperparams = p[3]
    run_description = p[4]
    cutoff = p[5]
    usedist = p[6]
    usefeat = p[7]

    model_dir = "{:s}_{:s}_{:s}_{:s}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(prior, cutoff, usedist, usefeat, params[0], params[1], params[2], params[3], params[4])
    modelpath = os.path.join(opts.outdir, "z"+str(config.zmax), config.run_description, model_dir)

    print(modelpath)

    write_params(modelpath, p)

    model = WriteModel(modelpath, opts.chrom)

    # Load rsid dictionary
    annot_dict = defaultdict(list)
    if usefeat == "1kg":
        annotfile = os.path.join(config.annot1kg_dir, "1KG."+str(opts.chrom)+".annot.gz")
        print(annotfile)
        with gzip.open(annotfile, 'r') as instream:
            _ = instream.readline()
            for line in instream:
                arr = line.decode().strip().split(" ")
                rsid = arr[0]
                annots = list(map(int, arr[1:]))
                annot_dict[rsid] = annots

    for i, gene in enumerate(genes):

        # if gene number is outside of the range, do not calculate and continue
        if opts.split and opts.section >= 0 and (i < batch_size*opts.section or i >= (batch_size*opts.section + batch_size)):
            continue

        k = indices[i]

        printStamp("Learning for gene "+str(gene.ensembl_id))

        # select only the cis-SNPs
        cismask = mfunc.select_snps(gene, f_snps, config.window)
        if len(cismask) > 0:
            target = expression[k, exprmask]
            target = scale(target, with_mean=True, with_std=True)
            predictor = gt[cismask][:, vcfmask]
            snpmask = cismask

            # later used for LD prunning, but needed before removing some cis-snps
            min_pos = f_snps[cismask[0]].bp_pos - 1000
            max_pos = f_snps[cismask[-1]].bp_pos + 1000

            # if number of cis SNPs > threshold, use p-value cut-off          
            if len(cismask) > config.min_snps:
                assoc_model = LinRegAssociation(predictor, target, config.min_snps, config.pval_cutoff, cutoff)
                pvalmask = cismask[assoc_model.selected_variables]
                if pvalmask.shape[0] == 0:
                    print("No significant SNPs found for gene {:s}".format(gene.ensembl_id))
                    continue
                print ("Found {:d} SNPs, reduced to {:d} SNPs (max p-value {:g}) for {:s}".format(len(cismask), len(pvalmask), assoc_model.ordered_pvals[len(pvalmask) - 1], gene.name))
                predictor = gt[pvalmask][:, vcfmask]
                snpmask = pvalmask
            else:
                print ("Found {:d} SNPs for {:s}".format(len(cismask), gene.name))

            if config.shuffle_geno:
                np.random.shuffle(predictor.T)

            selected_snps = [f_snps[x] for x in snpmask]

            if config.prune_LD:
                ld_indices = snp_annotator.get_snps_LD(gene, selected_snps, min_pos, max_pos, config.genofile_plink, config.ldstorepath, config.ld_path)
                snpmask = np.delete(snpmask, np.reshape(ld_indices, -1))
                predictor = gt[snpmask][:, vcfmask]

                # replace with the pruned snsp in LD
                selected_snps = [f_snps[x] for x in snpmask]
                
                print ("Reduced to {:d} SNPs".format(len(snpmask)))

            # read the features
            if usefeat == "1kg":
                nsnps_used = len(selected_snps)
                feature0 = np.ones((nsnps_used, 1))
                
                current_annot = list()
                for snp in selected_snps:
                    if len(annot_dict[snp.varid]) > 0:
                        current_annot.append(annot_dict[snp.varid])
                    else:
                        current_annot.append([0,0,0,0,0])
                        print("not found {:s}!".format(snp.varid))
                feature1kg = np.array(current_annot)

                features = np.concatenate((feature0, feature1kg), axis=1)
            else:
                features = snp_annotator.get_features(selected_snps, usefeat)

            # Get DHS distance feature
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
            emp_bayes = EmpiricalBayes(predictor, target, features, dist_feature, 1, init_params, method="new")
            emp_bayes.fit()
            if config.zmax > 1:
                if emp_bayes.success:
                    res = emp_bayes.params
                    print ("Starting second optimization from previous results ================")
                    # Python Error: C library could not compute z-components. Check C errors above.
                else:
                    res = init_params
                    print ("Starting second optimization from initial parameters ================")
                emp_bayes = EmpiricalBayes(predictor, target, features, dist_feature, config.zmax, res, method="new")
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
                model.write_failed_gene(gene, np.zeros_like(init_params))
                print ("Failed optimization")
        else:
            print("No cis SNPs found for {:s}".format(gene.ensembl_id))
