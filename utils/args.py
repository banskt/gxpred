#!/usr/bin/env python

import os
import argparse
import logging
from utils.logs import MyLogger

def version():
    ## Get the version from version.py without importing the package
    vfn = os.path.join(os.path.dirname(__file__), '../version.py')
    exec(compile(open(vfn).read(), vfn, 'exec'))
    res = locals()['__version__']
    return res

class Error(Exception):
    pass

class SECTION_LARGER_THAN_BATCH(Error):
    pass

class OUTDIR_NOT_EMPTY(Error):
    pass


def batch_num(mstring):
    batchnum = mstring.split(":")
    split = int(batchnum[0].strip())
    section = int(batchnum[1].strip())
    if section != None and split != None and section > split and section > -1:
        raise SECTION_LARGER_THAN_BATCH("SECTION number cannot be greater than number of available batches to run (SPLIT)")
    batchlist = [split, section]
    return batchlist
    
def cutoff_strings(mstring):
    try:
        assert (mstring == 'soft' or mstring == 'hard')
    except AssertionError:
        raise argparse.ArgumentTypeError('Please specify a correct cutoff type ("hard" or "soft")')
    return mstring

def dataset_strings(mstring):
    try:
        assert (mstring == 'cardiogenics' or mstring == 'gtex' or mstring == 'geuvadis')
    except AssertionError:
        raise argparse.ArgumentTypeError('Please specify a correct dataset ("gtex" | "cardiogenix" | "geuvadis")')
    return mstring


class Args():

    def __init__(self):

        self.logger = MyLogger(__name__)
   
        args = None
        args = self.parse_args()

        self.vcf_file  = args.vcf_filename
        self.oxf_file  = args.oxf_filename
        self.fam_file  = args.fam_filename
        self.gx_file   = args.gx_filename
        self.gtf_file  = args.gtf_filename
        self.chrom     = args.chrom
        self.dataset   = args.dataset
        self.outprefix = args.outprefix
        self.split     = args.batchnum[0]
        self.section   = args.batchnum[1]
        self.window    = 1000000
        self.cmax      = args.cmax
        self.min_snps  = args.min_snps
        self.pval_cutoff = args.pval_cutoff
        self.cutoff    = args.cutoff
        self.outdir    = args.outdir

    #TODO
    def write_args(self):
        if os.path.exists(self.outdir) and not os.listdir(self.outdir):
            raise OUTDIR_NOT_EMPTY("Destination folder is not empty.")
        else:
            os.makedirs(self.outdir)
        with open(os.path.join(self.outdir, "params.txt"), 'w') as outstream:
            headers = ["Prior","pi","mu","sigma","sigmabg","tau","pi_prior","mu_prior","sigma_prior","sigmabg_prior","tau_prior"]
            dict_values = []
            if params[3] != None:
                for i in params[3].keys():
                    headers.append(i)
                    dict_values.append(str(params[3][i]))
            data = [params[0]] + [str(item) for sublist in params[1:3] for item in sublist] + dict_values
            outdict = dict(zip(headers, data))
            for i in outdict:
                outstream.write(i+"\t"+outdict[i]+"\n")

    def parse_args(self):

        self.logger.info('Running GXpred')

        parser = argparse.ArgumentParser(description='Bayesian model for learning genetic contribution in gene expression')

        parser.add_argument('--out',
                            type=str,
                            dest='outdir',
                            metavar='DIR',
                            help='Name of the output directory for storing the model')

        parser.add_argument('--batch-section',
                            type=batch_num,
                            dest='batchnum',
                            metavar='BATCH:NUM',
                            help='Splits calculations in BATCH sections, and runs section number NUM')

        parser.add_argument('--chr',
                            type=int,
                            dest='chrom',
                            default=-1,
                            metavar='CHROM',
                            help='Chromosome number to use')

        # parser.add_argument('--dist',
        #                     type=str,
        #                     dest='dist',
        #                     metavar='DIST',
        #                     help='Include distance feature or not')

        parser.add_argument('--vcf',
                            type=str,
                            dest='vcf_filename',
                            metavar='FILE',
                            help='input VCF file')

        parser.add_argument('--oxf',
                            type=str,
                            dest='oxf_filename',
                            metavar='FILE',
                            help='input Oxford file')
    
        parser.add_argument('--fam',
                            type=str,
                            dest='fam_filename',
                            metavar='FILE',
                            help='input fam file')
    
        parser.add_argument('--gx',
                            type=str,
                            dest='gx_filename',
                            metavar='FILE',
                            help='input expression file')

        parser.add_argument('--gtf',
                            type=str,
                            dest='gtf_filename',
                            metavar='FILE',
                            help='input gtf file')

        parser.add_argument('--dataset',
                            type=dataset_strings,
                            dest='dataset',
                            metavar='DATASET',
                            help='Dataset format to use ( "gtex" | "cardiogenix" | "geuvadis" )')
        
        parser.add_argument('--outprefix',
                            type=str,
                            dest='outprefix',
                            metavar='OUTPREFIX',
                            help='prefix for all output files')

        parser.add_argument('--cmax',
                            type=int,
                            default=1,
                            dest='cmax',
                            metavar='CMAX',
                            help='number of maximum causal SNPs to consider')

        parser.add_argument('--min-snps',
                            type=int,
                            default=20,
                            dest='min_snps',
                            metavar='MIN_SNPS',
                            help='number of minimum SNPs to train the model for each gene')

        parser.add_argument('--pvalcutoff',
                            type=int,
                            default=0.001,
                            dest='pval_cutoff',
                            metavar='PVAL_CUTOFF',
                            help='P-value threshold for pre-selection of SNPs (by linear association)')

        parser.add_argument('--cutoff',
                            type=cutoff_strings,
                            default='soft',
                            dest='cutoff',
                            metavar='CUTOFF',
                            help='Type of cutoff for pre-selection of SNPS [HARD | SOFT]')

        res = parser.parse_args()
        return res
