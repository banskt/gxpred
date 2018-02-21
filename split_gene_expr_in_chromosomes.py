
#!/usr/bin/env python

import argparse
import numpy as np

import os
from iotools.readrpkm import ReadRPKM
from iotools.io_model import WriteModel
from utils import hyperparameters
from iotools import readgtf
from utils import gtutils
from utils import mfunc
from utils.containers import ZstateInfo
from utils.printstamp import printStamp

import pdb

from sklearn.preprocessing import scale

def parse_args():

    parser = argparse.ArgumentParser(description='Split expression file in it\'s different chromosomes')


    parser.add_argument('--expr',
                        type=str,
                        dest='rpkmpath',
                        metavar='FILE',
                        help='RNA-Seq RPKM counts for genes')


    parser.add_argument('--gtf',
                        type=str,
                        dest='gtfpath',
                        metavar='FILE',
                        help='Gene Annotation file')

    parser.add_argument('--chr',
                        #nargs='*',
                        type=int,
                        dest='chrom',
                        metavar='CHR',
                        help='choose genes from this chromosome only')

    parser.add_argument('--outdir',
                        #nargs='*',
                        type=str,
                        dest='outdir',
                        metavar='OUTDIR',
                        help='outdir')

    opts = parser.parse_args()
    return opts

opts = parse_args()

# Annotation (use complete gene name in gtf without trimming the version)
gene_info = readgtf.gencode_v12(opts.gtfpath, include_chrom = opts.chrom, trim=False)
mydict = {}
for i in gene_info:
    mydict[i.ensembl_id] = i.chrom

basename = os.path.basename(opts.rpkmpath)
outpath = os.path.join(opts.outdir, "chr_"+str(opts.chrom)+"_"+basename)

genecount = 0
with open(outpath, 'w') as outstream:
    with open(opts.rpkmpath, 'r') as instream:
        header = instream.readline()
        for line in instream:
            arr = line.strip().split()
            if arr[0] in mydict:
                genecount += 1
                outstream.write(line)
print("Exported "+str(genecount)+" genes for chr "+str(opts.chrom))




