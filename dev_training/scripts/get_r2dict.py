#!/usr/bin/env python

import os
import argparse
import collections
import pandas as pd
import numpy as np
from scipy import stats


def parse_args():

    parser = argparse.ArgumentParser(description='Prediction of gene expression')

    parser.add_argument('--pred',
                        type=str,
                        dest='predpath',
                        metavar='FILE',
                        help='predicted gene expression file')

    parser.add_argument('--obs',
                        type=str,
                        dest='obsvpath',
                        metavar='FILE',
                        help='observed gene expression file')

    parser.add_argument('--out',
                        type=str,
                        dest='outpath',
                        metavar='FILE',
                        help='output file of r^2')

    opts = parser.parse_args()
    return opts


def read_gx_df(filepath, geneids):
    df = pd.read_csv(filepath, sep='\t', index_col=1, header=None)
    df.drop(df.columns[0], axis=1, inplace=True)
    df.columns = geneids
    df = df.transpose()
    df.index.name = 'gene_id'
    df.columns = df.columns.str.strip()
    return df


opts = parse_args()

geneids_filepath = os.path.splitext(opts.predpath)[0] + ".geneids"
geneids = list()
with open(geneids_filepath, 'r') as mfile:
    for mline in mfile:
        geneids.append(mline.strip().split()[0])

pred_df = read_gx_df(opts.predpath, geneids)
obsv_df = read_gx_df(opts.obsvpath, geneids)
common_ids   = [x for x in obsv_df.columns.tolist() if x in pred_df.columns.tolist()]
pred_common_df = pred_df[common_ids]
obsv_common_df = obsv_df[common_ids]

r2dict = collections.defaultdict(lambda: 0)
with open(opts.outpath, 'w') as mfile:
    for gene in geneids:
        Y = pred_common_df.loc[gene].as_matrix()
        X = obsv_common_df.loc[gene].as_matrix()
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
        rsq = r_value * r_value
        mfile.write("{:s} {:g} {:g}\n".format(gene, rsq, p_value))
