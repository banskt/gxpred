#!/usr/bin/env python

''' This defines all containers used in this code
'''

import collections

SNPINFO_FIELDS = ['chrom', 'varid', 'bp_pos', 'ref_allele', 'alt_allele', 'maf']
class SnpInfo(collections.namedtuple('_SnpInfo', SNPINFO_FIELDS)):
    __slots__ = ()


GENEINFO_FIELDS = ['name', 'ensembl_id', 'chrom', 'start', 'end']
class GeneInfo(collections.namedtuple('_GeneInfo', GENEINFO_FIELDS)):
    __slots__ = ()


class GeneExpressionArray(collections.namedtuple('_GeneExpressionArray', ['geneid', 'expr_arr'])):
    __slots__ = ()


ZSTATEINFO_FIELDS = ['state', 'prob', 'exp']
class ZstateInfo(collections.namedtuple('_ZstateInfo', ZSTATEINFO_FIELDS)):
    __slots__ = ()
