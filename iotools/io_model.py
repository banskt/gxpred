#/usr/bin/env python

''' I/O classes for the model.
    Usage:
       from iotools.io_model import IOModel
       model = IOModel(modelpath, chrom)
       
       ## for writing:
       for gene in genes:
           model.write_gene(gene, snps, zstates)

       ## for reading:
       # first get the list of all genes
       genes = model.genes
       for gene in genes:
           # for each gene, get the snps and zstates
           model.read_gene(gene)
           snps = model.snps
           zstates = model.zstates
       
'''

import numpy as np
import ast
import os
from utils.containers import SnpInfo
from utils.containers import GeneInfo
from utils.containers import ZstateInfo

class IOModel:

    def __init__(self, modelpath, chrom):
        self._dirpath = os.path.join(modelpath, "chr{:d}".format(chrom))
        self._genefilename = os.path.join(self._dirpath, "genes.txt")
        self._SNPFILEFORMAT = "{:s}_snps.txt"
        self._ZSTATEFILEFORMAT = "{:s}_zstates.txt"
        if not os.path.exists(self._dirpath):
            os.makedirs(self._dirpath)
        

    @property
    def genes(self):
        if os.path.isfile(self._genefilename):
            self._read_genes()
        return self._genes


    def _read_genes(self):
        genes = list()
        with open(self._genefilename, 'r') as mfile:
            for mline in mfile:
                linesplit = mline.strip().split('\t')
                gene_name = linesplit[0]
                gene_id = linesplit[1]
                this_gene = GeneInfo(name       = gene_name,
                                     ensembl_id = gene_id,
                                     chrom      = 0,
                                     start      = 0,
                                     end        = 0)
                genes.append(this_gene)
        self._genes = genes


    def _write_gene(self):
        if os.path.exists(self._genefilename):
            fmode = 'a'
        else:
            fmode = 'w'
        with open(self._genefilename, fmode) as mfile:
            mfile.write("{0}\t{1}\n".format(self._setgene.name, self._setgene.ensembl_id))


    def read_gene(self, gene):
        self._setgene = gene
        self._read_snps()
        self._read_zstates()


    def write_gene(self, gene, snps, zstates):
        self._setgene = gene
        self._snps = snps
        self._zstates = zstates
        self._write_gene()
        self._write_snps()
        self._write_zstates()


    @property
    def snps(self):
        return self._snps


    @property
    def zstates(self):
        return self._zstates


    def snpfilename(self):
        filename = os.path.join(self._dirpath, self._SNPFILEFORMAT.format(self._setgene.ensembl_id))
        return filename


    def zstatesfilename(self):
        filename = os.path.join(self._dirpath, self._ZSTATEFILEFORMAT.format(self._setgene.ensembl_id))
        return filename


    def _read_snps(self):
        filename = self.snpfilename()
        snps = list()
        with open(filename, 'r') as mfile:
            for mline in mfile:
                linesplit = mline.strip().split('\t')
                chrom = int(linesplit[0])
                pos   = int(linesplit[1])
                varid = linesplit[2]
                ref   = linesplit[3]
                alt   = linesplit[4]
                maf   = float(linesplit[5])
                this_snp = SnpInfo(chrom      = chrom,
                                   bp_pos     = pos,
                                   varid      = varid,
                                   ref_allele = ref,
                                   alt_allele = alt,
                                   maf        = maf)
                snps.append(this_snp)
        self._snps = snps


    def _write_snps(self):
        filename = self.snpfilename()
        with open(filename, 'w') as mfile:
            for snp in self._snps:
                mfile.write("{:d}\t{:d}\t{:s}\t{:s}\t{:s}\t{:g}\n".format(snp.chrom, snp.bp_pos, snp.varid, snp.ref_allele, snp.alt_allele, snp.maf))


    def _read_zstates(self):
        filename = self.zstatesfilename()
        zstates = list()
        with open(filename, 'r') as mfile:
            for mline in mfile:
                linesplit = mline.strip().split(';')
                state = ast.literal_eval(linesplit[0].strip())
                prob = float(linesplit[1].strip())
                exp = np.array(ast.literal_eval(linesplit[2].strip()))
                this_zstate = ZstateInfo(state = state,
                                         prob  = prob,
                                         exp   = exp)
                zstates.append(this_zstate)
        self._zstates = zstates


    def _write_zstates(self):
        filename = self.zstatesfilename()
        with open(filename, 'w') as mfile:
            for z in self._zstates:
                str_state = '[' + ', '.join(['{:d}'.format(x) for x in z.state]) + ']'
                str_exp   = '[' + ', '.join(['{:g}'.format(x) for x in z.exp])   + ']'
                mfile.write("{:s}; {:g}; {:s}\n".format(str_state, z.prob, str_exp))
