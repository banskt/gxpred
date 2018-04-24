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

class WriteModel:

    def __init__(self, modelpath, chrom):
        self._dirpath = os.path.join(modelpath, "chr{:d}".format(chrom))
        self._genefilename = os.path.join(self._dirpath, "genes.txt")
        self._SNPFILEFORMAT = "{:s}_snps.txt"
        self._ZSTATEFILEFORMAT = "{:s}_zstates.txt"
        self._create_file_once = False

    def _create_file(self, params):
        if not self._create_file_once:
            if not os.path.exists(self._dirpath):
                os.makedirs(self._dirpath)
            if not os.path.exists(self._genefilename):
                nfeat = params.shape[0] - 4
                gammas = ["{:8s}".format("Gamma"+str(i)) for i in range(0, nfeat)]
                with open(self._genefilename, 'w') as mfile:
                    f = "{:25s}\t{:25s}\t{:7s}\t"+"\t".join(["{:8s}"] * nfeat)+"\t{:8s}\t{:8s}\t{:8s}\t{:8s}\n"
                    # print(f.format("Ensembl_ID", "Gene_Name", "Success", "Mu", "Sigma", "Sigma_bg", "Sigma_tau", *gammas))
                    mfile.write(f.format("Ensembl_ID", "Gene_Name", "Success", "Mu", "Sigma", "Sigma_bg", "Sigma_tau", *gammas))
            self._create_file_once = True

    def snpfilename(self):
        filename = os.path.join(self._dirpath, self._SNPFILEFORMAT.format(self._setgene.ensembl_id))
        return filename


    def zstatesfilename(self):
        filename = os.path.join(self._dirpath, self._ZSTATEFILEFORMAT.format(self._setgene.ensembl_id))
        return filename

    def _write_gene(self, success, params):
        nfeat = params.shape[0] - 4
        gammas = ["{:8.5f}".format(params[i]) for i in range(0, nfeat)]
        with open(self._genefilename, 'a') as mfile:
            f = "{:25s}\t{:25s}\t{!r}\t"+"\t".join(["{:8.5f}"]* nfeat)+"\t{:8.5f}\t{:8.5f}\t{:8.5f}\t{:8.5f}\n"
            mfile.write(f.format(self._setgene.ensembl_id, 
                                 self._setgene.name,
                                 success,
                                 params[nfeat + 0], 
                                 params[nfeat + 1], 
                                 params[nfeat + 2], 
                                 params[nfeat + 3],
                                 *params[:nfeat] ))

    def write_success_gene(self, gene, snps, zstates, params):
        self._setgene = gene
        self._snps = snps
        self._zstates = zstates
        self._create_file(params)
        self._write_gene(True, params)
        self._write_snps()
        self._write_zstates()


    def write_failed_gene(self, gene, params):
        self._setgene = gene
        self._write_gene(False, params)


    def _write_snps(self):
        filename = self.snpfilename()
        with open(filename, 'w') as mfile:
            for snp in self._snps:
                mfile.write("{:d}\t{:d}\t{:s}\t{:s}\t{:s}\t{:g}\n".format(snp.chrom, snp.bp_pos, snp.varid, snp.ref_allele, snp.alt_allele, snp.maf))


    def _write_zstates(self):
        filename = self.zstatesfilename()
        with open(filename, 'w') as mfile:
            for z in self._zstates:
                str_state = '[' + ', '.join(['{:d}'.format(x) for x in z.state]) + ']'
                str_exp   = '[' + ', '.join(['{:g}'.format(x) for x in z.exp])   + ']'
                mfile.write("{:s}; {:g}; {:s}\n".format(str_state, z.prob, str_exp))


class ReadModel:

    def __init__(self, modelpath, chrom):
        self._dirpath = os.path.join(modelpath, "chr{:d}".format(chrom))
        self._genefilename = os.path.join(self._dirpath, "genes.txt")
        self._SNPFILEFORMAT = "{:s}_snps.txt"
        self._ZSTATEFILEFORMAT = "{:s}_zstates.txt"


    def snpfilename(self):
        filename = os.path.join(self._dirpath, self._SNPFILEFORMAT.format(self._setgene.ensembl_id))
        return filename


    def zstatesfilename(self):
        filename = os.path.join(self._dirpath, self._ZSTATEFILEFORMAT.format(self._setgene.ensembl_id))
        return filename


    @property
    def genes(self):
        if os.path.isfile(self._genefilename):
            self._read_genes()
        else:
            raise Exception("File "+self._genefilename+" does not exist")
        return self._genes


    @property
    def snps(self):
        return self._snps


    @property
    def zstates(self):
        return self._zstates


    def _read_genes(self):
        genes = list()
        with open(self._genefilename, 'r') as mfile:
            for mline in mfile:
                line = mline.strip()
                if line == "":
                    continue
                linesplit = line.split('\t')
                gene_name = linesplit[1].strip()
                gene_id = linesplit[0].strip()
                if gene_id == "Ensembl_ID":
                    continue
                this_gene = GeneInfo(name       = gene_name,
                                     ensembl_id = gene_id,
                                     chrom      = 0,
                                     start      = 0,
                                     end        = 0)
                if linesplit[2] == "True":
                    genes.append(this_gene)
        self._genes = genes


    def read_gene(self, gene):
        self._setgene = gene
        self._read_snps()
        self._read_zstates()


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
