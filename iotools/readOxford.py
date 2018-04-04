#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import gzip
from collections import defaultdict
from utils.printstamp import printStamp


# Parse data from dosage files
import collections
import numpy as np
import re
import pdb

from utils.containers import SnpInfo

SNP_COMPLEMENT = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}

class ReadOxford:

    _read_samples_once = False
    _read_genotype_once = False
    _nloci = 0
    _nsample = 0


    def __init__(self, gtfile, samplefile, chrom, dataset, nlocilimit=-1):
        self._chrom = chrom
        self._gtfile = gtfile
        self._samplefile = samplefile
        self._dataset = dataset
        self._nlocilimit = nlocilimit
        self._read_genotypes()

        
    @property
    def nsample(self):
        self._read_samples()
        return self._nsample

    @property
    def samplenames(self):
        self._read_samples()
        return self._samplenames
    
    @property
    def nloci(self):
        return self._nloci

    @property
    def snps_info(self):
        self._read_genotypes()
        return tuple(self._snps_info)

    @property
    def nsnps(self):
        self._run_once()
        return tuple(self._nsnps)

    @property
    def dosage(self):
        self._read_genotypes()
        return tuple(self._dosage)

    # @property
    # def genotype(self):
    #     self._read_genotypes()
    #     return tuple(self._genotype)


    def _read_samples(self):
        if self._read_samples_once:
           return
        self._read_samples_once = True

        with open(self._samplefile, 'r') as samfile:
            # header = samfile.readline().strip().split()
            # header_types = samfile.readline().strip().split()
            sample = 0
            samplenames = list()
            for line in samfile:
                if re.search('^#', line):
                    continue
                sample += 1
                samplenames.append(line.strip().split()[0])
            
        self._nsample = sample
        self._samplenames = samplenames

    def _read_cardiogenics(self):
        dosage = list()
        allsnps = list()
        self._nloci = 0
        with gzip.open(self._gtfile, 'r') as filereader:
            for snpline in filereader:
                self._nloci += 1
                # print('reading '+str(self._nloci)+' snps, size of dosages is '+str(sys.getsizeof(dosage)), end='\r')
                mline = snpline.split()
                
                ngenotypes = (len(mline) - 5) / 3
                if float(ngenotypes).is_integer():
                    if ngenotypes != self._nsample:
                        raise ValueError('Number of samples differ from genotypes')
                else:
                    raise ValueError('Number of columns in genotype not divisible by 3')

                gt_freqs = np.array(list(map(float,mline[5:])))
                # genotype.append(gt_freqs)

                indsAA = np.arange(0,self._nsample)*3
                indsAB = indsAA + 1
                indsBB = indsAB + 1

                snp_dosage = 2*gt_freqs[indsBB] + gt_freqs[indsAB] # [AA, AB, BB] := [0, 1, 2]

                freq = sum(snp_dosage) / 2 / len(snp_dosage)
                maf = freq

                this_snp = SnpInfo(    chrom      = int(mline[0]), # chrom      = int(self._chrom),
                                       bp_pos     = int(mline[2]),
                                       varid      = mline[1].decode("utf-8"),
                                       ref_allele = mline[3].decode("utf-8"),
                                       alt_allele = mline[4].decode("utf-8"),
                                       maf        = maf)

                allsnps.append(this_snp)
                dosage.append(snp_dosage)
                if self._nlocilimit > 0:
                    if self._nloci > self._nlocilimit:
                        break
        return allsnps, dosage

    def _read_gtex(self):
        dosage = list()
        allsnps = list()
        self._nloci = 0
        with gzip.open(self._gtfile, 'r') as filereader:
            for snpline in filereader:
                self._nloci += 1
                # print('reading '+str(self._nloci)+' snps, size of dosages is '+str(sys.getsizeof(dosage)), end='\r')
                mline = snpline.split()
                
                ngenotypes = len(mline) - 6
                if float(ngenotypes).is_integer():
                    if ngenotypes != self._nsample:
                        raise ValueError('Number of samples differ from genotypes')
                else:
                    raise ValueError('Number of columns in genotype not divisible by 3')
                    
                snp_dosage = np.array(list(map(float,mline[6:])))

                freq = float(mline[5])
                maf = freq

                this_snp = SnpInfo(    chrom      = int(mline[0]), # chrom      = int(self._chrom),
                                       bp_pos     = int(mline[2]),
                                       varid      = mline[1].decode("utf-8"),
                                       ref_allele = mline[3].decode("utf-8"),
                                       alt_allele = mline[4].decode("utf-8"),
                                       maf        = maf)
                allsnps.append(this_snp)
                dosage.append(snp_dosage)
                if self._nlocilimit > 0:
                    if self._nloci > self._nlocilimit:
                        break
        return allsnps, dosage

    def _read_genotypes(self):
        if self._read_genotype_once:
            return
        self._read_genotype_once = True

        self._read_samples() # otherwise, self._nsample is not set

        dosage = list()
        allsnps = list()
        # genotype = list()

        printStamp("started reading genotype")

        if self._dataset == "cardiogenics":
            allsnps, dosage = self._read_cardiogenics()
        if self._dataset == "gtex":
            allsnps, dosage = self._read_gtex()
    
        print("Read "+str(self._nloci)+" snps in "+str(self._nsample)+" samples.",end='\n')
        printStamp("Finished readings snps")
        self._dosage = np.array(dosage)
        # self._genotype = genotype
        self._snps_info = allsnps

    def _filter_snps(self):
        # Predixcan style filtering of snps
        newsnps = list()
        for snp in self._snps_info:
            pos = snp.bp_pos
            refAllele = snp.ref_allele
            effectAllele = snp.alt_allele
            rsid = snp.varid
            # Skip non-single letter polymorphisms
            if len(refAllele) > 1 or len(effectAllele) > 1:
                continue
            # Skip ambiguous strands
            if SNP_COMPLEMENT[refAllele] == effectAllele:
                continue
            if rsid == '.':
                continue
            newsnps.append(snp)
        return newsnps

    def write_dosages(self, outfile, format="predixcan"):
        if not self._read_genotype_once:
            raise ValueError("No dosages to write. Run read_genotypes first.",end='\n')
        print("Writing dosages to file "+outfile)

        if format == "predixcan":
            with open(outfile+".annot", 'w') as outstream2:
                with open(outfile, 'w') as outstream:
                    headers = "Id "+" ".join(self._samplenames)+"\n"
                    annotheader = "\t".join(["chr","pos","varID","refAllele","effectAllele","rsid"])
                    outstream.write(headers)
                    outstream2.write(annotheader+"\n")
                    newsnps = self._filter_snps()
                    for i, snp in enumerate(newsnps):
                        variant_id = "_".join([self._chrom, str(snp.bp_pos), snp.ref_allele, snp.alt_allele, "b37"])
                        annot_header = " ".join(["chr", "position", "VariantID", "RefAllele", "AlternativeAllele", "rsid", "rsid"])
                        dosage_row = " ".join([variant_id] + list(map(str, self._dosage[i])))
                        outstream.write(dosage_row+"\n")

                        annotline = "\t".join([self._chrom, str(snp.bp_pos), variant_id, snp.ref_allele, snp.alt_allele, snp.varid])
                        outstream2.write(annotline+"\n")
            print("Done writing dosages")
        else:
            print("Unsupported Format")
            return False

