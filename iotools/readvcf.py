#/usr/bin/env python

''' Parse dosage data from VCF file.
    Usage:
       from iotools.readvcf import ReadVCF
       readvcf = ReadVCF(vcf_filepath)
       snpinfo = readvcf.snpinfo
          ...
    Returns:
       a) information of all the variants read
            - chrom
            - pos
            - id
            - reference allele
            - alternate allele
            - minor allele frequency
       b) genotype matrix
       c) identity of the donors
'''

import numpy as np
import gzip
from utils.containers import SnpInfo

class ReadVCF:


    _read_genotype_once = False


    def __init__(self, filepath):
        self._filepath = filepath


    @property
    def snpinfo(self):
        self._run_once()
        return tuple(self._snpinfo)


    @property
    def dosage(self):
        self._run_once()
        return self._dosage


    @property
    def donor_ids(self):
        self._run_once()
        return tuple(self._donor_ids)


    def _run_once(self):
        if self._read_genotype_once:
            return
        self._read_genotype_once = True
        self._read_dosage()


    def _read_dosage(self):
        dosage = list()
        snpinfo = list()
        with gzip.open(self._filepath, 'r') as vcf:
            for line in vcf:
                linestrip = line.decode().strip()
                if linestrip[:2] == '##': continue
                if linestrip[:6] == '#CHROM':
                    linesplit = linestrip.split("\t")
                    donor_ids = linesplit[9:]
                else:
                    linesplit = linestrip.split("\t")
                    chrom = linesplit[0]
                    pos   = linesplit[1]
                    varid = linesplit[2]
                    ref   = linesplit[3]
                    alt   = linesplit[4]

                    dsindx = linesplit[8].split(':').index("DS")
                    ds = [x.split(':')[dsindx] for x in linesplit[9:]]
                    ds_notna = [float(x) for x in ds if x != "."]
                    freq = sum(ds_notna) / 2 / len(ds_notna)
                    maf = freq
                    snpdosage = [float(x) if x != '.' else 2 * maf for x in ds]
                    if freq > 0.5:
                        maf = 1 - freq
                        ref = linesplit[4]
                        alt = linesplit[3]
                        snpdosage = [2 - x for x in snpdosage]

                    this_snp = SnpInfo(chrom      = chrom,
                                       bp_pos     = pos,
                                       varid      = varid,
                                       ref_allele = ref,
                                       alt_allele = alt,
                                       maf        = maf)

                    dosage.append(snpdosage)
                    snpinfo.append(this_snp)

        self._dosage = np.array(dosage)
        self._snpinfo = snpinfo
        self._donor_ids = donor_ids
