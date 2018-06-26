import numpy as np
import math
from iotools.readOxford import ReadOxford
from iotools.readrpkm import ReadRPKM
from iotools import readgtf
from utils.containers import GeneInfo
from utils import mfunc
from utils import gtutils
from utils.logs import MyLogger
from inference.linreg_association import LinRegAssociation


class Error(Exception):
    pass

class SPLIT_gt_GENES(Error):
    pass

class Data():

    def __init__(self, args):
        self.args = args
        self._gt = None
        self._snpinfo = None
        self._geneinfo = None
        self._expr = None
        self._1kg_run_once = False
        self._gene_batch = list()
        self.logger = MyLogger(__name__)


    @property
    def dosage(self):
        return self._gt

    @property
    def gt_norm(self):
        return self._gt_norm

    @property
    def snpinfo(self):
        return self._snpinfo

    @property
    def geneinfo(self):
        return self._geneinfo

    @property
    def expression(self):
        return self._expr

    @property
    def genes(self):
        return self._genes

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def samplenames(self):
        return self._samplenames

    @property
    def gene_batch(self):
        return self._gene_batch

    @property
    def expr_batch(self):
        return self._expr_batch

    @property
    def cis_snps(self):
        return self._cis_snps

    def get_1kg_annots(self):
        self.load_1kg_annotations()
        #then load the annotations for the cis_snps
        nsnps_used = len(self._cis_snps)            
        current_annot = list()
        for snp in self._cis_snps:
            if len(annot_dict[snp.varid]) > 0:
                current_annot.append(annot_dict[snp.varid])
            else:
                current_annot.append([0,0,0,0,0])
                self.logger.debug("SNP {:s}: no annotations found!".format(snp.varid))
        feature1kg = np.array(current_annot)

    def load_1kg_annotations(self):
        if self._1kg_run_once:
            return None
        else:
            # Load rsid dictionary
            self._annot_dict = defaultdict(list)
            annotfile = os.path.join(self.args.annots_dir, "1KG."+str(self.args.chrom)+".annot.gz")
            with gzip.open(annotfile, 'r') as instream:
                _ = instream.readline()
                for line in instream:
                    arr = line.decode().strip().split(" ")
                    rsid = arr[0]
                    annots = list(map(int, arr[1:]))
                    self._annot_dict[rsid] = annots
            self._1kg_run_once = True

    def load_annotations(self, usefeat):
        nsnps_used = len(self._cis_snps)
        features = np.ones((nsnps_used, 1))
        if usefeat == "1kg":
            feat_array_1kg = get_1kg_annotations()
            features = np.concatenate((features, feature1kg), axis=1)
        # add other features
        return features

    def select_batch_genes(self):
        # if gene number is outside of the range, do not calculate and continue
        expr_indx = np.empty(0, dtype=int)
        for i in range(0, self._gene_number):
            if (i >= self._batch_size*self.args.section and i < (self._batch_size*self.args.section + self._batch_size)):
                self._gene_batch.append(self._genes[i])
                expr_indx = np.append(expr_indx, i)
        self._expr_batch = self._expr[expr_indx,:]


    def select_cis_snps(self, gene, target):
        self._cis_snps = None
        cismask = mfunc.select_snps(gene, self._snpinfo, self.args.window)
        print(cismask)
        if len(cismask) > 0:
            snpmask = cismask
            predictor = self._gt_norm[cismask,:]
            if len(cismask) > self.args.min_snps:
                assoc_model = LinRegAssociation(predictor, target, self.args.min_snps, self.args.pval_cutoff, self.args.cutoff)
                pvalmask = cismask[assoc_model.selected_variables]
                if pvalmask.shape[0] == 0:
                    self.logger.info("No significant SNPs found for gene {:s}".format(gene.ensembl_id))
                    self.logger.info("Using all initial cis-SNPs: {:d}".format(len(cismask)))
                else:
                    self.logger.info("Found {:d} SNPs, reduced to {:d} SNPs (max p-value {:g}) for {:s}".format(len(cismask), len(pvalmask), assoc_model.ordered_pvals[len(pvalmask) - 1], gene.name))
                    snpmask = pvalmask
                    predictor = self._gt_norm[pvalmask,:]
                self._cis_snps = [self._snpinfo[i] for i in snpmask]
            else:
                self.logger.info("Found {:d} SNPs for {:s}".format(len(cismask), gene.name))
            return predictor
        else:
            self.logger.info("No cis-SNPs found")
            return np.empty(0)
        

    def load(self):
        # Annotation (use complete gene name in gtf without trimming the version)
        # load annotation for whole genome
        gene_info = readgtf.gencode_v12(self.args.gtf_file, include_chrom = self.args.chrom, trim=False)

        # read Genotype

        if self.args.oxf_file:
            oxf = ReadOxford(self.args.oxf_file, self.args.fam_file, self.args.chrom, self.args.dataset)
            genotype = oxf.dosage
            samplenames = oxf.samplenames
            snps = oxf.snps_info

        if self.args.vcf_file:
            pass

        # Filter genotype and Quality control
        self._snpinfo, self._gt = gtutils.remove_low_maf(snps, genotype, 0.1)
        self._gt_norm = gtutils.normalize(self._snpinfo, self._gt)

        # Gene Expression
        rpkm = ReadRPKM(self.args.gx_file, "gtex")
        expression = rpkm.expression
        expr_donors = rpkm.donor_ids
        gene_names = rpkm.gene_names

        # Selection and match gx to gt
        self.logger.info("Selecting and matching samples from GT and GX")
        vcfmask, exprmask = mfunc.select_donors(samplenames, expr_donors)
        genes, indices = mfunc.select_genes(gene_info, gene_names)

        self._samplenames = [expr_donors[i] for i in exprmask]
        self._genes = genes
        self._gene_number = len(genes)
        self._expr = expression[:, exprmask][indices, :]
        self._gt = self._gt[:, vcfmask]
        self._gt_norm = self._gt_norm[:, vcfmask]

        self._batch_size = None

        if self.args.split and self.args.split > 1:
            if self._gene_number > self.args.split:
                self.logger.info("Gene number: {:d}".format(self._gene_number))
                self._batch_size = math.ceil(self._gene_number / self.args.split)
                self.logger.info("Splitting in {:d} batches of {:d} genes each.".format(self.args.split, self._batch_size))
                self.select_batch_genes()
            else:
                raise SPLIT_gt_GENES("Split number is greater than number of genes. Cannot split the job")



