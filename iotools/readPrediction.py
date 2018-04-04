import re, os
import numpy as np

class ReadPrediction:
      
    def __init__(self, predpath, samplepath, predictor, trim=False, prefix=None):
        self._pred_path = predpath
        self._predictor = predictor #gxpred or predixcan
        self._expr_mat = None
        self._samplepath = samplepath
        self._prefix = prefix
        self._load_samples()
        self._load_chromosomes()
        self._prev_available = False
        if trim:
            self._trim_gene_names()
    
    
    def _trim_gene_names(self):
        self._gene_names = [g.split(".")[0] for g in self._gene_names]
    
    def _load_samples(self):
        samples = list()
        with open(self._samplepath, 'r') as instream:
            for line in instream:
                if re.search('^#', line):
                    continue
                arr = line.strip().split()
                samples.append(arr[0])
        self._samples = samples            
    
    # return expression matrix in samples-by-genes format
    def _load_chromosome(self, chrom):
        
        if self._predictor == "gxpred":
            if self._prefix:
                basefile = self._prefix+chrom
            else:
                basefile = "pred_chr"+chrom
            pred_geneids = os.path.join(self._pred_path,basefile+".geneids")
            pred_exp = os.path.join(self._pred_path,basefile+".txt")
            if os.path.exists(pred_geneids) and os.path.exists(pred_exp):
                with open(pred_geneids, 'r') as infile:
                    gene_names = [l.strip() for l in infile.readlines()]
                    ncolumns = len(gene_names)
                        
                # need the number of columns (num of genes), to skip first two sample id columns
                expr_mat = np.loadtxt(pred_exp, usecols=range(2,ncolumns+2))            
            else:
                print("No prediction found for CHR {:s}".format(chrom))
                return False, [], []
            
        if self._predictor == "predixcan":
            if self._prefix:
                filename = self._prefix+chrom
            else:
                filename = "predixcan_chr"+chrom+".output"
            filename = "predixcan_chr"+chrom+".output"
            filepath = os.path.join(self._pred_path, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as instream:
                    gene_names = instream.readline().strip().split()
                expr_mat = np.loadtxt(filepath, skiprows=1)
            else:
                print("No prediction found for CHR {:s}".format(chrom))
                return False, [], []
            
        print("Loaded {:d} genes in CHR {:s}".format(len(gene_names), chrom))
        return True, expr_mat, gene_names
                
    def _load_chromosomes(self):
        self._gene_names = []
        self._expr_mat = None
               
        for chrom in range(1,23):
            success, expr_mat, gene_names = self._load_chromosome(str(chrom))
            if success:
                if self._predictor == "gxpred":
                    if type(self._expr_mat) is not np.ndarray:
                        print(len(expr_mat.shape))
                        if len(expr_mat.shape) < 2:
                            self._expr_mat = expr_mat[np.newaxis].T
                        else:
                            self._expr_mat = expr_mat
                    else:
                        self._expr_mat = np.concatenate((self._expr_mat, expr_mat), axis=1)
                    self._gene_names += gene_names

                if self._predictor == "predixcan":
                    if type(self._expr_mat) is not np.ndarray:
                        self._expr_mat = expr_mat
                    else:
                        self._expr_mat += expr_mat
                    self._gene_names = gene_names

    def sort_by_samples(self, samplelist, use_prev = False):
        if use_prev:
            if self._prev_available:
                common_samples, ix = self._get_common_elements(self._sorted_samples, samplelist)
                # [x for x in samplelist if x in self._sorted_samples]
                print("Samples found: {:d} of {:d}".format(len(common_samples), len(samplelist)))
                print(self._sorted_expr_mat.shape)
                self._sorted_expr_mat = self._sorted_expr_mat[ix,:]    # index samples rows
                self._sorted_samples = [self._sorted_samples[i] for i in ix]        # index samples names
            else:
                print("No sorted results found. Run without use_prev at least once")
        else:
            common_samples, ix = self._get_common_elements(self._samples, samplelist)
            # [x for x in samplelist if x in self._samples]
            print("Samples found: {:d} of {:d}".format(len(common_samples), len(samplelist)))
            self._sorted_expr_mat = self._expr_mat[ix,:]   # index samples rows
            self._sorted_samples = [self._samples[i] for i in ix]       # index samples names
            self._sorted_gene_names = self._gene_names
            self._prev_available = True
             
    
    def _get_common_elements(self, mylist, targetlist):
        common_elements = [g for g in targetlist if g in mylist]
        if len(common_elements) == 0:
            raise Exception("No common elements found")
        ix = [mylist.index(i) for i in common_elements]
        return common_elements, ix
    
    def sort_by_gene(self, genelist, use_prev = False):
        # sorts expression matrix by the given genelist

        if use_prev:
            if self._prev_available:
                common_genes, ix = self._get_common_elements(self._sorted_gene_names, genelist)
                # [g for g in genelist if g in self._sorted_gene_names]
                print("Genes found: {:d} of {:d}".format(len(common_genes)), len(genelist))
                self._sorted_expr_mat = self._sorted_expr_mat[:,ix]     # index gene columns
                self._sorted_gene_names = [self._sorted_gene_names[i] for i in ix]   # index gene names
            else:
                print("No sorted results found. Run without use_prev at least once")
        else:
            common_genes, ix = self._get_common_elements(self._gene_names, genelist)
            # [g for g in genelist if g in self._gene_names]
            print("Genes found: {:d} of {:d}".format(len(common_genes), len(genelist)))
            self._sorted_expr_mat = self._expr_mat[:,ix]     # index gene columns
            self._sorted_gene_names = [self._gene_names[i] for i in ix]   # index gene names
            self._sorted_samples = self._samples
            self._prev_available = True
    
    
    @property
    def expr_mat(self):
        return self._expr_mat
    
    @property
    def gene_names(self):
        return self._gene_names
    
    @property
    def samples(self):
        return self._samples
    
    @property
    def sorted_samples(self):
        return self._sorted_samples
    
    @property
    def sorted_gene_names(self):
        return self._sorted_gene_names
    
    @property
    def sorted_expr_mat(self):
        return self._sorted_expr_mat