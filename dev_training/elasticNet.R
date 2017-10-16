library(glmnet)

expression_RDS = 'predictdb_data/intermediate/expression_phenotypes/geuvadis.expr.RDS'
geno_file = 'predictdb_data/intermediate/genotypes/geuvadis.snps.chr21.txt'
gene_annot_RDS = 'predictdb_data/intermediate/annotations/gene_annotation/gencode.v12.genes.parsed.RDS'
snp_annot_RDS = 'predictdb_data/intermediate/annotations/snp_annotation/geuvadis.annot.chr21.RDS'
n_k_folds = 10
alpha = 0.5
out_dir = 'predictdb_data/model'
tis = 'gEUVADIS'
chrom = 21
snpset = 'HapMap'
window = 1e6

expression <- readRDS(expression_RDS)
class(expression) <- 'numeric'
genotype <- read.table(geno_file, header = TRUE, row.names = 'Id', stringsAsFactors = FALSE)
# Transpose genotype for glmnet
genotype <- t(genotype)
gene_annot <- readRDS(gene_annot_RDS)
gene_annot <- subset(gene_annot, gene_annot$chr == chrom)
snp_annot <- readRDS(snp_annot_RDS)

rownames(gene_annot) <- gene_annot$gene_id
# Subset expression data to only include genes with gene_info
expression <- expression[, intersect(colnames(expression), rownames(gene_annot))]
exp_samples <- rownames(expression)
exp_genes <- colnames(expression)
n_samples <- length(exp_samples)
n_genes <- length(exp_genes)
seed <- sample(1:2016, 1)
set.seed(seed)
groupid <- sample(1:n_k_folds, length(exp_samples), replace = TRUE)

i = which (exp_genes == 'ENSG00000160284.10')
gene <- exp_genes[i]
# Reduce genotype data to only include SNPs within specified window of gene.
geneinfo <- gene_annot[gene,]
start <- geneinfo$start - window
end <- geneinfo$end + window
# Pull cis-SNP info
cissnps <- subset(snp_annot, snp_annot$pos >= start & snp_annot$pos <= end)
# Pull cis-SNP genotypes
cisgenos <- genotype[,intersect(colnames(genotype), cissnps$varID), drop = FALSE]
# Reduce cisgenos to only include SNPs with at least 1 minor allele in dataset
cm <- colMeans(cisgenos, na.rm = TRUE)
minorsnps <- subset(colMeans(cisgenos), cm > 0.05 & cm < 1.95)
minorsnps <- names(minorsnps)
cisgenos <- cisgenos[,minorsnps, drop = FALSE]

exppheno <- expression[,gene]
# Scale for fastLmPure to work properly
exppheno <- scale(exppheno, center = TRUE, scale = TRUE)

exppheno[is.na(exppheno)] <- 0
rownames(exppheno) <- rownames(expression)
# Run Cross-Validation
# parallel = TRUE is slower on tarbell
bestbetas <- tryCatch(
  { fit <- cv.glmnet(as.matrix(cisgenos),
                     as.vector(exppheno),
                     nfolds = n_k_folds,
                     alpha = alpha,
                     keep = TRUE,
                     foldid = groupid,
                     parallel = FALSE)
  # Pull info from fit to find the best lambda   
  fit.df <- data.frame(fit$cvm, fit$lambda, 1:length(fit$cvm))
  # Needs to be min or max depending on cv measure (MSE min, AUC max, ...)
  best.lam <- fit.df[which.min(fit.df[,1]),]
  cvm.best <- best.lam[,1]
  lambda.best <- best.lam[,2]
  # Position of best lambda in cv.glmnet output
  nrow.best <- best.lam[,3]
  # Get the betas from the best lambda value
  ret <- as.data.frame(fit$glmnet.fit$beta[,nrow.best])
  ret[ret == 0.0] <- NA
  # Pull the non-zero betas from model
  as.vector(ret[which(!is.na(ret)),])
  },
  error = function(cond) {
    # Should fire only when all predictors have 0 variance.
    message('Error with gene ' %&% gene %&% ', index ' %&% i)
    message(geterrmessage())
    return(data.frame())
  }
)

names(bestbetas) <- rownames(ret)[which(!is.na(ret))]
# Pull out the predictions at the best lambda value.    
pred.mat <- fit$fit.preval[,nrow.best]
res <- summary(lm(exppheno~pred.mat))
genename <- as.character(gene_annot[gene, 3])
rsq <- res$r.squared
pval <- res$coef[2,4]
print(c(gene, cvm.best, nrow.best, lambda.best, length(bestbetas), rsq, pval, genename))
# Output best shrunken betas for PrediXcan
#bestbetalist <- names(bestbetas)
#bestbetainfo <- snp_annot[bestbetalist,]
#betatable <- as.matrix(cbind(bestbetainfo,bestbetas))
write.table(rownames(cisgenos), 'sample_names.csv', sep=",", col.names = FALSE, row.names = FALSE)
