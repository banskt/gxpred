#!/bin/bash

GENE="RDH16"
CODEBASE="../"
TRAINVCF="${GENE}_genotype.vcf.gz"
TRAINRPKM="${GENE}_expression.txt"
TRAINGTF="${GENE}_annotation.gtf.gz"
CHROM="12" # change it to whichever chromosome you are interested 
PREDVCF=""
MODELDIR="model"

python ${CODEBASE}/learn.py   --vcf ${TRAINVCF} --expr ${TRAINRPKM} --gtf ${TRAINGTF} --chr ${CHROM} --params 0.01 0.0 0.01 0.001 0.001 --out ${MODELDIR}
#python ${CODEBASE}/predict.py --vcf ${PREDVCF}  --model ${MODELDIR} --chrom ${CHROM} --outprefix predicted_gene_expression
