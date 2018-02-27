#!/bin/bash

GXPRED='/home/franco/gxpred/learn_gtex.py'
GTFOLDER="/media/disk1/gtex"
GTFILE="GTEx_450Indiv_genot_imput_info04_maf01_HWEp1E6_dbSNP135IDs_donorIDs_dosage_chr1.gz"
SAMPLEFILE="donor_ids.fam"
CHROM="1"
EXPR="/media/disk1/gtex/gtex_wholeblood_normalized.expression.txt"
GTFPATH="/media/disk1/geuvadis/gencode.v12.annotation.gtf.gz"
OUTDIR="/media/disk1/gtex/gxpred_models"


python $GXPRED --gen ${GTFOLDER}/${GTFILE} --sample ${GTFOLDER}/${SAMPLEFILE} --gtf ${GTFPATH} --chr ${CHROM} --expr ${EXPR} --out ${OUTDIR}
