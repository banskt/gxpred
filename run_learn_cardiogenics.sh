#!/bin/bash

GXPRED='/home/franco/gxpred/learn_cardiogenics.py'
GTFOLDER="/media/disk1/Cardiogenics_eQTL/complete_set_imputed"
GTFILE="CG_1.imputed"
SAMPLEFILE="CG.sample"
CHROM="1"
EXPR="/media/disk1/Cardiogenics_eQTL/expression/lumi_mono_all_vsn_batches_adjusted.RData.exp.txt"
GTFPATH="/media/disk1/geuvadis/gencode.v12.annotation.gtf.gz"
OUTDIR="/media/disk1/Cardiogenics_eQTL/complete_set_imputed/gxpred_models"


python $GXPRED --gen ${GTFOLDER}/${GTFILE} --sample ${GTFOLDER}/${SAMPLEFILE} --gtf ${GTFPATH} --chr ${CHROM} --expr ${EXPR} --out ${OUTDIR}


# 2017-11-28 13:59:14 - started reading genotype
