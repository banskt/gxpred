#!/bin/bash

module load intel/compiler/64/2017/17.0.2
module load intel/mkl/64/2017/2.174

HOME="/usr/users/fsimone"
OUTPATH="/scratch/fsimone/gtex_models"


ENV="${HOME}/myenv"
GXPRED="${HOME}/gxpred_annots/gxpred/learn_refactored.py"
OUTDIR="${OUTPATH}/gxpred_refactored_test"
OUTPREFIX="gxpred_refactored_test"
DATASET="gtex"


CHROM=12
GTFOLDER="${HOME}/datasets/gtex"
OXF_FILE="${GTFOLDER}/GTEx_450Indiv_genot_imput_info04_maf01_HWEp1E6_dbSNP135IDs_donorIDs_dosage_chr${CHROM}.gz"
SAMPLE_FILE="${GTFOLDER}/donor_ids.fam"
EXPR_FILE="${HOME}/datasets/gtex/gtex_wholeblood_normalized.expression.txt" #gtex_wholeblood_normalized.lm_corr_final2.exp.txt" #
GTF_FILE="${HOME}/datasets/gtex/gencode.v19.annotation.gtf.gz"
Z="1"
BATCH_SECTION="40:1"


LOG_SUFFIX="chr"$CHROM
RUNTIME="20:00"

# bsub -n 8 -q mpi -R span[hosts=1] -a openmp 
# bsub -x -q mpi -W ${RUNTIME} -R scratch\
#         -R span[hosts=1] \
#         -o ${OUTDIR}/${LOG_SUFFIX}.log \
#         -e ${OUTDIR}/${LOG_SUFFIX}.err \
        echo $ENV/bin/python $GXPRED --oxf ${OXF_FILE} \
                                --fam ${SAMPLE_FILE} \
                                --gtf ${GTF_FILE} \
                                --gx ${EXPR_FILE} \
                                --chr ${CHROM} \
                                --out ${OUTDIR} \
                                --outprefix ${OUTPREFIX} \
                                --batch-section ${BATCH_SECTION} \
                                --cmax ${Z} \
                                --outprefix ${OUTPREFIX} \
                                --dataset ${DATASET}
