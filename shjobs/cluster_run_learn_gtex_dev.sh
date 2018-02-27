#!/bin/bash

# create conda env with
# conda create --prefix /usr/users/fsimone/myenv numpy scipy sklearn


module load intel/compiler/64/2017/17.0.2
module load intel/mkl/64/2017/2.174

HOME='/usr/users/fsimone'
ENV="${HOME}/myenv"
GXPRED="${HOME}/gxpred/learn_gtex_memtest.py"
GTFOLDER="${HOME}/datasets/gtex"
SAMPLEFILE="donor_ids.fam"
CHROM="1"
GTFILE="GTEx_450Indiv_genot_imput_info04_maf01_HWEp1E6_dbSNP135IDs_donorIDs_dosage_chr${CHROM}.gz"
LOG_SUFFIX="chr"$CHROM
EXPR="${HOME}/datasets/gtex/gtex_wholeblood_normalized.expression.txt" # gtex_wholeblood_normalized.lm_corr_final2.exp.txt
GTFPATH="${HOME}/datasets/gtex/gencode.v19.annotation.gtf.gz"
OUTDIR="/cbscratch/franco/datasets/gtex_models/dev_models"
# QUEUE="mpi mpi2 mpi3_all hh sa"

FROM="0"
TO="10"
echo $FROM $TO
bsub -n 8 -a openmp -q mpi -R span[hosts=1] -R cbscratch \
	 -o ${OUTDIR}/dev.log -e ${OUTDIR}/dev.err \
	 		$ENV/bin/python $GXPRED \
			--gen ${GTFOLDER}/${GTFILE} --sample ${GTFOLDER}/${SAMPLEFILE} \
			--gtf ${GTFPATH} --chr ${CHROM} \
			--expr ${EXPR} --out ${OUTDIR} --from $FROM --to $TO
