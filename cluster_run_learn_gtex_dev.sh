#!/bin/bash

# create conda env with
# conda create --prefix /usr/users/fsimone/myenv numpy scipy sklearn


module load intel/compiler/64/2017/17.0.2
module load intel/mkl/64/2017/2.174


ENV='/usr/users/fsimone/myenv'
GXPRED='/usr/users/fsimone/gxpred/learn_gtex_memtest.py'
GTFOLDER="/usr/users/fsimone/datasets/gtex"
SAMPLEFILE="donor_ids.fam"
CHROM="1"
GTFILE="GTEx_450Indiv_genot_imput_info04_maf01_HWEp1E6_dbSNP135IDs_donorIDs_dosage_chr${CHROM}.gz"
LOG_SUFFIX="chr"$CHROM
EXPR="/usr/users/fsimone/datasets/gtex/gtex_wholeblood_normalized.expression.txt"
GTFPATH="/usr/users/fsimone/datasets/gtex/gencode.v19.annotation.gtf.gz"
OUTDIR="/cbscratch/franco/datasets/gtex_models/dev_models"
# QUEUE="mpi mpi2 mpi3_all hh sa"

FROM="0"
TO="1"
echo $FROM $TO
#echo bsub -n 8 -a openmp -q mpi -R span[hosts=1] -R cbscratch \
	 # -o ${OUTDIR}/dev.log -e ${OUTDIR}/dev.err
	 echo 		$ENV/bin/python $GXPRED \
			--gen ${GTFOLDER}/${GTFILE} --sample ${GTFOLDER}/${SAMPLEFILE} \
			--gtf ${GTFPATH} --chr ${CHROM} \
			--expr ${EXPR} --out ${OUTDIR} --from $FROM --to $TO
