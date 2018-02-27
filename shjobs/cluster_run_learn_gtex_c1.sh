#!/bin/bash

# create conda env with
# conda create --prefix /usr/users/fsimone/myenv numpy scipy sklearn


module load intel/compiler/64/2017/17.0.2
module load intel/mkl/64/2017/2.174


SPLIT=$1

if [ -z ${SPLIT} ]; 
	then echo "SPLIT is not set"; 
	exit; 
elif [ $SPLIT -lt 1 ]; 
	then echo "SPLIT is less than 1"; 
	exit;
fi

for CHROM in `seq 1 1`
do

# CHROM="1"

HOME="/usr/users/fsimone"
ENV="${HOME}/myenv"
GXPRED="${HOME}/gxpred/learn_gtex.py"
GTFOLDER="${HOME}/datasets/gtex"
SAMPLEFILE="donor_ids.fam"
GTFILE="GTEx_450Indiv_genot_imput_info04_maf01_HWEp1E6_dbSNP135IDs_donorIDs_dosage_chr${CHROM}.gz"
LOG_SUFFIX="chr"$CHROM
EXPR="${HOME}/datasets/gtex/gtex_wholeblood_normalized.expression.txt" #gtex_wholeblood_normalized.lm_corr_final2.exp.txt" #
GTFPATH="${HOME}/datasets/gtex/gencode.v19.annotation.gtf.gz"
OUTDIR="/scratch/fsimone/gtex_models/gxpred_models_c1_bslmm"
# QUEUE='mpi mpi2 mpi3_all hh sa'
RUNTIME="20:00"
Z="1"


for SECTION in `seq 0 $SPLIT`
	do
	echo "$CHROM $SECTION"
	# bsub -n 8 -q mpi -R span[hosts=1] -a openmp 
	bsub -x -q mpi -W ${RUNTIME} -R scratch\
			-R span[hosts=1] \
			-o ${OUTDIR}/section${SECTION}_${LOG_SUFFIX}.log \
			-e ${OUTDIR}/section${SECTION}_${LOG_SUFFIX}.err \
			$ENV/bin/python $GXPRED --gen ${GTFOLDER}/${GTFILE} \
									--sample ${GTFOLDER}/${SAMPLEFILE} \
							    	--gtf ${GTFPATH} \
							    	--chr ${CHROM} \
									--expr ${EXPR} \
									--out ${OUTDIR} \
									--split ${SPLIT} \
			 						--section ${SECTION} \
			 						--zmax ${Z}
done	

done