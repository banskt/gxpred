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

for CHROM in `seq 2 22`
do

# CHROM="1"

ENV='/usr/users/fsimone/myenv'
GXPRED='/usr/users/fsimone/gxpred/learn_gtex.py'
GTFOLDER="/usr/users/fsimone/datasets/gtex"
SAMPLEFILE="donor_ids.fam"
GTFILE="GTEx_450Indiv_genot_imput_info04_maf01_HWEp1E6_dbSNP135IDs_donorIDs_dosage_chr${CHROM}.gz"
LOG_SUFFIX="chr"$CHROM
EXPR="/usr/users/fsimone/datasets/gtex/gtex_wholeblood_normalized.lm_corr.exp.klinikum.txt"
GTFPATH="/usr/users/fsimone/datasets/gtex/gencode.v19.annotation.gtf.gz"
OUTDIR="/usr/users/fsimone/datasets/gtex/gxpred_models_klinikum"
# QUEUE='mpi mpi2 mpi3_all hh sa'
RUNTIME="12:00"


for SECTION in `seq 0 $SPLIT`
	do
	echo "$CHROM $SECTION"
	# bsub -n 8 -q mpi -R span[hosts=1] -a openmp 
	bsub -x -q mpi -W ${RUNTIME} \
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
			 						--section ${SECTION}
done	

done