#!/bin/bash

# create conda env with
# conda create --prefix /usr/users/fsimone/myenv numpy scipy sklearn


module load intel/compiler/64/2017/17.0.2
module load intel/mkl/64/2017/2.174



# $ENV/bin/python $PREDICT --gt ${MODELPATH}/${GTFILE} --model ${MODELPATH}/gxpred_models --sample ${MODELPATH}/${SAMPLEFILE} --chr ${CHROM} --out $OUTDIR

for CHROM in `seq 1 1`
do

	HOME='/usr/users/fsimone'
	ENV="${HOME}/myenv"
	PREDICT="${HOME}/gxpred/predict_gtex.py"
	MODELPATH="${HOME}/datasets/gtex"
	# CHROM="1"
	GTFILE="GTEx_450Indiv_genot_imput_info04_maf01_HWEp1E6_dbSNP135IDs_donorIDs_dosage_chr${CHROM}.gz"
	LOGDIR="${MODELPATH}/predictions_klinikum"
	OUTDIR="${MODELPATH}/predictions_klinikum/pred_chr${CHROM}"
	LOG_SUFFIX="chr"$CHROM
	SAMPLEFILE="donor_ids.fam"
	RUNTIME="12:00"
	DATASET="gtex"

	echo "$CHROM"
	# echo bsub -x -q mpi -W ${RUNTIME} \
	bsub -n 8 -q mpi -a openmp -W ${RUNTIME} \
			-R span[hosts=1] \
			-o ${LOGDIR}/prediction_${LOG_SUFFIX}.log \
			-e ${LOGDIR}/prediction_${LOG_SUFFIX}.err \
			$ENV/bin/python $PREDICT	--gt ${MODELPATH}/${GTFILE} \
										--model ${MODELPATH}/gxpred_models_klinikum \
										--sample ${MODELPATH}/${SAMPLEFILE} \
										--chr ${CHROM} --out $OUTDIR \
										--dataset ${DATASET}
done
