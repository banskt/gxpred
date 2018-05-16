#!/bin/bash

module load intel/compiler/64/2017/17.0.2
module load intel/mkl/64/2017/2.174


HOME='/usr/users/fsimone'
ENV="${HOME}/myenv"
LOG_SUFFIX="prediction_fullgenome"
GXPRED="${HOME}/gxpred_annots/gxpred/predict_with_annotations.py"
DESC="fullgenome"
DIR="/cbscratch/franco/datasets/gtex_models/gxpred_models_with_annots/z1/${DESC}"
# QUEUE='mpi mpi2 mpi3_all hh sa'
RUNTIME="24:00"
CONFIG="config_fullgenome"


for CHROM in `seq 1 22`
do

for MODEL in `ls $DIR`
    do
    LOG_SUFFIX="prediction_fullgenome_${CHROM}"
    echo "$CHROM -- $MODEL"
    # bsub -n 8 -q mpi -R span[hosts=1] -a openmp 
    # bsub -x -q mpi -W ${RUNTIME} -R scratch\
    bsub -n 8 -a openmp -q mpi -W ${RUNTIME} -R span[hosts=1] -R cbscratch \
            -o ${DIR}/${MODEL}/${LOG_SUFFIX}.log \
            -e ${DIR}/${MODEL}/${LOG_SUFFIX}.err \
            $ENV/bin/python $GXPRED --dir ${DIR}/${MODEL} \
                                    --config ${CONFIG} \
                                    --chr ${CHROM}
done    

done
